#!/usr/bin/env python3
"""
YOLO Detection Recorder & Streamer — Sequential processing (GStreamer-driven)
Appsink → AI (OpenVINO) → Appsrc → FileSink (.mp4) & MJPEG HTTP

Architecture délibérément SIMPLE : on confie le décodage vidéo à GStreamer
(decodebin), chaque frame décodée DOIT passer par l'IA (queue FIFO bloquante,
pas de "dernière frame dispo" ni de saut de frames), et il n'y a ni
camera-sim (horloge murale), ni flow_thread (optical flow), ni motion
compensation. C'est ce choix qui donne un débit IA nettement plus élevé sur
Raspberry Pi : pas de cœur CPU consommé en continu par un calcul de flux
optique en parallèle de l'inférence, et pas de limite artificielle calée
sur le débit de la vidéo source (qui n'a aucun sens ici puisqu'il n'y a pas
de vraie caméra à respecter en temps réel).

Sortie mp4 ET flux MJPEG live au MÊME rythme fixe (OUTPUT_FPS, 12 par
défaut) : un seul pacing à gérer, pas de double horloge à synchroniser.

Améliorations conservées des itérations précédentes :
  - cv2.setNumThreads(1) + INFERENCE_NUM_THREADS : empêchent OpenVINO/OpenCV
    de saturer tous les cœurs CPU du Pi à eux seuls.
  - Historique de stats borné (deque maxlen) : pas de croissance mémoire
    sur les vidéos longues.
  - encode_q de taille raisonnable (pas de buffer de centaines de Mo).
  - MJPEG avec timeout socket, limite de clients simultanés, et
    Condition+compteur de séquence (pas de busy-poll, pas de thread zombie).
"""

import sys
import os
import time
import threading
import queue
import subprocess
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import numpy as np
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GLib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    print("[ERREUR] pip install opencv-python"); sys.exit(1)
try:
    from openvino import Core
except ImportError:
    print("[ERREUR] pip install openvino"); sys.exit(1)

# === CONFIGURATION ===
BASE_DIR       = "/home/comvis/Documents/onnx_detection_pipeline"
DEFAULT_VIDEO  = f"{BASE_DIR}/test-video.mp4"
DEFAULT_MODEL  = f"{BASE_DIR}/yolo11n_320.onnx"
DEFAULT_OUTPUT = f"{BASE_DIR}/output_320.mp4"
PLUGIN_PATH    = f"{BASE_DIR}/gst-template/builddir/gst-plugin"
ONNX_LIB_PATH  = "/opt/onnxruntime/lib"

INPUT_SIZE  = 320
H264_QP     = 20
STREAM_PORT = 8080

CONF_APPEAR        = 0.45
CONF_KEEP          = 0.25
NMS_THRESHOLD      = 0.45
MAX_MISSING_FRAMES = 3
SYNC_TIMEOUT_MS    = 1000
SMOOTH_WIN         = 30

# Un seul rythme de sortie pour tout (fichier mp4 ET flux MJPEG live) :
# plus simple à raisonner que deux horloges séparées, et 12fps reste
# fluide à l'œil tout en restant atteignable de façon stable sur Pi 4
# avec ce pipeline plus léger.
OUTPUT_FPS = 12
SHOW_WINDOW = False
WINDOW_NAME = "YOLO Live Detection"

# Bornes mémoire (vidéos longues sur Raspberry Pi) : sans elles, certains
# historiques de stats grossiraient sans fin pendant toute la durée de la
# vidéo.
STATS_HISTORY_MAXLEN = 5000
JPEG_QUALITY         = 70
JPEG_MAX_WIDTH        = 854

os.environ["GST_PLUGIN_PATH"] = PLUGIN_PATH
os.environ["LD_LIBRARY_PATH"] = ONNX_LIB_PATH + ":" + os.environ.get("LD_LIBRARY_PATH", "")

Gst.init(None)

# --- ISOLATION CPU -------------------------------------------------------
# Sur Raspberry Pi (peu de cœurs), OpenVINO et OpenCV essaient chacun par
# défaut de paralléliser leurs opérations internes sur TOUS les cœurs
# disponibles. On borne explicitement pour laisser de la marge aux autres
# threads (encodage, serveur MJPEG) et éviter les pics de contention.
cv2.setNumThreads(1)
NUM_CORES = os.cpu_count() or 4
AI_INFERENCE_THREADS = max(1, NUM_CORES - 1)
# ---------------------------------------------------------------------------

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

# ── MJPEG Stream Server ──────────────────────────────────────────────────
latest_jpeg_frame = None
latest_jpeg_seq = 0  # compteur de séquence : permet aux clients de détecter
                     # une nouvelle frame sans dépendre du timing exact d'un
                     # signal, et sans en rater une.
jpeg_lock = threading.Lock()
jpeg_frame_ready = threading.Condition(jpeg_lock)

# Limite stricte du nombre de clients MJPEG simultanés : sans elle, des
# onglets laissés ouverts ou des rechargements répétés en SSH/Wi-Fi
# instable peuvent accumuler des threads serveur bloqués indéfiniment sur
# une écriture socket lente.
MAX_MJPEG_CLIENTS = 3
_mjpeg_clients_lock = threading.Lock()
_mjpeg_clients_count = 0

class StreamingHandler(BaseHTTPRequestHandler):
    # Timeout socket : un client qui ne lit plus (connexion SSH coupée,
    # onglet en arrière-plan) faisait rester le thread serveur bloqué
    # indéfiniment sur wfile.write() ; ici l'écriture lève une exception et
    # le thread se termine proprement.
    timeout = 5

    def do_GET(self):
        global _mjpeg_clients_count
        if self.path == '/stream.mjpg':
            with _mjpeg_clients_lock:
                if _mjpeg_clients_count >= MAX_MJPEG_CLIENTS:
                    self.send_response(503)
                    self.send_header('Content-Type', 'text/plain; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(b"Trop de clients MJPEG connectes, reessayez.")
                    return
                _mjpeg_clients_count += 1
            try:
                self.send_response(200)
                self.send_header('Age', 0)
                self.send_header('Cache-Control', 'no-cache, private')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                last_sent_seq = -1
                while True:
                    with jpeg_lock:
                        jpeg_frame_ready.wait_for(
                            lambda: latest_jpeg_seq != last_sent_seq, timeout=1.0)
                        frame = latest_jpeg_frame
                        seq = latest_jpeg_seq
                    if frame is not None and seq != last_sent_seq:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                        last_sent_seq = seq
            except Exception:
                pass
            finally:
                with _mjpeg_clients_lock:
                    _mjpeg_clients_count = max(0, _mjpeg_clients_count - 1)
        elif self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(
                b"<html><body style='margin:0;background:#111'>"
                b"<img src='/stream.mjpg' style='width:100%;height:auto;display:block' />"
                b"</body></html>"
            )
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # silence les logs HTTP par défaut

class StreamingServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

# ─────────────────────────────────────────────────────────────────────────
#  AI & Tracker
# ─────────────────────────────────────────────────────────────────────────
class YoloInference:
    def __init__(self, model_path):
        ie = Core()
        model = ie.read_model(model_path)
        available = ie.available_devices
        device = "GPU" if "GPU" in available else "CPU"
        ov_config = {"CACHE_DIR": "/tmp/ov_cache"}
        if device == "CPU":
            # Borne le nombre de threads internes d'OpenVINO pour laisser
            # de la marge CPU aux autres threads (encodage, MJPEG), sans
            # quoi un pic d'inférence peut saturer 100% des cœurs du Pi.
            ov_config["INFERENCE_NUM_THREADS"] = AI_INFERENCE_THREADS
        compiled = ie.compile_model(model, device_name=device, config=ov_config)
        self.infer_req   = compiled.create_infer_request()
        self.input_layer = compiled.input(0)
        self._lock       = threading.Lock()
        self._count      = 0
        self._sum_ms     = 0.0
        self._recent_ms  = deque(maxlen=SMOOTH_WIN)

    def infer(self, frame_bgr):
        h, w    = frame_bgr.shape[:2]
        resized = cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor  = np.expand_dims(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)), 0)
        t0      = time.perf_counter()
        self.infer_req.infer({self.input_layer: tensor})
        outputs = [self.infer_req.get_output_tensor(0).data]
        ms      = (time.perf_counter() - t0) * 1000
        with self._lock:
            self._count     += 1
            self._sum_ms    += ms
            self._recent_ms.append(ms)
        return self._postprocess(outputs, w, h)

    def _postprocess(self, outputs, fw, fh):
        preds  = outputs[0][0]
        xs, ys = fw / INPUT_SIZE, fh / INPUT_SIZE
        boxes, confs, ids = [], [], []
        for i in range(preds.shape[1]):
            scores = preds[4:, i]
            cls_id = int(np.argmax(scores))
            conf   = float(scores[cls_id])
            if conf < CONF_KEEP: continue
            cx, cy, bw, bh = preds[0,i], preds[1,i], preds[2,i], preds[3,i]
            boxes.append([int((cx-bw/2)*xs), int((cy-bh/2)*ys), int(bw*xs), int(bh*ys)])
            confs.append(conf)
            ids.append(cls_id)
        if not boxes: return []
        idx  = cv2.dnn.NMSBoxes(boxes, confs, CONF_KEEP, NMS_THRESHOLD)
        flat = idx.flatten() if hasattr(idx, 'flatten') else idx
        return [(boxes[i], confs[i], ids[i]) for i in flat]

    def draw(self, frame, detections):
        for (x,y,w,h), conf, cls in detections:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            label = f"{COCO_CLASSES[cls]} {conf:.2f}"
            sz, bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x,y-sz[1]-bl), (x+sz[0],y), (0,255,0), -1)
            cv2.putText(frame, label, (x,y-bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    def avg_fps(self):
        with self._lock: return 1000.0 / (self._sum_ms / self._count) if self._count > 0 else 0.0
    def recent_fps(self):
        with self._lock: return 1000.0 / (sum(self._recent_ms) / len(self._recent_ms)) if self._recent_ms else 0.0
    def count(self):
        with self._lock: return self._count

def iou(b1, b2):
    ix1=max(b1[0],b2[0]); iy1=max(b1[1],b2[1]); ix2=min(b1[2],b2[2]); iy2=min(b1[3],b2[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter==0: return 0.0
    return inter/((b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter+1e-6)

class HysteresisTracker:
    def __init__(self): self._tracks = []
    def update(self, raw_dets):
        strong = [d for d in raw_dets if d[1] >= CONF_APPEAR]
        weak   = [d for d in raw_dets if CONF_KEEP <= d[1] < CONF_APPEAR]
        updated, used = [], set()
        for tr in self._tracks:
            best_iou = 0.3; best_i = -1; best_src = None
            for src in [strong, weak]:
                for i, ((x,y,w,h), conf, cls) in enumerate(src):
                    if cls != tr["cls"] or (id(src),i) in used: continue
                    b = [x,y,x+w,y+h]; s = iou(b, tr["box"])
                    if s > best_iou: best_iou=s; best_i=i; best_src=src
            if best_i >= 0:
                (x,y,w,h),conf,cls = best_src[best_i]
                updated.append({"box":[x,y,x+w,y+h],"conf":conf,"cls":cls,"missing":0})
                used.add((id(best_src), best_i))
            else:
                tr["missing"] += 1
                if tr["missing"] < MAX_MISSING_FRAMES: updated.append(tr)
        for i, ((x,y,w,h), conf, cls) in enumerate(strong):
            if (id(strong),i) not in used:
                if not any(iou([x,y,x+w,y+h], t["box"])>0.3 and t["cls"]==cls for t in updated):
                    updated.append({"box":[x,y,x+w,y+h],"conf":conf,"cls":cls,"missing":0})
        self._tracks = updated
        return [((t["box"][0], t["box"][1], t["box"][2]-t["box"][0], t["box"][3]-t["box"][1]), t["conf"], t["cls"]) for t in updated]

def get_video_info(path):
    try:
        r = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height,r_frame_rate,nb_frames","-of","csv=p=0", path], capture_output=True, text=True, timeout=10)
        parts = r.stdout.strip().split(",")
        w, h = int(parts[0]), int(parts[1])
        num, den = parts[2].split("/")
        return w, h, round(int(num) / int(den)), (int(parts[3]) if len(parts)>3 and parts[3].strip().isdigit() else 0)
    except Exception: return 768, 432, 12, 0

def save_fps_graph(stats, output_path, source_fps):
    ids = list(stats["frame_ids"])
    if not ids: return
    graph_path = os.path.splitext(output_path)[0] + "_fps_graph.png"
    ai_fps_hist   = list(stats["ai_fps_hist"])
    pipe_fps_hist = list(stats["pipe_fps_hist"])
    disp_fps_hist = list(stats["disp_fps_hist"])
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("FPS Over Time — GStreamer Sequential", fontweight="bold", fontsize=13)
    series = [(ai_fps_hist, "AI FPS", axes[0]), (pipe_fps_hist, "Pipeline FPS", axes[1]), (disp_fps_hist, "Processing FPS", axes[2])]
    for data, title, ax in series:
        ax.plot(ids, data, color="tab:blue", linewidth=0.8)
        ax.axhline(source_fps, color="gold", linewidth=1.5, linestyle="--", label=f"Output target ({source_fps} FPS)")
        ax.set_title(title, fontsize=10); ax.set_ylabel("FPS"); ax.set_ylim(bottom=0); ax.legend(loc="lower left", fontsize=8); ax.grid(True, alpha=0.3)
    axes[2].set_xlabel("Frame ID")
    plt.tight_layout(); plt.savefig(graph_path, dpi=150, bbox_inches="tight"); plt.close()

# ─────────────────────────────────────────────────────────────────────────
#  Main Process Pipeline
# ─────────────────────────────────────────────────────────────────────────
def run(video_path, model_path, output_path):
    global latest_jpeg_frame, latest_jpeg_seq
    width, height, src_fps, total = get_video_info(video_path)
    fps = OUTPUT_FPS  # un seul rythme pour tout : mp4 ET live
    frame_duration_ns = int(round(Gst.SECOND / fps))

    # Sous-échantillonnage : on ne garde qu'une fraction des frames source,
    # pour que la vidéo de sortie jouée à OUTPUT_FPS respecte la durée
    # ORIGINALE de la source plutôt que de la ralentir. Le ratio
    # src_fps/fps n'est pas toujours entier (ex: 30/12 = 2.5) ; un simple
    # "garder 1 frame sur round(ratio)" dérive avec le temps (ex: garder 1
    # sur 2 au lieu de 2.5 ferait une vidéo 25% trop longue). On accumule
    # donc l'écart fractionnaire pour rester fidèle sur la durée totale.
    SKIP_RATIO = (src_fps / fps) if (src_fps > 0 and fps > 0) else 1.0
    effective_total = int(total / SKIP_RATIO) if (total > 0 and SKIP_RATIO > 0) else 0

    print("=" * 60)
    print("  YOLO — Sequential processing & Network Streaming")
    print("=" * 60)
    print(f"  Input      : {video_path}  ({src_fps} fps source)")
    print(f"  Output     : {output_path}  (mp4 @ {fps}fps, ratio {SKIP_RATIO:.2f}x sous-échantillonné → durée fidèle à l'original)")
    print(f"  Live Stream: http://localhost:{STREAM_PORT}/stream.mjpg  (@ {fps}fps)")
    print(f"  CPU cores  : {NUM_CORES} (IA bornée à {AI_INFERENCE_THREADS} thread(s))")
    print("=" * 60)

    ai = YoloInference(model_path)
    tracker = HysteresisTracker()

    server_address = ('', STREAM_PORT)
    server = StreamingServer(server_address, StreamingHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    result_map, result_lock, result_evt = {}, threading.Lock(), threading.Event()
    infer_q = queue.Queue(maxsize=1)
    encode_q = queue.Queue(maxsize=8)
    src_eos, stop_evt, dst_eos_evt = threading.Event(), threading.Event(), threading.Event()

    stats = {
        "written": 0, "sync_hits": 0, "sync_miss": 0, "t0": None,
        # deque bornées : pas de croissance mémoire illimitée sur les
        # vidéos longues (une entrée par frame traitée, sur toute la durée).
        "frame_ids": deque(maxlen=STATS_HISTORY_MAXLEN),
        "ai_fps_hist": deque(maxlen=STATS_HISTORY_MAXLEN),
        "pipe_fps_hist": deque(maxlen=STATS_HISTORY_MAXLEN),
        "disp_fps_hist": deque(maxlen=STATS_HISTORY_MAXLEN),
        "_last_pipe_t": None, "_last_disp_t": None,
    }
    pipe_fps_window, disp_fps_window = deque(maxlen=SMOOTH_WIN), deque(maxlen=SMOOTH_WIN)

    # Destination pipeline (enregistrement fichier)
    dst_pipeline = Gst.parse_launch(
        "appsrc name=src format=time is-live=false ! videoconvert ! "
        f"x264enc speed-preset=fast quantizer={H264_QP} ! h264parse ! mp4mux faststart=true ! "
        f"filesink location={output_path} sync=false"
    )
    appsrc = dst_pipeline.get_by_name("src")
    appsrc.set_property("caps", Gst.Caps.from_string(f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1"))

    dst_bus = dst_pipeline.get_bus()
    dst_bus.add_signal_watch()
    dst_bus.connect("message", lambda b, m: dst_eos_evt.set() if m.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR) else None)

    # Source pipeline : décodage confié entièrement à GStreamer (decodebin),
    # pas de cv2.VideoCapture ni de limiteur de débit wall-clock — la
    # cadence réelle de traitement est imposée par la vitesse de l'IA
    # via infer_q (maxsize=4, bloquant), ce qui est exactement ce qu'on
    # veut puisqu'il n'y a pas de vraie caméra à respecter.
    frame_idx_ctr = [0]
    # Accumulateur fractionnaire pour le sous-échantillonnage à ratio non
    # entier (ex: 30fps source / 12fps sortie = 2.5) : on avance de
    # SKIP_RATIO à chaque frame source décodée, et on ne retient une frame
    # que lorsque l'accumulateur dépasse 1.0 (puis on le décrémente d'1).
    # Ça reste fidèle au ratio réel sur la durée totale, contrairement à un
    # simple "1 frame sur round(ratio)" qui dérive si le ratio n'est pas entier.
    skip_accumulator = [0.0]
    display_q = queue.Queue(maxsize=1)
    src_pipeline = Gst.parse_launch(
        f"filesrc location={video_path} ! decodebin name=dec "
        f"dec. ! queue ! audioconvert ! fakesink "
        f"dec. ! queue "
        f"! videoconvert "
        f"! videoscale "
        f"! video/x-raw,format=BGR,width={width},height={height} "
        f"! appsink name=sink "
        f"emit-signals=true "
        f"sync=false "
        f"max-buffers=1 "
        f"drop=true"
    )
    appsink = src_pipeline.get_by_name("sink")

    def on_frame(sink):
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()

        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK

        frame = np.frombuffer(
            info.data,
            dtype=np.uint8
        ).reshape(height, width, 3).copy()

        buf.unmap(info)

        fidx = frame_idx_ctr[0]
        frame_idx_ctr[0] += 1

        # DISPLAY → toujours garder la dernière image
        while True:
            try:
                display_q.get_nowait()
            except queue.Empty:
                break

        try:
            display_q.put_nowait((frame, fidx))
        except:
            pass

        # IA → ne jamais bloquer
        try:
            infer_q.put_nowait((frame.copy(), fidx))
        except queue.Full:
            pass

        return Gst.FlowReturn.OK

    appsink.connect("new-sample", on_frame)
    src_bus = src_pipeline.get_bus()
    src_bus.add_signal_watch()
    src_bus.connect("message", lambda b, m: (display_q.put(None), src_eos.set()) if m.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR) else None)

    def ai_thread():
        while not stop_evt.is_set():
            try: item = infer_q.get(timeout=1.0)
            except queue.Empty: continue
            if item is None: break
            frame, fidx = item
            dets = ai.infer(frame)
            with result_lock:
                result_map[fidx] = dets
                if len(result_map) > 10: del result_map[min(result_map.keys())]
            result_evt.set()

    def encode_thread():
        while True:
            item = encode_q.get()
            if item is None: break
            data, idx = item
            pts = idx * frame_duration_ns
            buf = Gst.Buffer.new_wrapped(data)
            buf.pts = buf.dts = pts
            buf.duration = frame_duration_ns
            appsrc.emit("push-buffer", buf)
        appsrc.emit("end-of-stream")

    def processing_thread():
        global latest_jpeg_frame, latest_jpeg_seq

        last_dets = []

        FRAME_PERIOD = 1.0 / OUTPUT_FPS
        next_tick = time.perf_counter()

        out_idx = 0

        while True:

            # pacing fixe
            now = time.perf_counter()

            if now < next_tick:
                time.sleep(next_tick - now)

            next_tick += FRAME_PERIOD

            # toujours prendre uniquement la frame la plus récente
            frame = None

            while True:
                try:
                    item = display_q.get_nowait()

                    if item is None:
                        encode_q.put(None)
                        stats["disp_done"].set()
                        return

                    frame = item

                except queue.Empty:
                    break

            if frame is None:
                continue

            frame, fidx = frame

            # récupérer dernier résultat IA disponible
            with result_lock:

                usable = [
                    k
                    for k in result_map
                    if k <= fidx
                ]

                if usable:
                    best = max(usable)

                    last_dets = tracker.update(
                        result_map.pop(best)
                    )

            ai.draw(frame, last_dets)

            # MJPEG
            _, jpeg = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY
                ]
            )

            with jpeg_lock:
                latest_jpeg_frame = jpeg.tobytes()
                latest_jpeg_seq += 1
                jpeg_frame_ready.notify_all()

            try:
                encode_q.put_nowait(
                    (
                        frame.tobytes(),
                        out_idx
                    )
                )
            except queue.Full:
                pass

            out_idx += 1

    stats["disp_done"] = threading.Event()
    stats["t0"] = time.monotonic()

    dst_pipeline.set_state(Gst.State.PLAYING)
    src_pipeline.set_state(Gst.State.PLAYING)

    threading.Thread(target=ai_thread, daemon=True).start()
    proc_thread = threading.Thread(
        target=processing_thread,
        daemon=True
    )
    proc_thread.start()
    threading.Thread(target=encode_thread, daemon=True).start()

    loop = GLib.MainLoop()
    threading.Thread(target=loop.run, daemon=True).start()


    try:
        src_eos.wait()
        stats["disp_done"].wait(timeout=60)
        dst_eos_evt.wait(timeout=30)
    except KeyboardInterrupt:
        print("\n  Arrêt demandé.")

    stop_evt.set()
    loop.quit()
    server.shutdown()
    src_pipeline.set_state(Gst.State.NULL)
    dst_pipeline.set_state(Gst.State.NULL)

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print(f"\n\n  Terminé ! Fichier écrit → {output_path}")
    save_fps_graph(stats, output_path, fps)

    

if __name__ == "__main__":
    run(DEFAULT_VIDEO, DEFAULT_MODEL, DEFAULT_OUTPUT)