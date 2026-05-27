#!/usr/bin/env python3
"""
YOLO Detection Recorder — sequential processing (like the C++ script)
Appsink → external AI (ONNX Runtime) → Appsrc

Each frame waits for its own detection before being written.
This is possible because the AI (49fps) is faster than the source (25fps).
Boxes are perfectly synchronized with objects, no risk of mismatch or drift.

If the AI becomes slower than the source, an automatic fallback uses
the last known detection plus anti-flicker hysteresis.

Encode thread is separated from display thread to avoid push-buffer blocking.

"""

import sys
import os
import time
import threading
import queue
import subprocess
from collections import deque
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
    import numpy as np
except ImportError:
    print("[ERREUR] pip install openvino"); sys.exit(1)

# === CONFIGURATION ===
BASE_DIR       = "/home/marvin/Documents/onnx_detection/onnx_detection_gpu"
DEFAULT_VIDEO  = f"{BASE_DIR}/build/test-video_2.mp4"
DEFAULT_MODEL  = f"{BASE_DIR}/yolo11n.onnx"
DEFAULT_OUTPUT = f"{BASE_DIR}/output_vino.mp4"
PLUGIN_PATH    = f"{BASE_DIR}/gst-template/builddir/gst-plugin"
ONNX_LIB_PATH  = "/opt/onnxruntime/lib"

INPUT_SIZE  = 640
H264_QP     = 20

# ── Anti-flicker thresholds ──────────────────────────────────────────────────
CONF_APPEAR        = 0.45
CONF_KEEP          = 0.25
NMS_THRESHOLD      = 0.45
MAX_MISSING_FRAMES = 3

# ── Timeout for sequential mode ─────────────────────────────────────────────
SYNC_TIMEOUT_MS = 80   # ~2 frames at 25fps

# ── Lissage graphique ────────────────────────────────────────────────────────
SMOOTH_WIN = 30   # fenêtre glissante en frames (~1s à 25fps)

os.environ["GST_PLUGIN_PATH"] = PLUGIN_PATH
os.environ["LD_LIBRARY_PATH"] = ONNX_LIB_PATH + ":" + os.environ.get("LD_LIBRARY_PATH", "")

Gst.init(None)

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


# ─────────────────────────────────────────────────────────────────────────────
#  AI
# ─────────────────────────────────────────────────────────────────────────────
class YoloInference:
    def __init__(self, model_path):
        ie = Core()
        model = ie.read_model(model_path)
        device = get_best_device()
        compiled = ie.compile_model(model, device_name=device,
            config={"CACHE_DIR": "/tmp/ov_cache"})
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
        tensor  = np.expand_dims(
            np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)), 0)
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
            if conf < CONF_KEEP:
                continue
            cx, cy, bw, bh = preds[0,i], preds[1,i], preds[2,i], preds[3,i]
            boxes.append([int((cx-bw/2)*xs), int((cy-bh/2)*ys),
                          int(bw*xs), int(bh*ys)])
            confs.append(conf)
            ids.append(cls_id)
        if not boxes:
            return []
        idx  = cv2.dnn.NMSBoxes(boxes, confs, CONF_KEEP, NMS_THRESHOLD)
        flat = idx.flatten() if hasattr(idx, 'flatten') else idx
        return [(boxes[i], confs[i], ids[i]) for i in flat]

    def draw(self, frame, detections):
        for (x,y,w,h), conf, cls in detections:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            label = f"{COCO_CLASSES[cls]} {conf:.2f}"
            sz, bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x,y-sz[1]-bl), (x+sz[0],y), (0,255,0), -1)
            cv2.putText(frame, label, (x,y-bl),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    def avg_fps(self):
        """FPS moyen global (OSD vidéo)."""
        with self._lock:
            return 1000.0 / (self._sum_ms / self._count) if self._count > 0 else 0.0

    def recent_fps(self):
        """FPS lissé sur fenêtre glissante (graphique)."""
        with self._lock:
            if not self._recent_ms:
                return 0.0
            return 1000.0 / (sum(self._recent_ms) / len(self._recent_ms))

    def count(self):
        with self._lock:
            return self._count


# ─────────────────────────────────────────────────────────────────────────────
#  Tracker hysteresis
# ─────────────────────────────────────────────────────────────────────────────
def iou(b1, b2):
    ix1=max(b1[0],b2[0]); iy1=max(b1[1],b2[1])
    ix2=min(b1[2],b2[2]); iy2=min(b1[3],b2[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter==0: return 0.0
    return inter/((b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter+1e-6)

class HysteresisTracker:
    def __init__(self):
        self._tracks = []

    def update(self, raw_dets):
        strong = [d for d in raw_dets if d[1] >= CONF_APPEAR]
        weak   = [d for d in raw_dets if CONF_KEEP <= d[1] < CONF_APPEAR]
        updated  = []
        used     = set()

        for tr in self._tracks:
            best_iou = 0.3; best_i = -1; best_src = None
            for src in [strong, weak]:
                for i, ((x,y,w,h), conf, cls) in enumerate(src):
                    if cls != tr["cls"] or (id(src),i) in used: continue
                    b = [x,y,x+w,y+h]
                    s = iou(b, tr["box"])
                    if s > best_iou: best_iou=s; best_i=i; best_src=src
            if best_i >= 0:
                (x,y,w,h),conf,cls = best_src[best_i]
                updated.append({"box":[x,y,x+w,y+h],"conf":conf,"cls":cls,"missing":0})
                used.add((id(best_src), best_i))
            else:
                tr["missing"] += 1
                if tr["missing"] < MAX_MISSING_FRAMES:
                    updated.append(tr)

        for i, ((x,y,w,h), conf, cls) in enumerate(strong):
            if (id(strong),i) not in used:
                already = any(iou([x,y,x+w,y+h], t["box"])>0.3 and t["cls"]==cls
                              for t in updated)
                if not already:
                    updated.append({"box":[x,y,x+w,y+h],"conf":conf,"cls":cls,"missing":0})
        self._tracks = updated
        return [((t["box"][0], t["box"][1],
                  t["box"][2]-t["box"][0], t["box"][3]-t["box"][1]),
                 t["conf"], t["cls"]) for t in updated]


# ─────────────────────────────────────────────────────────────────────────────
#  Info video
# ─────────────────────────────────────────────────────────────────────────────
def get_video_info(path):
    try:
        r = subprocess.run(
            ["ffprobe","-v","error","-select_streams","v:0",
             "-show_entries","stream=width,height,r_frame_rate,nb_frames",
             "-of","csv=p=0", path],
            capture_output=True, text=True, timeout=10)
        parts    = r.stdout.strip().split(",")
        w, h     = int(parts[0]), int(parts[1])
        num, den = parts[2].split("/")
        fps      = round(int(num) / int(den))
        total    = int(parts[3]) if len(parts)>3 and parts[3].strip().isdigit() else 0
        return w, h, fps, total
    except Exception as e:
        print(f"[WARN] fallback ({e})")
        return 768, 432, 12, 0


# ─────────────────────────────────────────────────────────────────────────────
#  Info device
# ─────────────────────────────────────────────────────────────────────────────
def get_best_device():
    ie = Core()
    available = ie.available_devices
    print(f"  Devices disponibles : {available}")
    priority = ["GPU", "CPU"]
    for device in priority:
        if device in available:
            print(f"  Device sélectionné  : {device}")
            return device
    return "CPU"


# ─────────────────────────────────────────────────────────────────────────────
#  Graphique FPS
# ─────────────────────────────────────────────────────────────────────────────
def save_fps_graph(stats, output_path, source_fps):
    ids = stats["frame_ids"]
    if not ids:
        print("  [WARN] Aucune donnée FPS collectée — graphique ignoré")
        return

    graph_path = os.path.splitext(output_path)[0] + "_fps_graph.png"

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("FPS Over Time — GStreamer (Single-Thread)",
                 fontweight="bold", fontsize=13)

    series = [
        (stats["ai_fps_hist"],   "AI FPS",       axes[0]),
        (stats["pipe_fps_hist"], "Pipeline FPS", axes[1]),
        (stats["disp_fps_hist"], "Display FPS",  axes[2]),
    ]

    for data, title, ax in series:
        ax.plot(ids, data, color="tab:red", linewidth=0.8)
        ax.axhline(source_fps, color="gold", linewidth=1.5,
                   linestyle="--", label=f"Source ({source_fps} FPS)")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("FPS")
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel("Frame ID")
    plt.tight_layout()
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graphique FPS  : {graph_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run(video_path, model_path, output_path):
    width, height, fps, total = get_video_info(video_path)

    frame_duration_ns = int(round(Gst.SECOND / fps))

    print("=" * 60)
    print("  YOLO — Traitement sequentiel (boites synchrones)")
    print("=" * 60)
    print(f"  Entree      : {video_path}")
    print(f"  Modele      : {model_path}")
    print(f"  Sortie      : {output_path}")
    print(f"  Dims        : {width}x{height} @ {fps}fps  ({total} frames)")
    print(f"  Timeout sync: {SYNC_TIMEOUT_MS}ms (fallback si IA trop lente)")
    print(f"  Lissage FPS : fenêtre de {SMOOTH_WIN} frames")
    print("=" * 60)
    print("  Chargement modele ONNX...")
    ai      = YoloInference(model_path)
    tracker = HysteresisTracker()

    # ── AI result map indexed by frame_id ─────────────────────────────────
    result_map  = {}
    result_lock = threading.Lock()
    result_evt  = threading.Event()

    # ── Queues ───────────────────────────────────────────────────────────────
    infer_q  = queue.Queue(maxsize=4)
    # encode_q découple display_thread du push-buffer bloquant de x264enc
    encode_q = queue.Queue(maxsize=8)
    src_eos  = threading.Event()
    stop_evt = threading.Event()
    stats    = {
        "written":       0,
        "sync_hits":     0,
        "sync_miss":     0,
        "t0":            None,
        "frame_ids":     [],
        "ai_fps_hist":   [],
        "pipe_fps_hist": [],
        "disp_fps_hist": [],
        "_last_pipe_t":  None,
        "_last_disp_t":  None,
    }

    # Fenêtres glissantes pour lissage FPS
    pipe_fps_window = deque(maxlen=SMOOTH_WIN)
    disp_fps_window = deque(maxlen=SMOOTH_WIN)

    # ── Destination pipeline EOS ─────────────────────────────────────────────
    dst_eos_evt = threading.Event()

    # ── Destination pipeline ─────────────────────────────────────────────────
    dst_pipeline = Gst.parse_launch(
        "appsrc name=src format=time is-live=false ! "
        "videoconvert ! "
        f"x264enc speed-preset=fast quantizer={H264_QP} ! "
        "h264parse ! mp4mux faststart=true ! "
        f"filesink location={output_path} sync=false"
    )
    appsrc = dst_pipeline.get_by_name("src")
    appsrc.set_property("caps", Gst.Caps.from_string(
        f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1"
    ))

    dst_bus = dst_pipeline.get_bus()
    dst_bus.add_signal_watch()
    def on_dst_msg(bus, msg):
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"\n[Dest] ERREUR: {err.message}\n{dbg}")
            dst_eos_evt.set()
        elif msg.type == Gst.MessageType.EOS:
            print("\n  [Dest] EOS recu — mp4mux finalise, fichier OK")
            dst_eos_evt.set()
    dst_bus.connect("message", on_dst_msg)

    # ── Pipeline source ──────────────────────────────────────────────────────
    frame_idx_ctr = [0]
    display_q     = queue.Queue()

    src_pipeline = Gst.parse_launch(
        f"filesrc location={video_path} ! "
        f"decodebin name=dec "
        f"dec. ! queue ! audioconvert ! fakesink "
        f"dec. ! queue ! videoconvert ! videoscale ! "
        f"video/x-raw,format=BGR,width={width},height={height} ! "
        f"appsink name=sink emit-signals=true sync=true "
        f"max-buffers=30 drop=false"
    )
    appsink = src_pipeline.get_by_name("sink")

    def on_frame(sink):
        sample = sink.emit("pull-sample")
        if not sample: return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if ok:
            frame = np.frombuffer(info.data, dtype=np.uint8).reshape(
                height, width, 3).copy()
            buf.unmap(info)
            fidx = frame_idx_ctr[0]
            frame_idx_ctr[0] += 1
            display_q.put((frame, fidx))
            infer_q.put((frame.copy(), fidx))
        return Gst.FlowReturn.OK

    appsink.connect("new-sample", on_frame)
    src_bus = src_pipeline.get_bus()
    src_bus.add_signal_watch()
    def on_src_msg(bus, msg):
        if msg.type == Gst.MessageType.EOS:
            display_q.put(None)
            src_eos.set()
        elif msg.type == Gst.MessageType.ERROR:
            err,_ = msg.parse_error()
            print(f"\n[Source] {err.message}")
            display_q.put(None); src_eos.set()
    src_bus.connect("message", on_src_msg)

    # ── AI thread ─────────────────────────────────────────────────────────────
    def ai_thread():
        while not stop_evt.is_set():
            try:
                item = infer_q.get(timeout=1.0)
            except queue.Empty:
                if src_eos.is_set(): break
                continue
            if item is None: break
            frame, fidx = item
            dets = ai.infer(frame)
            with result_lock:
                result_map[fidx] = dets
                if len(result_map) > 10:
                    oldest = min(result_map.keys())
                    del result_map[oldest]
            result_evt.set()

    # ── Encode thread ─────────────────────────────────────────────────────────
    # Séparé de display_thread pour ne plus être bloqué par x264enc.
    # display_thread dépose (bytes, frame_idx) dans encode_q et continue
    # immédiatement ; encode_thread fait le push-buffer bloquant de son côté.
    encode_done = threading.Event()
    def encode_thread():
        while True:
            item = encode_q.get()
            if item is None:
                break
            data, idx = item

            # Log diagnostic : détecte si x264enc bloque trop longtemps
            t_push = time.perf_counter()
            pts = idx * frame_duration_ns
            buf = Gst.Buffer.new_wrapped(data)
            buf.pts      = pts
            buf.dts      = pts
            buf.duration = frame_duration_ns
            appsrc.emit("push-buffer", buf)
            push_ms = (time.perf_counter() - t_push) * 1000
            if push_ms > 40:
                print(f"\n  [WARN] push-buffer lent : {push_ms:.0f}ms (frame {idx})")

        appsrc.emit("end-of-stream")
        encode_done.set()

    # ── Display thread ────────────────────────────────────────────────────────
    display_done = threading.Event()
    def display_thread():
        out_idx   = 0
        last_dets = []

        while True:
            item = display_q.get()
            if item is None:
                # Fin du stream : signaler encode_thread puis sortir
                encode_q.put(None)
                break
            frame, fidx = item

            # Attente du résultat IA pour cette frame précise
            deadline = time.perf_counter() + SYNC_TIMEOUT_MS / 1000.0
            dets = None
            while time.perf_counter() < deadline:
                with result_lock:
                    if fidx in result_map:
                        dets = result_map.pop(fidx)
                        break
                result_evt.wait(timeout=0.005)
                result_evt.clear()

            if dets is not None:
                last_dets = tracker.update(dets)
                stats["sync_hits"] += 1
            else:
                stats["sync_miss"] += 1

            ai.draw(frame, last_dets)

            # ── Métriques Pipeline FPS (avant dépôt encode_q) ────────────────
            now = time.perf_counter()
            if stats["_last_pipe_t"] is not None:
                pipe_fps_window.append(1.0 / max(1e-9, now - stats["_last_pipe_t"]))
            stats["_last_pipe_t"] = now
            pipe_fps = sum(pipe_fps_window) / len(pipe_fps_window) if pipe_fps_window else 0.0

            # OSD sur la vidéo
            ai_fps_osd = ai.avg_fps()
            sync_r     = stats["sync_hits"] / max(1, stats["sync_hits"] + stats["sync_miss"])
            txt        = f"AI: {ai_fps_osd:.1f} fps  sync:{sync_r*100:.0f}%"
            cv2.putText(frame, txt, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
            cv2.putText(frame, txt, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,80), 2)

            # Dépôt non-bloquant dans encode_q → display_thread ne patiente
            # plus sur x264enc
            encode_q.put((frame.tobytes(), out_idx))

            stats["written"] += 1
            out_idx += 1

            # ── Métriques Display FPS (après dépôt encode_q) ─────────────────
            disp_now = time.perf_counter()
            if stats["_last_disp_t"] is not None:
                disp_fps_window.append(1.0 / max(1e-9, disp_now - stats["_last_disp_t"]))
            stats["_last_disp_t"] = disp_now
            disp_fps = sum(disp_fps_window) / len(disp_fps_window) if disp_fps_window else 0.0

            ai_fps_graph = ai.recent_fps()

            stats["frame_ids"].append(stats["written"])
            stats["ai_fps_hist"].append(ai_fps_graph)
            stats["pipe_fps_hist"].append(pipe_fps)
            stats["disp_fps_hist"].append(disp_fps)

            elapsed  = time.monotonic() - stats["t0"]
            pct      = stats["written"] / total * 100 if total > 0 else 0
            bar      = "#"*int(25*pct/100) + "-"*(25-int(25*pct/100))
            realtime = stats["written"] / fps
            speed    = realtime / elapsed if elapsed > 0 else 0
            print(f"\r  [{bar}] {pct:5.1f}%  {stats['written']}/{total}"
                  f"  |  AI:{ai_fps_osd:.1f}fps"
                  f"  sync:{sync_r*100:.0f}%"
                  f"  vitesse:{speed:.1f}x",
                  end="", flush=True)

        display_done.set()

    # ── Demarrage ─────────────────────────────────────────────────────────────
    print("  Traitement...\n")
    stats["t0"] = time.monotonic()

    dst_pipeline.set_state(Gst.State.PLAYING)
    src_pipeline.set_state(Gst.State.PLAYING)

    t_ai   = threading.Thread(target=ai_thread,      daemon=True)
    t_disp = threading.Thread(target=display_thread, daemon=True)
    t_enc  = threading.Thread(target=encode_thread,  daemon=True)
    t_ai.start()
    t_disp.start()
    t_enc.start()

    loop        = GLib.MainLoop()
    loop_thread = threading.Thread(target=loop.run, daemon=True)
    loop_thread.start()

    try:
        src_eos.wait()
        display_done.wait(timeout=60)
        encode_done.wait(timeout=60)
        print("\n  Attente finalisation mp4mux...")
        if not dst_eos_evt.wait(timeout=30):
            print("  [WARN] Timeout EOS destination — le fichier pourrait etre incomplet")

    except KeyboardInterrupt:
        print("\n  Interrompu.")
        display_q.put(None)

    stop_evt.set()
    loop.quit()

    time.sleep(0.5)

    src_pipeline.set_state(Gst.State.NULL)
    dst_pipeline.set_state(Gst.State.NULL)

    elapsed = time.monotonic() - stats["t0"]
    sync_r  = stats["sync_hits"] / max(1, stats["sync_hits"]+stats["sync_miss"])
    print(f"\n\n  Termine ! → {output_path}")
    print(f"  Frames ecrites  : {stats['written']}/{total}")
    print(f"  Frames IA       : {ai.count()}")
    print(f"  FPS IA moyen    : {ai.avg_fps():.1f}")
    print(f"  Sync reussies   : {stats['sync_hits']} ({sync_r*100:.0f}%)")
    print(f"  Fallback        : {stats['sync_miss']}")
    print(f"  Temps total     : {elapsed:.1f}s\n")

    save_fps_graph(stats, output_path, fps)

    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Batch mode
# ─────────────────────────────────────────────────────────────────────────────
def process_batch(video_list):
    for i, vpath in enumerate(video_list):
        if not os.path.exists(vpath): print(f"[SKIP] {vpath}"); continue
        base = os.path.splitext(os.path.basename(vpath))[0]
        out  = os.path.join(os.path.dirname(vpath), f"{base}_yolo.mp4")
        print(f"\n[{i+1}/{len(video_list)}] {os.path.basename(vpath)}")
        run(vpath, DEFAULT_MODEL, out)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 1 and all(a.endswith(".mp4") for a in args):
        process_batch(args)
    else:
        output = args[0] if args else DEFAULT_OUTPUT
        video  = args[1] if len(args) > 1 else DEFAULT_VIDEO
        if not os.path.exists(video):
            print(f"[ERREUR] Video introuvable : {video}"); sys.exit(1)
        if not os.path.exists(DEFAULT_MODEL):
            print(f"[ERREUR] Modele introuvable : {DEFAULT_MODEL}"); sys.exit(1)
        run(video, DEFAULT_MODEL, output)