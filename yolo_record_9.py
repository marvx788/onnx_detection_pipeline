#!/usr/bin/env python3
"""
YOLO Detection Recorder & Streamer — Sequential processing
Appsink → External AI (OpenVINO) → Appsrc → Tee → [FileSink & HTTPServer]

No more X11/cv2.imshow dependency. High stability over SSH.
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
DEFAULT_VIDEO  = f"{BASE_DIR}/test_video_3.mp4"
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

# ── MJPEG Stream Server Variables ────────────────────────────────────────────
latest_jpeg_frame = None
jpeg_lock = threading.Lock()

class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_jpeg_frame
        if self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with jpeg_lock:
                        frame = latest_jpeg_frame
                    if frame is not None:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.04) # ~25 FPS
            except Exception as e:
                pass
        else:
            self.send_response(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

# ─────────────────────────────────────────────────────────────────────────────
#  AI & Tracker
# ─────────────────────────────────────────────────────────────────────────────
class YoloInference:
    def __init__(self, model_path):
        ie = Core()
        model = ie.read_model(model_path)
        available = ie.available_devices
        device = "GPU" if "GPU" in available else "CPU"
        compiled = ie.compile_model(model, device_name=device, config={"CACHE_DIR": "/tmp/ov_cache"})
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
    ids = stats["frame_ids"]
    if not ids: return
    graph_path = os.path.splitext(output_path)[0] + "_fps_graph.png"
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("FPS Over Time — GStreamer Distributed", fontweight="bold", fontsize=13)
    series = [(stats["ai_fps_hist"], "AI FPS", axes[0]), (stats["pipe_fps_hist"], "Pipeline FPS", axes[1]), (stats["disp_fps_hist"], "Processing FPS", axes[2])]
    for data, title, ax in series:
        ax.plot(ids, data, color="tab:blue", linewidth=0.8)
        ax.axhline(source_fps, color="gold", linewidth=1.5, linestyle="--", label=f"Source ({source_fps} FPS)")
        ax.set_title(title, fontsize=10); ax.set_ylabel("FPS"); ax.set_ylim(bottom=0); ax.legend(loc="lower left", fontsize=8); ax.grid(True, alpha=0.3)
    axes[2].set_xlabel("Frame ID")
    plt.tight_layout(); plt.savefig(graph_path, dpi=150, bbox_inches="tight"); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  Main Process Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run(video_path, model_path, output_path):
    global latest_jpeg_frame
    width, height, fps, total = get_video_info(video_path)
    frame_duration_ns = int(round(Gst.SECOND / fps))

    print("=" * 60)
    print("  YOLO — Sequential processing & Network Streaming")
    print("=" * 60)
    print(f"  Input      : {video_path}")
    print(f"  Output     : {output_path}")
    print(f"  Live Stream: http://localhost:{STREAM_PORT}/stream.mjpg")
    print("=" * 60)

    ai = YoloInference(model_path)
    tracker = HysteresisTracker()

    # Lance le micro-serveur Web pour diffuser la vidéo
    server_address = ('', STREAM_PORT)
    server = StreamingServer(server_address, StreamingHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    result_map, result_lock, result_evt = {}, threading.Lock(), threading.Event()
    infer_q, encode_q = queue.Queue(maxsize=4), queue.Queue(maxsize=8)
    src_eos, stop_evt, dst_eos_evt = threading.Event(), threading.Event(), threading.Event()

    stats = {"written": 0, "sync_hits": 0, "sync_miss": 0, "t0": None, "frame_ids": [], "ai_fps_hist": [], "pipe_fps_hist": [], "disp_fps_hist": [], "_last_pipe_t": None, "_last_disp_t": None}
    pipe_fps_window, disp_fps_window = deque(maxlen=SMOOTH_WIN), deque(maxlen=SMOOTH_WIN)

    # Destination pipeline (Enregistrement fichier uniquement)
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

    # Source Pipeline
    frame_idx_ctr = [0]
    display_q = queue.Queue()
    src_pipeline = Gst.parse_launch(
        f"filesrc location={video_path} ! decodebin name=dec "
        f"dec. ! queue ! audioconvert ! fakesink "
        f"dec. ! queue ! videoconvert ! videoscale ! video/x-raw,format=BGR,width={width},height={height} ! "
        f"appsink name=sink emit-signals=true sync=true max-buffers=30 drop=false"
    )
    appsink = src_pipeline.get_by_name("sink")

    def on_frame(sink):
        sample = sink.emit("pull-sample")
        if not sample: return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if ok:
            frame = np.frombuffer(info.data, dtype=np.uint8).reshape(height, width, 3).copy()
            buf.unmap(info)
            fidx = frame_idx_ctr[0]; frame_idx_ctr[0] += 1
            display_q.put((frame, fidx))
            infer_q.put((frame.copy(), fidx))
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
        global latest_jpeg_frame
        out_idx, last_dets = 0, []
        while True:
            item = display_q.get()
            if item is None: encode_q.put(None); break
            frame, fidx = item

            deadline = time.perf_counter() + SYNC_TIMEOUT_MS / 1000.0
            dets = None
            while time.perf_counter() < deadline:
                with result_lock:
                    if fidx in result_map:
                        dets = result_map.pop(fidx); break
                result_evt.wait(timeout=0.005); result_evt.clear()

            if dets is not None: last_dets = tracker.update(dets); stats["sync_hits"] += 1
            else: stats["sync_miss"] += 1

            ai.draw(frame, last_dets)

            now = time.perf_counter()
            if stats["_last_pipe_t"] is not None: pipe_fps_window.append(1.0 / max(1e-9, now - stats["_last_pipe_t"]))
            stats["_last_pipe_t"] = now
            pipe_fps = sum(pipe_fps_window) / len(pipe_fps_window) if pipe_fps_window else 0.0

            ai_fps_osd = ai.avg_fps()
            sync_r = stats["sync_hits"] / max(1, stats["sync_hits"] + stats["sync_miss"])
            txt = f"AI: {ai_fps_osd:.1f} fps  sync:{sync_r*100:.0f}%"
            cv2.putText(frame, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
            cv2.putText(frame, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,80), 2)

            # ── ENVOI FLUX RESEAU (MJPEG) ────────────────────────────────────
            # On redimensionne l'envoi réseau en 960x540 pour consommer moins de bande passante
            small_frame = cv2.resize(frame, (960, 540))
            _, jpeg_buf = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with jpeg_lock:
                latest_jpeg_frame = jpeg_buf.tobytes()

            encode_q.put((frame.tobytes(), out_idx))
            stats["written"] += 1; out_idx += 1

            disp_now = time.perf_counter()
            if stats["_last_disp_t"] is not None: disp_fps_window.append(1.0 / max(1e-9, disp_now - stats["_last_disp_t"]))
            stats["_last_disp_t"] = disp_now
            disp_fps = sum(disp_fps_window) / len(disp_fps_window) if disp_fps_window else 0.0

            stats["frame_ids"].append(stats["written"])
            stats["ai_fps_hist"].append(ai.recent_fps())
            stats["pipe_fps_hist"].append(pipe_fps)
            stats["disp_fps_hist"].append(disp_fps)

            elapsed = time.monotonic() - stats["t0"]
            pct = stats["written"] / total * 100 if total > 0 else 0
            print(f"\r  Processing: {pct:5.1f}% ({stats['written']}/{total}) | AI:{ai_fps_osd:.1f}fps", end="", flush=True)

        stats["disp_done"].set()

    stats["disp_done"] = threading.Event()
    stats["t0"] = time.monotonic()

    dst_pipeline.set_state(Gst.State.PLAYING)
    src_pipeline.set_state(Gst.State.PLAYING)

    threading.Thread(target=ai_thread, daemon=True).start()
    threading.Thread(target=processing_thread, daemon=True).start()
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

    print(f"\n\n  Terminé ! Fichier écrit → {output_path}")
    save_fps_graph(stats, output_path, fps)

if __name__ == "__main__":
    run(DEFAULT_VIDEO, DEFAULT_MODEL, DEFAULT_OUTPUT)