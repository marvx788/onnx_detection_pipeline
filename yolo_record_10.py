#!/usr/bin/env python3
"""
YOLO Detection Recorder & Streamer — Sequential processing
Appsink → External AI (OpenVINO) → Appsrc → Tee → [FileSink & HTTPServer]

No more X11/cv2.imshow dependency. High stability over SSH.

[PROFILING] Each pipeline stage is measured in ms:
  - decode       : GStreamer decoding time (appsink callback)
  - infer        : AI inference time (OpenVINO)
  - postprocess  : YOLO post-processing (NMS, filtering)
  - track        : HysteresisTracker update
  - draw         : bounding box drawing
  - jpeg_encode  : JPEG encoding for network stream
  - h264_push    : buffer push to appsrc (H264)
  - frame_total  : total frame-to-frame time (processing_thread)

Output graphs (one file per graph):
  - <output>_graph_1_latency_timeline.png   : per-stage latency over time (smoothed)
  - <output>_graph_2_cdf.png                : CDF per stage (like percentile chart)
  - <output>_graph_3_stacked_budget.png     : stacked area — time budget per frame
  - <output>_graph_4_model_comparison.png   : stacked bar comparison across models/runs
  - <output>_fps_graph.png                  : FPS over time

For multi-model comparison, call save_comparison_graph() with a list of
(label, profiler) tuples after running multiple models.
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
import matplotlib.gridspec as gridspec

try:
    import cv2
except ImportError:
    print("[ERROR] pip install opencv-python"); sys.exit(1)
try:
    from openvino import Core
except ImportError:
    print("[ERROR] pip install openvino"); sys.exit(1)

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

# ── MJPEG Stream Server ───────────────────────────────────────────────────────
latest_jpeg_frame = None
jpeg_lock = threading.Lock()

class StreamingHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass  # Silence HTTP access logs

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
                    time.sleep(0.04)
            except Exception:
                pass
        else:
            self.send_response(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# ─────────────────────────────────────────────────────────────────────────────
#  Profiler — collects per-stage timing data
# ─────────────────────────────────────────────────────────────────────────────
class StageProfiler:
    """
    Collects stage durations (in ms) for each named pipeline stage.
    Thread-safe: one lock per stage.

    Usage:
        profiler = StageProfiler()
        t = time.perf_counter()
        ... do the stage work ...
        profiler.record("infer", t)   # t = start time

    Methods:
        profiler.record(stage, t_start)  → records (now - t_start) * 1000
        profiler.stats(stage)            → dict: mean, p50, p95, p99, max (ms)
        profiler.history(stage)          → list of recorded ms values
        profiler.stages()                → list of stage names with data
        profiler.summary()               → prints a summary table to stdout
    """
    # Stage display order + associated colors
    STAGE_META = {
        "decode":      {"color": "#4E9AF1", "label": "Decode (GStreamer)"},
        "preprocess":  {"color": "#A78BFA", "label": "Pre-process (CPU)"},
        "infer":       {"color": "#E05C5C", "label": "Inference (OpenVINO)"},
        "postprocess": {"color": "#F1A020", "label": "Post-process (YOLO)"},
        "track":       {"color": "#8B5CF6", "label": "Tracking (Hysteresis)"},
        "draw":        {"color": "#10B981", "label": "Draw boxes"},
        "jpeg_encode": {"color": "#06B6D4", "label": "JPEG encode (stream)"},
        "h264_push":   {"color": "#F59E0B", "label": "H264 push (filesink)"},
        "frame_total": {"color": "#EC4899", "label": "Total frame-to-frame"},
    }

    def __init__(self, maxlen=10000):
        self._data: dict[str, deque] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._maxlen = maxlen
        for stage in self.STAGE_META:
            self._data[stage] = deque(maxlen=maxlen)
            self._locks[stage] = threading.Lock()

    def record(self, stage: str, t_start: float):
        """Record the duration from t_start to now, in ms."""
        ms = (time.perf_counter() - t_start) * 1000.0
        if stage not in self._data:
            self._data[stage] = deque(maxlen=self._maxlen)
            self._locks[stage] = threading.Lock()
        with self._locks[stage]:
            self._data[stage].append(ms)

    def history(self, stage: str) -> list:
        with self._locks.get(stage, threading.Lock()):
            return list(self._data.get(stage, []))

    def stats(self, stage: str) -> dict:
        h = self.history(stage)
        if not h:
            return {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0, "n": 0}
        a = np.array(h)
        return {
            "mean": float(np.mean(a)),
            "p50":  float(np.percentile(a, 50)),
            "p95":  float(np.percentile(a, 95)),
            "p99":  float(np.percentile(a, 99)),
            "max":  float(np.max(a)),
            "n":    len(h),
        }

    def stages(self) -> list:
        return [s for s in self.STAGE_META if self._data.get(s)]

    def summary(self):
        print("\n" + "═" * 74)
        print("  PIPELINE PROFILING — Stage latencies")
        print("═" * 74)
        fmt = "  {:<28} {:>6}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}"
        print(fmt.format("Stage", "N", "avg ms", "p50", "p95", "p99", "max"))
        print("  " + "─" * 70)
        for stage in self.STAGE_META:
            s = self.stats(stage)
            if s["n"] == 0: continue
            label = self.STAGE_META[stage]["label"]
            print(fmt.format(label[:28], s["n"],
                             f"{s['mean']:.2f}", f"{s['p50']:.2f}",
                             f"{s['p95']:.2f}", f"{s['p99']:.2f}", f"{s['max']:.2f}"))
        print("═" * 74)


# ─────────────────────────────────────────────────────────────────────────────
#  AI & Tracker
# ─────────────────────────────────────────────────────────────────────────────
class YoloInference:
    def __init__(self, model_path, profiler: StageProfiler, input_size: int):
        self._profiler = profiler
        self.input_size = input_size
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
        t_preprocess = time.perf_counter()

        h, w    = frame_bgr.shape[:2]
        resized = cv2.resize(frame_bgr, (self.input_size, self.input_size))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor  = np.expand_dims(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)), 0)

        self._profiler.record("preprocess", t_preprocess)

        # [PROFILING] pure inference
        t_infer = time.perf_counter()
        self.infer_req.infer({self.input_layer: tensor})
        self._profiler.record("infer", t_infer)

        outputs = [self.infer_req.get_output_tensor(0).data]
        ms = (time.perf_counter() - t_infer) * 1000
        with self._lock:
            self._count     += 1
            self._sum_ms    += ms
            self._recent_ms.append(ms)

        # [PROFILING] post-processing (NMS, filtering)
        t_post = time.perf_counter()
        result = self._postprocess(outputs, w, h)
        self._profiler.record("postprocess", t_post)
        return result

    def _postprocess(self, outputs, fw, fh):
        preds  = outputs[0][0]
        xs, ys = fw / self.input_size, fh / self.input_size
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

    def draw(self, frame, detections, profiler: StageProfiler):
        # [PROFILING] bounding box drawing
        t_draw = time.perf_counter()
        for (x,y,w,h), conf, cls in detections:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            label = f"{COCO_CLASSES[cls]} {conf:.2f}"
            sz, bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x,y-sz[1]-bl), (x+sz[0],y), (0,255,0), -1)
            cv2.putText(frame, label, (x,y-bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        profiler.record("draw", t_draw)

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
        r = subprocess.run(
            ["ffprobe","-v","error","-select_streams","v:0",
             "-show_entries","stream=width,height,r_frame_rate,nb_frames",
             "-of","csv=p=0", path],
            capture_output=True, text=True, timeout=10)
        parts = r.stdout.strip().split(",")
        w, h = int(parts[0]), int(parts[1])
        num, den = parts[2].split("/")
        return w, h, round(int(num) / int(den)), (int(parts[3]) if len(parts)>3 and parts[3].strip().isdigit() else 0)
    except Exception:
        return 768, 432, 12, 0


# ─────────────────────────────────────────────────────────────────────────────
#  Shared graph helpers
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#0F1117"
AX_BG   = "#1A1D26"
GRID_C  = "#2A2D35"
TEXT_C  = "#C8C8D0"
TITLE_C = "#F0F0F0"


def _style_ax(ax, title: str, fontsize: int = 11):
    ax.set_facecolor(AX_BG)
    ax.set_title(title, color=TITLE_C, fontsize=fontsize, pad=8, fontweight="semibold")
    ax.tick_params(colors=TEXT_C, labelsize=8)
    ax.grid(True, color=GRID_C, linewidth=0.6, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)


def _smooth(data: list, w: int = 20) -> list:
    if len(data) < w:
        return data
    kernel = np.ones(w) / w
    return np.convolve(data, kernel, mode='valid').tolist()


def _base_path(output_path: str) -> str:
    return os.path.splitext(output_path)[0]


# ─────────────────────────────────────────────────────────────────────────────
#  Graph 0 — FPS over time (original, kept separate)
# ─────────────────────────────────────────────────────────────────────────────
def save_fps_graph(stats: dict, output_path: str, source_fps: int):
    ids = stats["frame_ids"]
    if not ids:
        return
    path = _base_path(output_path) + "_fps_graph.png"
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, facecolor=BG)
    fig.suptitle("FPS Over Time — YOLO Pipeline", fontweight="bold", fontsize=13, color=TITLE_C)
    series = [
        (stats["ai_fps_hist"],   "AI Inference FPS",  axes[0]),
        (stats["pipe_fps_hist"], "Pipeline FPS",       axes[1]),
        (stats["disp_fps_hist"], "Processing FPS",     axes[2]),
    ]
    for data, title, ax in series:
        _style_ax(ax, title)
        ax.plot(ids, data, color="#4E9AF1", linewidth=0.9)
        ax.axhline(source_fps, color="#F59E0B", linewidth=1.5, linestyle="--",
                   label=f"Source ({source_fps} FPS)")
        ax.set_ylabel("FPS", color=TEXT_C)
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower left", fontsize=8, facecolor=AX_BG,
                  edgecolor=GRID_C, labelcolor=TEXT_C)
    axes[2].set_xlabel("Frame ID", color=TEXT_C)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [Graph 0] FPS timeline         → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Graph 1 — Latency timeline per stage (smoothed)
# ─────────────────────────────────────────────────────────────────────────────
def save_graph_latency_timeline(profiler: "StageProfiler", output_path: str):
    """One line per pipeline stage, smoothed over 20 frames."""
    stages = [s for s in StageProfiler.STAGE_META if profiler.history(s)]
    if not stages:
        return
    path = _base_path(output_path) + "_graph_1_latency_timeline.png"

    histories = [profiler.history(s) for s in stages]
    n = min(len(h) for h in histories)
    histories = [h[:n] for h in histories]

    fig, ax = plt.subplots(figsize=(16, 6), facecolor=BG)
    _style_ax(ax, "Pipeline Stage Latency Over Time  (20-frame rolling average)")

    for stage, hist in zip(stages, histories):
        meta = StageProfiler.STAGE_META[stage]
        s = _smooth(hist)
        x = list(range(n - len(s), n))
        ax.plot(x, s, color=meta["color"], linewidth=1.3, label=meta["label"], alpha=0.92)

    ax.set_ylabel("Latency (ms)", color=TEXT_C)
    ax.set_xlabel("Frame", color=TEXT_C)
    ax.legend(loc="upper right", fontsize=8, ncol=2,
              facecolor=AX_BG, edgecolor=GRID_C, labelcolor=TEXT_C)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [Graph 1] Latency timeline       → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Graph 2 — CDF per stage  (one subplot per stage, same style as reference)
# ─────────────────────────────────────────────────────────────────────────────
def save_graph_cdf(profiler: "StageProfiler", output_path: str):
    """
    Cumulative Distribution Function for each stage.
    P50 / P95 / P99 are annotated as red dots, matching the reference style.
    """
    stages = [s for s in StageProfiler.STAGE_META if profiler.history(s)]
    if not stages:
        return
    path = _base_path(output_path) + "_graph_2_cdf.png"

    n_stages = len(stages)
    ncols = 3
    nrows = (n_stages + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), facecolor=BG)
    axes_flat = np.array(axes).flatten()

    fig.suptitle("CDF — Per-Stage Latency Distribution",
                 color=TITLE_C, fontsize=14, fontweight="bold", y=1.01)

    PERCENTILES = [50, 95, 99]

    for idx, stage in enumerate(stages):
        ax = axes_flat[idx]
        meta = StageProfiler.STAGE_META[stage]
        color = meta["color"]
        data  = np.array(profiler.history(stage))
        data_sorted = np.sort(data)
        cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted) * 100.0

        _style_ax(ax, meta["label"], fontsize=9)
        ax.plot(data_sorted, cdf, color=color, linewidth=2.0)

        # Horizontal reference lines at P50/P95/P99
        for pct in PERCENTILES:
            ax.axhline(pct, color=GRID_C, linewidth=0.8, linestyle="--")

        # Annotate percentile dots
        for pct in PERCENTILES:
            val = float(np.percentile(data, pct))
            ax.plot(val, pct, "o", color="#E05C5C", markersize=7, zorder=5)
            ax.annotate(f"P{pct}: {val:.1f}ms",
                        xy=(val, pct), xytext=(6, 4),
                        textcoords="offset points",
                        fontsize=8, color="#E05C5C", fontweight="bold")

        ax.set_ylim(0, 102)
        ax.set_xlabel("Latency (ms)", color=TEXT_C)
        ax.set_ylabel("Cumulative %", color=TEXT_C)

    # Hide empty subplots
    for idx in range(n_stages, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [Graph 2] CDF per stage          → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Graph 3 — Stacked area: time budget per frame
# ─────────────────────────────────────────────────────────────────────────────
def save_graph_stacked_budget(profiler: "StageProfiler", output_path: str):
    """
    Stacked area chart showing how the per-frame time budget is spent.
    frame_total is excluded (it would double-count everything).
    """
    stack_stages = [s for s in StageProfiler.STAGE_META
                    if s != "frame_total" and profiler.history(s)]
    if not stack_stages:
        return
    path = _base_path(output_path) + "_graph_3_stacked_budget.png"

    histories = [profiler.history(s) for s in stack_stages]
    n = min(len(h) for h in histories)
    smoothed = [_smooth(h[:n]) for h in histories]
    n_s = min(len(s) for s in smoothed)
    smoothed = [s[:n_s] for s in smoothed]

    colors = [StageProfiler.STAGE_META[s]["color"] for s in stack_stages]
    labels = [StageProfiler.STAGE_META[s]["label"] for s in stack_stages]
    x = list(range(n_s))

    fig, ax = plt.subplots(figsize=(16, 6), facecolor=BG)
    _style_ax(ax, "Per-Frame Time Budget — Stacked by Stage  (20-frame rolling average)")

    ax.stackplot(x, smoothed, colors=colors, labels=labels, alpha=0.85)

    # Overlay the frame_total line for reference
    if profiler.history("frame_total"):
        total_hist = profiler.history("frame_total")
        total_s = _smooth(total_hist[:n])[:n_s]
        ax.plot(list(range(len(total_s))), total_s,
                color="#FFFFFF", linewidth=1.2, linestyle="--",
                alpha=0.7, label="Total frame-to-frame (measured)")

    ax.set_ylabel("Cumulative time (ms)", color=TEXT_C)
    ax.set_xlabel("Frame", color=TEXT_C)
    ax.legend(loc="upper right", fontsize=7.5, ncol=2,
              facecolor=AX_BG, edgecolor=GRID_C, labelcolor=TEXT_C)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [Graph 3] Stacked time budget    → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Graph 4 — Multi-model stacked bar comparison
# ─────────────────────────────────────────────────────────────────────────────
def save_graph_model_comparison(runs: list[tuple[str, "StageProfiler"]], output_path: str):
    """
    Stacked bar chart comparing average stage latencies across multiple model runs.

    Args:
        runs: list of (model_label, profiler) — e.g. [("YOLO11n-320", p1), ("YOLO11n-640", p2)]
        output_path: base path for the output file

    This is designed to be called after running several models in sequence.
    Example:
        profiler_320 = run(video, model_320, output_320)
        profiler_640 = run(video, model_640, output_640)
        save_graph_model_comparison(
            [("YOLO11n-320", profiler_320), ("YOLO11n-640", profiler_640)],
            output_path
        )
    """
    if not runs:
        return
    path = _base_path(output_path) + "_graph_4_model_comparison.png"

    # Stages to compare (exclude frame_total — redundant with sum of parts)
    cmp_stages = [s for s in StageProfiler.STAGE_META if s != "frame_total"]

    fig, ax = plt.subplots(figsize=(max(8, 3.5 * len(runs)), 8), facecolor=BG)
    _style_ax(ax, "Average Timing Breakdown by Model — Stacked Bar (ms)")

    x_positions = np.arange(len(runs))
    bar_width   = 0.55

    bottoms = np.zeros(len(runs))

    for stage in cmp_stages:
        means = []
        for _label, profiler in runs:
            s = profiler.stats(stage)
            means.append(s["mean"])
        means = np.array(means)

        color = StageProfiler.STAGE_META[stage]["color"]
        label = StageProfiler.STAGE_META[stage]["label"]
        bars = ax.bar(x_positions, means, bar_width,
                      bottom=bottoms, color=color, label=label, alpha=0.88)

        # Annotate each bar segment with its value if large enough
        for i, (bar, val) in enumerate(zip(bars, means)):
            if val >= 1.0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bottoms[i] + val / 2,
                        f"{val:.1f}ms",
                        ha="center", va="center",
                        fontsize=8, color="#FFFFFF", fontweight="bold")
        bottoms += means

    # Annotate total on top of each bar
    for i, (total, (_lbl, _p)) in enumerate(zip(bottoms, runs)):
        ax.text(x_positions[i], total + 0.5, f"Total: {total:.1f}ms",
                ha="center", va="bottom", fontsize=9,
                color=TITLE_C, fontweight="bold")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([lbl for lbl, _ in runs], fontsize=11, color=TITLE_C)
    ax.set_ylabel("Average Latency (ms)", color=TEXT_C)
    ax.set_ylim(bottom=0, top=bottoms.max() * 1.15)
    ax.legend(loc="upper right", fontsize=8, ncol=1,
              facecolor=AX_BG, edgecolor=GRID_C, labelcolor=TEXT_C)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [Graph 4] Model comparison       → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: save all 4 timing graphs for a single run
# ─────────────────────────────────────────────────────────────────────────────
def save_all_timing_graphs(profiler: "StageProfiler", output_path: str,
                           model_label: str = ""):
    """
    Save graphs 1-3 for a single run.
    Graph 4 (comparison) requires multiple runs — call save_graph_model_comparison() manually.
    """
    label = model_label or os.path.basename(output_path)
    print(f"\n  Saving timing graphs for [{label}]:")
    save_graph_latency_timeline(profiler, output_path)
    save_graph_cdf(profiler, output_path)
    save_graph_stacked_budget(profiler, output_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Process Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run(video_path, model_path, output_path, input_size: int, model_label: str = "") -> "StageProfiler":
    """
    Run the full YOLO detection pipeline and return the StageProfiler so the
    caller can aggregate results across multiple models for graph 4.

    Example — comparing two models:
        p320 = run(video, model_320, out_320, model_label="YOLO11n-320")
        p640 = run(video, model_640, out_640, model_label="YOLO11n-640")
        save_graph_model_comparison(
            [("YOLO11n-320", p320), ("YOLO11n-640", p640)],
            out_320
        )
    """
    global latest_jpeg_frame
    width, height, fps, total = get_video_info(video_path)
    frame_duration_ns = int(round(Gst.SECOND / fps))

    tag = model_label or os.path.basename(model_path)
    print("=" * 60)
    print(f"  YOLO — Sequential processing & Network Streaming")
    print(f"  Model      : {tag}")
    print("=" * 60)
    print(f"  Input      : {video_path}")
    print(f"  Output     : {output_path}")
    print(f"  Live Stream: http://localhost:{STREAM_PORT}/stream.mjpg")
    print("=" * 60)

    # [PROFILING] Global profiler shared across threads
    profiler = StageProfiler()

    ai = YoloInference(model_path, profiler, input_size)
    tracker = HysteresisTracker()

    server_address = ('', STREAM_PORT)
    server = StreamingServer(server_address, StreamingHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    result_map, result_lock, result_evt = {}, threading.Lock(), threading.Event()
    infer_q, encode_q = queue.Queue(maxsize=4), queue.Queue(maxsize=8)
    src_eos, stop_evt, dst_eos_evt = threading.Event(), threading.Event(), threading.Event()

    stats = {
        "written": 0, "sync_hits": 0, "sync_miss": 0, "t0": None,
        "frame_ids": [], "ai_fps_hist": [], "pipe_fps_hist": [], "disp_fps_hist": [],
        "_last_pipe_t": None, "_last_disp_t": None,
    }
    pipe_fps_window, disp_fps_window = deque(maxlen=SMOOTH_WIN), deque(maxlen=SMOOTH_WIN)

    # Destination pipeline
    dst_pipeline = Gst.parse_launch(
        "appsrc name=src format=time is-live=false ! videoconvert ! "
        f"x264enc speed-preset=fast quantizer={H264_QP} ! h264parse ! mp4mux faststart=true ! "
        f"filesink location={output_path} sync=false"
    )
    appsrc = dst_pipeline.get_by_name("src")
    appsrc.set_property("caps", Gst.Caps.from_string(
        f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1"))

    dst_bus = dst_pipeline.get_bus()
    dst_bus.add_signal_watch()
    dst_bus.connect("message", lambda b, m:
        dst_eos_evt.set() if m.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR) else None)

    # Source Pipeline
    frame_idx_ctr = [0]
    display_q = queue.Queue()
    src_pipeline = Gst.parse_launch(
        f"filesrc location={video_path} ! decodebin name=dec "
        f"dec. ! queue ! audioconvert ! fakesink "
        f"dec. ! queue ! videoconvert ! videoscale ! "
        f"video/x-raw,format=BGR,width={width},height={height} ! "
        f"appsink name=sink emit-signals=true sync=true max-buffers=30 drop=false"
    )
    appsink = src_pipeline.get_by_name("sink")

    def on_frame(sink):
        # [PROFILING] Decode — measured from the start of the appsink callback
        t_decode = time.perf_counter()
        sample = sink.emit("pull-sample")
        if not sample: return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if ok:
            frame = np.frombuffer(info.data, dtype=np.uint8).reshape(height, width, 3).copy()
            buf.unmap(info)
            fidx = frame_idx_ctr[0]; frame_idx_ctr[0] += 1
            profiler.record("decode", t_decode)
            display_q.put((frame, fidx))
            infer_q.put((frame.copy(), fidx))
        return Gst.FlowReturn.OK

    appsink.connect("new-sample", on_frame)
    src_bus = src_pipeline.get_bus()
    src_bus.add_signal_watch()
    src_bus.connect("message", lambda b, m:
        (display_q.put(None), src_eos.set())
        if m.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR) else None)

    def ai_thread():
        while not stop_evt.is_set():
            try: item = infer_q.get(timeout=1.0)
            except queue.Empty: continue
            if item is None: break
            frame, fidx = item
            dets = ai.infer(frame)   # profiling for infer + postprocess done inside YoloInference
            with result_lock:
                result_map[fidx] = dets
                if len(result_map) > 10: del result_map[min(result_map.keys())]
            result_evt.set()

    def encode_thread():
        while True:
            item = encode_q.get()
            if item is None: break
            data, idx = item
            # [PROFILING] H264 buffer push
            t_push = time.perf_counter()
            pts = idx * frame_duration_ns
            buf = Gst.Buffer.new_wrapped(data)
            buf.pts = buf.dts = pts
            buf.duration = frame_duration_ns
            appsrc.emit("push-buffer", buf)
            profiler.record("h264_push", t_push)
        appsrc.emit("end-of-stream")

    def processing_thread():
        global latest_jpeg_frame
        out_idx, last_dets = 0, []
        while True:
            item = display_q.get()
            if item is None: encode_q.put(None); break
            frame, fidx = item

            # [PROFILING] total frame-to-frame time (start)
            t_frame_total = time.perf_counter()

            deadline = time.perf_counter() + SYNC_TIMEOUT_MS / 1000.0
            dets = None
            while time.perf_counter() < deadline:
                with result_lock:
                    if fidx in result_map:
                        dets = result_map.pop(fidx); break
                result_evt.wait(timeout=0.005); result_evt.clear()

            if dets is not None:
                # [PROFILING] tracking
                t_track = time.perf_counter()
                last_dets = tracker.update(dets)
                profiler.record("track", t_track)
                stats["sync_hits"] += 1
            else:
                stats["sync_miss"] += 1

            # draw() profiling is handled inside YoloInference.draw()
            ai.draw(frame, last_dets, profiler)

            now = time.perf_counter()
            if stats["_last_pipe_t"] is not None:
                pipe_fps_window.append(1.0 / max(1e-9, now - stats["_last_pipe_t"]))
            stats["_last_pipe_t"] = now
            pipe_fps = sum(pipe_fps_window) / len(pipe_fps_window) if pipe_fps_window else 0.0

            ai_fps_osd = ai.avg_fps()
            sync_r = stats["sync_hits"] / max(1, stats["sync_hits"] + stats["sync_miss"])
            txt = f"AI: {ai_fps_osd:.1f} fps  sync:{sync_r*100:.0f}%"
            cv2.putText(frame, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
            cv2.putText(frame, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,80), 2)

            # [PROFILING] JPEG encode for network stream
            t_jpeg = time.perf_counter()
            small_frame = cv2.resize(frame, (960, 540))
            _, jpeg_buf = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            profiler.record("jpeg_encode", t_jpeg)
            with jpeg_lock:
                latest_jpeg_frame = jpeg_buf.tobytes()

            encode_q.put((frame.tobytes(), out_idx))
            stats["written"] += 1; out_idx += 1

            # [PROFILING] end of total frame-to-frame
            profiler.record("frame_total", t_frame_total)

            disp_now = time.perf_counter()
            if stats["_last_disp_t"] is not None:
                disp_fps_window.append(1.0 / max(1e-9, disp_now - stats["_last_disp_t"]))
            stats["_last_disp_t"] = disp_now
            disp_fps = sum(disp_fps_window) / len(disp_fps_window) if disp_fps_window else 0.0

            stats["frame_ids"].append(stats["written"])
            stats["ai_fps_hist"].append(ai.recent_fps())
            stats["pipe_fps_hist"].append(pipe_fps)
            stats["disp_fps_hist"].append(disp_fps)

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
        print("\n  Interrupted by user.")

    stop_evt.set()
    loop.quit()
    server.shutdown()
    src_pipeline.set_state(Gst.State.NULL)
    dst_pipeline.set_state(Gst.State.NULL)

    print(f"\n\n  Done! Output file → {output_path}")

    # ── Summary + individual timing graphs ──────────────────────────────────
    profiler.summary()
    save_fps_graph(stats, output_path, fps)
    save_all_timing_graphs(profiler, output_path,
                           model_label=model_label or os.path.basename(model_path))
    print()

    print(f"  Stopping streaming server on port {STREAM_PORT}...")
    server.shutdown()      # Arrête la boucle serve_forever
    server.server_close()  # Libère le socket / port 8080
    server_thread.join()   # Attend la fin du thread

    return profiler


if __name__ == "__main__":
    # ── Single-model run (default) ───────────────────────────────────────────
    #run(DEFAULT_VIDEO, DEFAULT_MODEL, DEFAULT_OUTPUT, model_label="YOLO11n-320")

    # ── Multi-model comparison example ──────────────────────────────────────
    # Uncomment the block below to compare two input sizes on the same video.
    # Each run() saves its own graphs 1-3. Graph 4 is generated at the end.
    #
    DEFAULT_MODEL_640  = f"{BASE_DIR}/yolo11n.onnx"
    DEFAULT_OUTPUT_640 = f"{BASE_DIR}/output_640.mp4"
    #
    p320 = run(DEFAULT_VIDEO, DEFAULT_MODEL,     DEFAULT_OUTPUT, input_size = 320,
                model_label="YOLO11n-320")
    p640 = run(DEFAULT_VIDEO, DEFAULT_MODEL_640, DEFAULT_OUTPUT_640, input_size = 640,
                model_label="YOLO11n-640")
    
    save_graph_model_comparison(
         [("YOLO11n-320", p320), ("YOLO11n-640", p640)],
         DEFAULT_OUTPUT,   # graph saved alongside the first output file
     )