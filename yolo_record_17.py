#!/usr/bin/env python3
"""
YOLO Detection — MJPEG Stream v6
==================================
Nouveautés v6 :
  1. Enregistrement MP4 asynchrone (thread dédié, impact perf minimal)
     → fichier output_stream.mp4 écrit en temps réel depuis les JPEG
  2. Profiling par stage avec timestamps précis (perf_counter)
     → graphiques générés en fin de run :
        - output_320_graph_2_cdf.png   : CDF par stage
        - output_320_graph_3_stacked_budget.png : budget empilé par frame
        - output_320_graph_5_thread_avg.png : latence moyenne par thread (NOUVEAU)

Base : architecture v5 (Condition-synchronized, proven 12.5fps)
"""

import sys
import os
import time
import threading
import queue
import socket
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import signal

import numpy as np

try:
    import cv2
except ImportError:
    print("[ERREUR] pip install opencv-python"); sys.exit(1)
try:
    from openvino import Core
except ImportError:
    print("[ERREUR] pip install openvino"); sys.exit(1)

# === CONFIGURATION ===
BASE_DIR      = "/home/comvis/Documents/onnx_detection_pipeline"
DEFAULT_VIDEO = f"{BASE_DIR}/test_video_4_rognee.mp4"
DEFAULT_MODEL = f"{BASE_DIR}/yolo11n_256.onnx"

INPUT_SIZE    = 256
AI_QUEUE_SIZE = 1
DISPLAY_SCALE = 0.75
JPEG_QUALITY  = 70
HTTP_PORT     = 8080
SMOOTH_WIN    = 15

CONF_APPEAR   = 0.45
CONF_KEEP     = 0.25
NMS_THRESHOLD = 0.45

# === ENREGISTREMENT MP4 ===
RECORD_OUTPUT = "output_stream.mp4"   # chemin du fichier de sortie
RECORD_FPS    = 12.5                  # FPS cible du MP4 enregistré

# === PROFILING ===
PROFILING_ENABLED  = True             # activer la collecte de stats
PROFILING_ROLL_WIN = 20               # fenêtre rolling pour le graphique budget
GRAPH_CDF_OUT      = "output_320_graph_2_cdf.png"
GRAPH_BUDGET_OUT   = "output_320_graph_3_stacked_budget.png"
GRAPH_THREAD_OUT   = "output_320_graph_5_thread_avg.png"   # NOUVEAU

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

_rng = np.random.default_rng(42)
CLASS_COLORS = [
    tuple(int(c) for c in color)
    for color in _rng.integers(80, 230, size=(len(COCO_CLASSES), 3))
]


# ─────────────────────────────────────────────────────────────────────────────
#  Collecte de profiling
# ─────────────────────────────────────────────────────────────────────────────
class Profiler:
    """
    Collecte les durées par stage dans des listes thread-safe.
    Usage :
        token = profiler.begin()
        ... travail ...
        profiler.end("stage_name", token)
    """
    STAGES = [
        "decode",        # GStreamer / cap.read
        "preprocess",    # resize + cvtColor + normalize
        "inference",     # OpenVINO
        "postprocess",   # NMS + filtrage
        "tracking",      # hysteresis
        "draw",          # draw boxes
        "jpeg_encode",   # cv2.imencode
        "h264_push",     # VideoWriter.write (MP4)
        "frame_total",   # wall-clock frame-to-frame
    ]

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._lock   = threading.Lock()
        self._data   = {s: [] for s in self.STAGES}

    def begin(self):
        return time.perf_counter() if self.enabled else None

    def end(self, stage, t0):
        if not self.enabled or t0 is None:
            return
        ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self._data[stage].append(ms)

    def get(self, stage):
        with self._lock:
            return list(self._data[stage])

    def get_all(self):
        with self._lock:
            return {s: list(v) for s, v in self._data.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  Génération des graphiques
# ─────────────────────────────────────────────────────────────────────────────
def generate_graphs(profiler: Profiler):
    """Génère les PNG de profiling après la fin du run."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
    except ImportError:
        print("  [GRAPHIQUES] pip install matplotlib  ← manquant, graphiques non générés")
        return

    data = profiler.get_all()

    # Mapping stage → label lisible + couleur
    STAGE_META = {
        "decode":      ("Decode (GStreamer)",       "#5b9bd5"),
        "preprocess":  ("Pre-process (CPU)",        "#70b8ff"),
        "inference":   ("Inference (OpenVINO)",     "#e05c5c"),
        "postprocess": ("Post-process (YOLO)",      "#e8a030"),
        "tracking":    ("Tracking (Hysteresis)",    "#9b59b6"),
        "draw":        ("Draw boxes",               "#2ecc71"),
        "jpeg_encode": ("JPEG encode (stream)",     "#1abc9c"),
        "h264_push":   ("H264 push (filesink)",     "#f39c12"),
        "frame_total": ("Total frame-to-frame",     "#e91e8c"),
    }

    BG    = "#1a1a2e"
    GRID  = "#2a2a3e"
    TEXT  = "#cccccc"

    # helper : moyenne sécurisée
    def avg(stage):
        vals = data.get(stage, [])
        return float(np.mean(vals)) if vals else 0.0

    # ── 1. CDF par stage ──────────────────────────────────────────────────────
    stages_plot = [s for s in Profiler.STAGES if data[s]]
    n = len(stages_plot)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(18, 5 * nrows),
                             facecolor=BG)
    fig.suptitle("CDF — Per-Stage Latency Distribution",
                 color=TEXT, fontsize=14, fontweight="bold", y=1.01)

    for idx, stage in enumerate(stages_plot):
        ax   = axes[idx // ncols][idx % ncols]
        vals = np.array(data[stage])
        sorted_v = np.sort(vals)
        cdf      = np.arange(1, len(sorted_v) + 1) / len(sorted_v) * 100

        color = STAGE_META.get(stage, ("", "#aaaaaa"))[1]
        label = STAGE_META.get(stage, (stage, ""))[0]

        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.grid(True, color=GRID, linewidth=0.5)

        ax.plot(sorted_v, cdf, color=color, linewidth=1.5)
        ax.set_title(label, color=TEXT, fontsize=9, fontweight="bold")
        ax.set_xlabel("Latency (ms)", fontsize=8)
        ax.set_ylabel("Cumulative %", fontsize=8)
        ax.set_ylim(0, 105)

        for pct, pct_label in [(50, "P50"), (95, "P95"), (99, "P99")]:
            v = np.percentile(vals, pct)
            ax.axvline(v, color="#ff6b6b", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.scatter([v], [pct], color="#ff6b6b", s=30, zorder=5)
            ax.annotate(f"{pct_label}: {v:.1f}ms",
                        xy=(v, pct), xytext=(4, 2),
                        textcoords="offset points",
                        color="#ff6b6b", fontsize=7)

    # Masquer axes vides
    for idx in range(len(stages_plot), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    fig.savefig(GRAPH_CDF_OUT, dpi=120, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  [GRAPHIQUES] CDF sauvegardé : {GRAPH_CDF_OUT}")

    # ── 2. Stacked budget par frame ───────────────────────────────────────────
    STACK_STAGES = [
        "decode", "preprocess", "inference", "postprocess",
        "tracking", "draw", "jpeg_encode", "h264_push"
    ]
    available = [s for s in STACK_STAGES if data[s]]
    if not available:
        print("  [GRAPHIQUES] Pas assez de données pour le budget stacked")
    else:
        min_len = min(len(data[s]) for s in available)
        if min_len < 2:
            print("  [GRAPHIQUES] Pas assez de frames pour le budget stacked")
        else:
            roll = PROFILING_ROLL_WIN

            def rolling_mean(arr, w):
                out = np.convolve(arr, np.ones(w) / w, mode="valid")
                pad = np.full(w - 1, out[0])
                return np.concatenate([pad, out])

            frames = np.arange(min_len)

            fig2, ax2 = plt.subplots(figsize=(18, 5), facecolor=BG)
            ax2.set_facecolor(BG)
            ax2.tick_params(colors=TEXT)
            for spine in ax2.spines.values():
                spine.set_color(GRID)
            ax2.xaxis.label.set_color(TEXT)
            ax2.yaxis.label.set_color(TEXT)
            ax2.grid(True, color=GRID, linewidth=0.5, axis="y")

            bottom = np.zeros(min_len)
            legend_patches = []

            for stage in available:
                vals   = np.array(data[stage][:min_len])
                smooth = rolling_mean(vals, roll)
                color  = STAGE_META.get(stage, ("", "#aaaaaa"))[1]
                label  = STAGE_META.get(stage, (stage, ""))[0]
                ax2.fill_between(frames, bottom, bottom + smooth,
                                 color=color, alpha=0.9, linewidth=0)
                bottom += smooth
                legend_patches.append(Patch(facecolor=color, label=label))

            if data["frame_total"] and len(data["frame_total"]) >= min_len:
                total_smooth = rolling_mean(
                    np.array(data["frame_total"][:min_len]), roll)
                ax2.plot(frames, total_smooth, color="white",
                         linewidth=1.0, linestyle="--",
                         label="Total frame-to-frame (measured)", alpha=0.8)
                legend_patches.insert(0,
                    Patch(facecolor="none", edgecolor="white",
                          linestyle="--", label="Total frame-to-frame (measured)"))

            ax2.set_title(
                f"Per-Frame Time Budget — Stacked by Stage  ({roll}-frame rolling average)",
                color=TEXT, fontsize=12, fontweight="bold")
            ax2.set_xlabel("Frame", color=TEXT)
            ax2.set_ylabel("Cumulative time (ms)", color=TEXT)
            ax2.legend(handles=legend_patches, loc="upper right",
                       fontsize=7, facecolor=BG, edgecolor=GRID,
                       labelcolor=TEXT, ncol=2)

            plt.tight_layout()
            fig2.savefig(GRAPH_BUDGET_OUT, dpi=120, bbox_inches="tight",
                         facecolor=BG, edgecolor="none")
            plt.close(fig2)
            print(f"  [GRAPHIQUES] Budget stacked sauvegardé : {GRAPH_BUDGET_OUT}")

    # ── 3. NOUVEAU : Average Timing Breakdown by Thread — Stacked Bar ─────────
    #
    # Chaque thread regroupe ses stages "actifs" + une part "idle/attente"
    # calculée comme :  idle = frame_to_frame_avg - somme_des_stages_actifs
    # (clampée à 0 pour éviter les valeurs négatives dues aux imprécisions).
    #
    # Mapping thread → liste de stages mesurés dans ce thread :
    #   Thread Principal : decode, draw
    #   Thread IA        : preprocess, inference, postprocess, tracking
    #   Thread Encodeur  : jpeg_encode
    #   Thread Recorder  : h264_push
    # ─────────────────────────────────────────────────────────────────────────

    frame_avg = avg("frame_total")   # référence frame-to-frame moyenne

    THREADS = [
        {
            "name":   "Main Thread",
            "stages": ["decode", "draw"],
        },
        {
            "name":   "AI Thread",
            "stages": ["preprocess", "inference", "postprocess", "tracking"],
        },
        {
            "name":   "Encode Thread",
            "stages": ["jpeg_encode"],
        },
        {
            "name":   "Recorder Thread",
            "stages": ["h264_push"],
        },
    ]

    # Couleur hachures idle
    IDLE_COLOR   = "#404060"
    IDLE_LABEL   = "Idle / Attente"
    FRAME_COLOR  = "#e91e8c"

    fig3, ax3 = plt.subplots(figsize=(14, 7), facecolor=BG)
    ax3.set_facecolor(BG)
    ax3.tick_params(colors=TEXT, labelsize=10)
    for spine in ax3.spines.values():
        spine.set_color(GRID)
    ax3.xaxis.label.set_color(TEXT)
    ax3.yaxis.label.set_color(TEXT)
    ax3.grid(True, color=GRID, linewidth=0.5, axis="y", zorder=0)

    x_positions = np.arange(len(THREADS))
    bar_width   = 0.55

    # Construire les segments de chaque barre
    # On collecte d'abord tous les stages uniques pour la légende
    all_stages_used = []
    thread_segments = []   # list of [(stage_or_idle, avg_ms), ...]

    for thr in THREADS:
        segs = []
        total_active = 0.0
        for stage in thr["stages"]:
            ms = avg(stage)
            if ms > 0:
                segs.append((stage, ms))
                total_active += ms
                if stage not in all_stages_used:
                    all_stages_used.append(stage)
        # idle
        idle_ms = max(0.0, frame_avg - total_active)
        segs.append(("idle", idle_ms))
        thread_segments.append(segs)

    # Dessin des barres empilées
    for xi, (thr, segs) in enumerate(zip(THREADS, thread_segments)):
        bottom = 0.0
        total  = sum(ms for _, ms in segs)
        active = sum(ms for s, ms in segs if s != "idle")

        for stage, ms in segs:
            if ms <= 0:
                continue
            if stage == "idle":
                # Barre hachurée pour l'idle
                bar = ax3.bar(xi, ms, bar_width, bottom=bottom,
                              color=IDLE_COLOR, zorder=2,
                              linewidth=0)
                # Superposer le motif hachuré
                ax3.bar(xi, ms, bar_width, bottom=bottom,
                        fill=False, edgecolor="#7070a0", linewidth=0.5,
                        hatch="////", zorder=3)
                # Annotation idle
                if ms > 3:
                    ax3.text(xi, bottom + ms / 2, f"{ms:.1f}ms",
                             ha="center", va="center",
                             color=TEXT, fontsize=8, fontweight="bold")
            else:
                color = STAGE_META.get(stage, ("", "#aaaaaa"))[1]
                ax3.bar(xi, ms, bar_width, bottom=bottom,
                        color=color, zorder=2, linewidth=0)
                # Annotation valeur si assez de place
                if ms > 5:
                    ax3.text(xi, bottom + ms / 2, f"{ms:.1f}ms",
                             ha="center", va="center",
                             color="white", fontsize=8, fontweight="bold")
            bottom += ms

        # Annotation total au-dessus de la barre
        ax3.text(xi, total + 2, f"Total: {total:.1f}ms",
                 ha="center", va="bottom",
                 color=TEXT, fontsize=9, fontweight="bold")

        # Utilisation % (active / frame_avg * 100) sous le label X
        util = (active / frame_avg * 100) if frame_avg > 0 else 0.0
        ax3.text(xi, -12, f"{util:.1f}% util.",
                 ha="center", va="top",
                 color="#888888", fontsize=8,
                 transform=ax3.transData)

    # Ligne frame-to-frame
    if frame_avg > 0:
        ax3.axhline(frame_avg, color=FRAME_COLOR, linestyle="--",
                    linewidth=1.5, zorder=5)
        ax3.text(len(THREADS) - 0.5 + 0.35, frame_avg + 1,
                 f"Frame-to-frame: {frame_avg:.1f}ms",
                 color=FRAME_COLOR, fontsize=9, fontweight="bold",
                 va="bottom", ha="right")

    # Axes et labels
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels([t["name"] for t in THREADS], color=TEXT, fontsize=10)
    ax3.set_ylabel("Average Latency (ms)", color=TEXT, fontsize=10)
    ax3.set_title("Average Timing Breakdown by Thread — Stacked Bar (ms)",
                  color=TEXT, fontsize=13, fontweight="bold", pad=14)

    # Marge basse pour les annotations % util.
    ymin, ymax = ax3.get_ylim()
    ax3.set_ylim(ymin - 18, ymax * 1.08)
    ax3.set_xlim(-0.6, len(THREADS) - 0.4)

    # Légende
    legend_handles = []
    for stage in all_stages_used:
        color = STAGE_META.get(stage, ("", "#aaaaaa"))[1]
        label = STAGE_META.get(stage, (stage, ""))[0]
        legend_handles.append(Patch(facecolor=color, label=label))

    # Entrée frame-to-frame
    legend_handles.insert(0,
        Line2D([0], [0], color=FRAME_COLOR, linestyle="--", linewidth=1.5,
               label=f"Frame-to-frame ({frame_avg:.1f}ms)"))
    # Entrée idle
    idle_patch = Patch(facecolor=IDLE_COLOR, hatch="////",
                       edgecolor="#7070a0", label=IDLE_LABEL)
    legend_handles.append(idle_patch)

    ax3.legend(handles=legend_handles,
               loc="upper right", fontsize=8,
               facecolor=BG, edgecolor=GRID,
               labelcolor=TEXT, ncol=1)

    plt.tight_layout()
    fig3.savefig(GRAPH_THREAD_OUT, dpi=120, bbox_inches="tight",
                 facecolor=BG, edgecolor="none")
    plt.close(fig3)
    print(f"  [GRAPHIQUES] Thread avg sauvegardé : {GRAPH_THREAD_OUT}")


# ─────────────────────────────────────────────────────────────────────────────
#  IA (OpenVINO)
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

    def infer(self, frame_bgr, profiler: Profiler = None):
        h, w    = frame_bgr.shape[:2]

        t_pre = profiler.begin() if profiler else None
        resized = cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor  = np.expand_dims(
            np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)), 0)
        if profiler: profiler.end("preprocess", t_pre)

        t_inf = profiler.begin() if profiler else None
        t0 = time.perf_counter()
        self.infer_req.infer({self.input_layer: tensor})
        outputs = [self.infer_req.get_output_tensor(0).data]
        ms = (time.perf_counter() - t0) * 1000
        if profiler: profiler.end("inference", t_inf)

        with self._lock:
            self._count     += 1
            self._sum_ms    += ms
            self._recent_ms.append(ms)

        t_post = profiler.begin() if profiler else None
        result = self._postprocess(outputs, w, h)
        if profiler: profiler.end("postprocess", t_post)

        return result

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

    def draw(self, frame, detections, profiler: Profiler = None):
        t_draw = profiler.begin() if profiler else None
        for (x, y, w, h), conf, cls in detections:
            color = CLASS_COLORS[cls % len(CLASS_COLORS)]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{COCO_CLASSES[cls]}: {conf:.0%}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ly = max(y - 4, th + 4)
            cv2.rectangle(frame,
                          (x, ly - th - baseline - 4),
                          (x + tw + 4, ly),
                          color, cv2.FILLED)
            cv2.putText(frame, label, (x + 2, ly - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
        if profiler: profiler.end("draw", t_draw)

    def avg_fps(self):
        with self._lock:
            return 1000.0 / (self._sum_ms / self._count) if self._count > 0 else 0.0

    def recent_fps(self):
        with self._lock:
            if not self._recent_ms:
                return 0.0
            return 1000.0 / (sum(self._recent_ms) / len(self._recent_ms))

    def count(self):
        with self._lock:
            return self._count


def get_best_device():
    ie = Core()
    available = ie.available_devices
    print(f"  Devices available: {available}")
    for device in ["GPU", "CPU"]:
        if device in available:
            print(f"  Device sélected  : {device}")
            return device
    return "CPU"


# ─────────────────────────────────────────────────────────────────────────────
#  Serveur MJPEG — synchronisé par Condition (pas de sleep fixe)
# ─────────────────────────────────────────────────────────────────────────────
_jpeg_lock  = threading.Lock()
_jpeg_frame = None
_jpeg_seq   = 0
_jpeg_cond  = threading.Condition(_jpeg_lock)


def make_handler():
    class MJPEGHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                html = (
                    "<!DOCTYPE html><html><head>"
                    "<title>YOLO Stream v6</title>"
                    "<style>body{margin:0;background:#111;display:flex;"
                    "justify-content:center;align-items:center;height:100vh}"
                    "img{max-width:100%;max-height:100vh}</style>"
                    "</head><body>"
                    '<img src="/stream">'
                    "</body></html>"
                )
                self.wfile.write(html.encode())

            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                last_sent = -1
                try:
                    while True:
                        with _jpeg_cond:
                            if _jpeg_seq == last_sent:
                                _jpeg_cond.wait(timeout=1.0)
                            jpg = _jpeg_frame
                            seq = _jpeg_seq

                        if jpg is None or seq == last_sent:
                            continue

                        last_sent = seq
                        try:
                            self.wfile.write(
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n\r\n"
                                + jpg + b"\r\n"
                            )
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            break
                except Exception:
                    pass
            else:
                self.send_response(404)
                self.end_headers()

    return MJPEGHandler


# ─────────────────────────────────────────────────────────────────────────────
#  Thread enregistrement MP4
# ─────────────────────────────────────────────────────────────────────────────
def record_thread_fn(record_q: queue.Queue, output_path: str,
                     disp_w: int, disp_h: int, fps: float,
                     profiler: Profiler, stop_evt: threading.Event):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (disp_w, disp_h))
    if not writer.isOpened():
        print(f"  [RECORD] Impossible d'ouvrir VideoWriter pour {output_path}")
        return

    print(f"  [RECORD] Enregistrement MP4 : {output_path}  ({disp_w}x{disp_h} @ {fps}fps)")

    while not stop_evt.is_set() or not record_q.empty():
        try:
            item = record_q.get(timeout=0.3)
        except queue.Empty:
            continue
        if item is None:
            break

        t_h264 = profiler.begin()
        jpg_array = np.frombuffer(item, dtype=np.uint8)
        frame_bgr = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
        if frame_bgr is not None:
            if frame_bgr.shape[1] != disp_w or frame_bgr.shape[0] != disp_h:
                frame_bgr = cv2.resize(frame_bgr, (disp_w, disp_h))
            writer.write(frame_bgr)
        profiler.end("h264_push", t_h264)

    writer.release()
    print(f"  [RECORD] Fichier MP4 finalisé : {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def run(video_path, model_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir : {video_path}")
        return False

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    disp_w = int(width  * DISPLAY_SCALE)
    disp_h = int(height * DISPLAY_SCALE)

    local_ip = get_local_ip()

    print("=" * 60)
    print("  YOLO — MJPEG Stream v6 (+ MP4 record + profiling)")
    print("=" * 60)
    print(f"  Input   : {video_path}")
    print(f"  Source  : {width}x{height} @ {fps:.1f}fps ({total} frames)")
    print(f"  Display : {disp_w}x{disp_h} (scale {DISPLAY_SCALE})")
    print(f"  JPEG    : qualité {JPEG_QUALITY}")
    print(f"  MP4 out : {RECORD_OUTPUT}")
    print(f"  Profil. : {'activé' if PROFILING_ENABLED else 'désactivé'}")
    print("-" * 60)
    print(f"  >>> Ouvrir dans le navigateur :")
    print(f"  >>> http://{local_ip}:{HTTP_PORT}")
    print("-" * 60)
    print("  Chargement modèle ONNX...")

    ai       = YoloInference(model_path)
    profiler = Profiler(enabled=PROFILING_ENABLED)

    latest_dets = []
    dets_lock   = threading.Lock()
    stop_evt    = threading.Event()
    ai_done     = threading.Event()

    stats = {
        "displayed":       0,
        "ai_processed":    0,
        "ai_skipped":      0,
        "catchup":         0,
        "t0":              None,
        "disp_fps_window": deque(maxlen=SMOOTH_WIN),
        "_last_disp_t":    None,
        "_last_frame_t":   None,
    }

    ai_q      = queue.Queue(maxsize=AI_QUEUE_SIZE)
    encode_q  = queue.Queue(maxsize=1)
    record_q  = queue.Queue(maxsize=4)

    def signal_handler(sig, frame_):
        print("\n  [SIGINT] Arrêt demandé")
        stop_evt.set()
        with _jpeg_cond:
            _jpeg_cond.notify_all()
    signal.signal(signal.SIGINT, signal_handler)

    server = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), make_handler())
    http_thread = threading.Thread(target=server.serve_forever, daemon=True)
    http_thread.start()
    print(f"  Serveur MJPEG démarré sur le port {HTTP_PORT}")

    t_rec = threading.Thread(
        target=record_thread_fn,
        args=(record_q, RECORD_OUTPUT, disp_w, disp_h,
              RECORD_FPS, profiler, stop_evt),
        daemon=True,
        name="RecordThread"
    )
    t_rec.start()

    def ai_thread():
        while not stop_evt.is_set():
            try:
                frame = ai_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is None:
                break
            dets = ai.infer(frame, profiler)
            if DISPLAY_SCALE != 1.0 and dets:
                sx, sy = DISPLAY_SCALE, DISPLAY_SCALE
                dets = [
                    ([int(x*sx), int(y*sy), int(w*sx), int(h*sy)], c, i)
                    for (x, y, w, h), c, i in dets
                ]

            t_track = profiler.begin()
            time.sleep(0)
            profiler.end("tracking", t_track)

            with dets_lock:
                latest_dets[:] = dets
                stats["ai_processed"] += 1
        ai_done.set()

    t_ai = threading.Thread(target=ai_thread, daemon=True)
    t_ai.start()

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    def encode_thread():
        global _jpeg_frame, _jpeg_seq
        while not stop_evt.is_set():
            try:
                frame_to_encode = encode_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame_to_encode is None:
                break

            t_jpeg = profiler.begin()
            ok, jpg_buf = cv2.imencode(".jpg", frame_to_encode, encode_params)
            profiler.end("jpeg_encode", t_jpeg)

            if ok:
                jpg_bytes = jpg_buf.tobytes()

                with _jpeg_cond:
                    _jpeg_frame = jpg_bytes
                    _jpeg_seq += 1
                    _jpeg_cond.notify_all()

                try:
                    record_q.put_nowait(jpg_bytes)
                except queue.Full:
                    pass

    t_enc = threading.Thread(target=encode_thread, daemon=True)
    t_enc.start()

    print("  Démarrage...\n")
    stats["t0"] = time.monotonic()

    try:
        t_video_start = time.perf_counter()
        frame_idx     = 0

        while cap.isOpened() and not stop_evt.is_set():

            if stats["_last_frame_t"] is not None:
                profiler.end("frame_total", stats["_last_frame_t"])
            stats["_last_frame_t"] = profiler.begin()

            t_dec = profiler.begin()
            ret, frame = cap.read()
            profiler.end("decode", t_dec)
            if not ret:
                break
            cap.read()  # skip frame

            try:
                ai_q.put_nowait(frame)
            except queue.Full:
                try:
                    ai_q.get_nowait()
                    ai_q.put_nowait(frame)
                except queue.Empty:
                    pass
                stats["ai_skipped"] += 1

            if DISPLAY_SCALE != 1.0:
                display_frame = cv2.resize(frame, (disp_w, disp_h))
            else:
                display_frame = frame.copy()

            with dets_lock:
                dets = list(latest_dets)
            ai.draw(display_frame, dets, profiler)

            ai_fps = ai.recent_fps()
            now    = time.perf_counter()
            if stats["_last_disp_t"] is not None:
                stats["disp_fps_window"].append(
                    1.0 / max(1e-9, now - stats["_last_disp_t"]))
            stats["_last_disp_t"] = now
            disp_fps = (sum(stats["disp_fps_window"]) / len(stats["disp_fps_window"])
                        if stats["disp_fps_window"] else 0.0)

            txt = (f"Disp:{disp_fps:.1f}fps  AI:{ai_fps:.1f}fps"
                   f"  F:{stats['displayed']}  skip:{stats['ai_skipped']}")
            cv2.putText(display_frame, txt, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(display_frame, txt, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2)

            try:
                encode_q.put_nowait(display_frame)
            except queue.Full:
                try:
                    encode_q.get_nowait()
                    encode_q.put_nowait(display_frame)
                except queue.Empty:
                    pass

            stats["displayed"] += 1

            pct = stats["displayed"] / max(1, total // 2) * 100
            bar = "#" * int(25 * pct / 100) + "-" * (25 - int(25 * pct / 100))
            print(f"\r  [{bar}] {pct:5.1f}%  "
                  f"Disp:{disp_fps:5.1f}fps  AI:{ai_fps:5.1f}fps  "
                  f"IA:{stats['ai_processed']:4d}  skip:{stats['ai_skipped']:3d}",
                  end="", flush=True)

            frame_idx += 1
            target_t   = t_video_start + frame_idx * (2.0 / fps)
            sleep_time = target_t - time.perf_counter()

            if sleep_time < -0.04:
                t_video_start += 0.02
                sleep_time = target_t - time.perf_counter()
                stats["catchup"] += 1

            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"\n[ERREUR] {e}")
        import traceback; traceback.print_exc()
        stop_evt.set()

    print("\n\n  Finalisation...")
    stop_evt.set()
    with _jpeg_cond:
        _jpeg_cond.notify_all()
    server.shutdown()

    for q in (ai_q, encode_q):
        while not q.empty():
            try: q.get_nowait()
            except queue.Empty: break
        q.put(None)

    record_q.put(None)

    ai_done.wait(timeout=3.0)
    t_ai.join(timeout=2.0)
    t_enc.join(timeout=2.0)
    t_rec.join(timeout=10.0)
    cap.release()

    elapsed = time.monotonic() - stats["t0"]
    print(f"\n  Terminé !")
    print(f"  Frames affichées : {stats['displayed']}")
    print(f"  Frames IA        : {stats['ai_processed']}")
    print(f"  Skip IA          : {stats['ai_skipped']}")
    print(f"  Rattrapages      : {stats['catchup']}")
    print(f"  FPS IA moyen     : {ai.avg_fps():.1f}")
    print(f"  Durée totale     : {elapsed:.1f}s")
    print(f"  Vitesse affichage: {stats['displayed']/elapsed:.1f} FPS")

    if PROFILING_ENABLED:
        print("\n  Génération des graphiques de profiling...")
        generate_graphs(profiler)

    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args  = sys.argv[1:]
    video = args[0] if args else DEFAULT_VIDEO
    model = args[1] if len(args) > 1 else DEFAULT_MODEL

    if not os.path.exists(video):
        print(f"[ERREUR] Vidéo introuvable : {video}"); sys.exit(1)
    if not os.path.exists(model):
        print(f"[ERREUR] Modèle introuvable : {model}"); sys.exit(1)

    run(video, model)