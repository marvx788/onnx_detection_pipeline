#!/usr/bin/env python3
"""
YOLO Detection — MJPEG Stream v6 + Forward Temporal Prediction
==============================================================
Fix principal v6.1 :
  - Vitesse stockée en px/seconde (px/s) au lieu de px/frame
    → indépendante du framerate IA
  - predict() reçoit t_now (perf_counter) et extrapole depuis
    t_ia_result de chaque objet → boxes affichées là où l'objet SERA,
    pas là où il était à la fin de l'inférence
  - Suppression de la double comptabilisation du stage "tracking"
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
DEFAULT_VIDEO = f"{BASE_DIR}/test-video.mp4"
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
RECORD_OUTPUT = "output_compensation.mp4"
RECORD_FPS    = 12.5

# === PROFILING ===
PROFILING_ENABLED  = True
PROFILING_ROLL_WIN = 20
GRAPH_CDF_OUT      = "output_320_graph_2_cdf.png"
GRAPH_BUDGET_OUT   = "output_320_graph_3_stacked_budget.png"
GRAPH_THREAD_OUT   = "output_320_graph_5_thread_avg.png"

# === MOTION COMPENSATION ===
MC_ENABLED        = True
MC_MAX_PREDICT_MS = 500.0   # ne pas prédire au-delà de 500ms sans IA
MC_DECAY_CONF     = 0.02    # perte de confiance par frame equiv (×25fps)
MC_MIN_CONF       = 0.10
MC_IOU_MATCH      = 0.30

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
#  NMS utilitaire (réutilisé par le tracker et le postprocess)
# ─────────────────────────────────────────────────────────────────────────────
def _nms_detections(detections, iou_threshold=NMS_THRESHOLD):
    """
    NMS sur une liste [(box, conf, cls_id), ...].
    Supprime les doublons inter-classes si même position,
    et intra-classe si IoU > iou_threshold.
    """
    if not detections:
        return []

    # Trier par confiance décroissante
    dets = sorted(detections, key=lambda d: d[1], reverse=True)
    kept = []

    def iou(a, b):
        x1,y1,w1,h1 = a; x2,y2,w2,h2 = b
        xi1,yi1 = max(x1,x2), max(y1,y2)
        xi2,yi2 = min(x1+w1,x2+w2), min(y1+h1,y2+h2)
        inter = max(0,xi2-xi1)*max(0,yi2-yi1)
        union = w1*h1+w2*h2-inter
        return inter/union if union>0 else 0.0

    suppressed = [False] * len(dets)
    for i in range(len(dets)):
        if suppressed[i]:
            continue
        kept.append(dets[i])
        for j in range(i+1, len(dets)):
            if suppressed[j]:
                continue
            # Supprimer si même classe ET fort chevauchement
            if dets[i][2] == dets[j][2] and iou(dets[i][0], dets[j][0]) > iou_threshold:
                suppressed[j] = True

    return kept


# ─────────────────────────────────────────────────────────────────────────────
#  Motion Compensation — Tracker avec prédiction forward temporelle
# ─────────────────────────────────────────────────────────────────────────────
class MotionCompensatedTracker:
    """
    Chaque objet :
        {
            "id": int,
            "class": int,
            "box": [x, y, w, h],      # dernière position IA mesurée
            "box_ia": [x, y, w, h],   # idem (référence pour l'extrapolation)
            "conf": float,
            "conf_ia": float,          # confiance au moment de la mesure IA
            "t_ia_result": float,      # perf_counter() quand l'IA a fini
            "vx_ps": float,            # vitesse x en px/SECONDE  ← clé du fix
            "vy_ps": float,
            "vw_ps": float,
            "vh_ps": float,
            "last_update": float,
            "frames_since_ia": int,
            "age": int,
            "color": tuple,
        }
    """

    _next_id = 0
    _cls_lock = threading.Lock()

    def __init__(self, max_age=5, iou_threshold=MC_IOU_MATCH):
        self.objects = []
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self._lock = threading.Lock()
        self._frame_counter = 0

    # ------------------------------------------------------------------
    def _iou(self, box_a, box_b):
        x1, y1, w1, h1 = box_a
        x2, y2, w2, h2 = box_b
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
        union = w1*h1 + w2*h2 - inter
        return inter / union if union > 0 else 0.0

    def _center_dist(self, box_a, box_b):
        cx1, cy1 = box_a[0]+box_a[2]/2, box_a[1]+box_a[3]/2
        cx2, cy2 = box_b[0]+box_b[2]/2, box_b[1]+box_b[3]/2
        avg_size = (box_a[2]+box_a[3]+box_b[2]+box_b[3])/4 + 1e-6
        return ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5 / avg_size

    def _match_detections(self, detections):
        matched = set()
        assignments = []
        det_indices = sorted(range(len(detections)),
                             key=lambda i: detections[i][1], reverse=True)
        for det_idx in det_indices:
            box, conf, cls_id = detections[det_idx]
            best_obj, best_score = None, -1
            for obj_idx, obj in enumerate(self.objects):
                if obj_idx in matched or obj["class"] != cls_id:
                    continue
                # Matcher sur box_ia (dernière position mesurée par l'IA)
                # et non sur box (position prédite qui a dérivé), ce qui
                # évite les faux "non-match" → création de doublons
                ref = obj.get("box_ia", obj["box"])
                iou   = self._iou(box, ref)
                cdist = self._center_dist(box, ref)
                score = iou * 2.0 + max(0, 1.0 - cdist)
                if score > best_score and iou >= self.iou_threshold:
                    best_score, best_obj = score, obj_idx
            if best_obj is not None:
                matched.add(best_obj)
                assignments.append((det_idx, best_obj))
            else:
                assignments.append((det_idx, None))
        return assignments

    # ------------------------------------------------------------------
    def _update_velocity(self, obj, new_box, t_capture, alpha=0.6):
        """
        Vitesse en px/s calculée entre les deux INSTANTS DE CAPTURE.
        t_capture = moment où la frame courante a été lue depuis la vidéo.
        On ignore t_ia_result (fin d'inférence) car c'est du bruit de latence.
        """
        dt_s = t_capture - obj.get("t_capture", t_capture)
        if dt_s < 1e-4:
            dt_s = 0.04   # fallback ~25fps

        ref = obj["box_ia"]
        vx = (new_box[0] - ref[0]) / dt_s
        vy = (new_box[1] - ref[1]) / dt_s
        vw = (new_box[2] - ref[2]) / dt_s
        vh = (new_box[3] - ref[3]) / dt_s

        obj["vx_ps"] = alpha * vx + (1-alpha) * obj.get("vx_ps", 0.0)
        obj["vy_ps"] = alpha * vy + (1-alpha) * obj.get("vy_ps", 0.0)
        obj["vw_ps"] = alpha * vw + (1-alpha) * obj.get("vw_ps", 0.0)
        obj["vh_ps"] = alpha * vh + (1-alpha) * obj.get("vh_ps", 0.0)

    # ------------------------------------------------------------------
    def update(self, detections, t_capture=None):
        """
        Appelé depuis le thread IA quand une inférence est terminée.

        t_capture = perf_counter() au moment où la frame a été LUE depuis
                    la vidéo/caméra — pas quand l'IA a fini.
        C'est la référence temporelle correcte : box_ia représente la
        position de l'objet à t_capture, c'est donc depuis t_capture
        qu'on doit extrapoler vers t_now (display).
        """
        if t_capture is None:
            t_capture = time.perf_counter()

        with self._lock:
            assignments = self._match_detections(detections)

            for det_idx, obj_idx in assignments:
                box, conf, cls_id = detections[det_idx]
                if obj_idx is not None:
                    obj = self.objects[obj_idx]
                    self._update_velocity(obj, box, t_capture)
                    obj["box"]       = list(box)
                    obj["box_ia"]    = list(box)   # position à t_capture
                    obj["conf"]      = conf
                    obj["conf_ia"]   = conf
                    obj["t_capture"] = t_capture   # ← référence temporelle réelle
                    obj["last_update"]     = t_capture
                    obj["frames_since_ia"] = 0
                    obj["age"]            += 1
                else:
                    with MotionCompensatedTracker._cls_lock:
                        MotionCompensatedTracker._next_id += 1
                        new_id = MotionCompensatedTracker._next_id
                    self.objects.append({
                        "id":        new_id,
                        "class":     cls_id,
                        "box":       list(box),
                        "box_ia":    list(box),
                        "conf":      conf,
                        "conf_ia":   conf,
                        "t_capture": t_capture,    # ← référence temporelle réelle
                        "vx_ps": 0.0, "vy_ps": 0.0,
                        "vw_ps": 0.0, "vh_ps": 0.0,
                        "last_update":     t_capture,
                        "frames_since_ia": 0,
                        "age":  1,
                        "color": CLASS_COLORS[cls_id % len(CLASS_COLORS)],
                    })

            # Supprimer les objets trop vieux ou de confiance trop faible
            self.objects = [
                o for o in self.objects
                if o["frames_since_ia"] < self.max_age
                and o["conf"] > MC_MIN_CONF
            ]

            # Incrémenter frames_since_ia pour les non-mis-à-jour
            updated = {obj_idx for _, obj_idx in assignments if obj_idx is not None}
            for i, obj in enumerate(self.objects):
                if i not in updated and obj["frames_since_ia"] > 0:
                    obj["frames_since_ia"] += 1

            return len(self.objects)

    # ------------------------------------------------------------------
    def predict(self, display_scale=1.0, t_now=None):
        """
        Appelé à CHAQUE frame affichée.

        Extrapole depuis t_capture (quand la frame a été capturée) → t_now.
        dt_s représente donc le vrai temps écoulé depuis que la position
        a été OBSERVÉE par la caméra, pas depuis que l'IA a fini de calculer.
        """
        if t_now is None:
            t_now = time.perf_counter()

        with self._lock:
            self._frame_counter += 1
            result = []

            to_remove = []
            for i, obj in enumerate(self.objects):
                # dt depuis le moment de CAPTURE de la frame (pas fin d'inférence)
                dt_s = t_now - obj["t_capture"]

                if dt_s > MC_MAX_PREDICT_MS / 1000.0:
                    to_remove.append(i)
                    continue

                obj["frames_since_ia"] += 1

                ref = obj["box_ia"]
                x = ref[0] + obj["vx_ps"] * dt_s
                y = ref[1] + obj["vy_ps"] * dt_s
                w = max(1, ref[2] + obj["vw_ps"] * dt_s)
                h = max(1, ref[3] + obj["vh_ps"] * dt_s)

                obj["box"] = [x, y, w, h]

                conf = max(MC_MIN_CONF,
                           obj["conf_ia"] - MC_DECAY_CONF * dt_s * 25)
                obj["conf"] = conf

                sx = display_scale
                result.append((
                    [int(x*sx), int(y*sx), int(w*sx), int(h*sx)],
                    conf,
                    obj["class"]
                ))

            for i in reversed(to_remove):
                self.objects.pop(i)

            if len(result) >= 2:
                result = _nms_detections(result, iou_threshold=NMS_THRESHOLD)

            return result

    # ------------------------------------------------------------------
    def get_stats(self):
        with self._lock:
            n = len(self.objects)
            now = time.perf_counter()
            predicted = sum(
                1 for o in self.objects
                if (now - o["t_capture"]) > 0.02
            )
            return n, predicted


# ─────────────────────────────────────────────────────────────────────────────
#  Profiler
# ─────────────────────────────────────────────────────────────────────────────
class Profiler:
    STAGES = [
        "decode", "preprocess", "inference", "postprocess",
        "tracking", "mc_predict", "draw", "jpeg_encode", "h264_push", "frame_total",
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
#  Graphiques (inchangés)
# ─────────────────────────────────────────────────────────────────────────────
def generate_graphs(profiler: Profiler):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
    except ImportError:
        print("  [GRAPHIQUES] pip install matplotlib  ← manquant")
        return

    data = profiler.get_all()

    STAGE_META = {
        "decode":      ("Decode (GStreamer)",         "#5b9bd5"),
        "preprocess":  ("Pre-process (CPU)",          "#70b8ff"),
        "inference":   ("Inference (OpenVINO)",       "#e05c5c"),
        "postprocess": ("Post-process (YOLO)",        "#e8a030"),
        "tracking":    ("Tracking (IA thread)",       "#9b59b6"),
        "mc_predict":  ("MC Predict (display thread)","#c0392b"),
        "draw":        ("Draw boxes",                 "#2ecc71"),
        "jpeg_encode": ("JPEG encode (stream)",       "#1abc9c"),
        "h264_push":   ("H264 push (filesink)",       "#f39c12"),
        "frame_total": ("Total frame-to-frame",       "#e91e8c"),
    }

    BG   = "#1a1a2e"
    GRID = "#2a2a3e"
    TEXT = "#cccccc"

    def avg(stage):
        vals = data.get(stage, [])
        return float(np.mean(vals)) if vals else 0.0

    stages_plot = [s for s in Profiler.STAGES if data.get(s)]
    n = len(stages_plot)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5*nrows), facecolor=BG)
    if nrows == 1:
        axes = [axes]
    fig.suptitle("CDF — Per-Stage Latency Distribution",
                 color=TEXT, fontsize=14, fontweight="bold", y=1.01)

    for idx, stage in enumerate(stages_plot):
        ax = axes[idx//ncols][idx%ncols] if nrows > 1 else axes[idx%ncols]
        vals = np.array(data[stage])
        sorted_v = np.sort(vals)
        cdf = np.arange(1, len(sorted_v)+1) / len(sorted_v) * 100
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
                        xy=(v, pct), xytext=(4, 2), textcoords="offset points",
                        color="#ff6b6b", fontsize=7)

    for idx in range(len(stages_plot), nrows*ncols):
        ax = axes[idx//ncols][idx%ncols] if nrows > 1 else axes[idx%ncols]
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(GRAPH_CDF_OUT, dpi=120, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  [GRAPHIQUES] CDF sauvegardé : {GRAPH_CDF_OUT}")

    STACK_STAGES = ["decode","preprocess","inference","postprocess",
                    "tracking","mc_predict","draw","jpeg_encode","h264_push"]
    available = [s for s in STACK_STAGES if data.get(s)]
    if available:
        min_len = min(len(data[s]) for s in available)
        if min_len >= 2:
            roll = PROFILING_ROLL_WIN

            def rolling_mean(arr, w):
                out = np.convolve(arr, np.ones(w)/w, mode="valid")
                return np.concatenate([np.full(w-1, out[0]), out])

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
                vals = np.array(data[stage][:min_len])
                smooth = rolling_mean(vals, roll)
                color = STAGE_META.get(stage, ("", "#aaaaaa"))[1]
                label = STAGE_META.get(stage, (stage, ""))[0]
                ax2.fill_between(frames, bottom, bottom+smooth, color=color, alpha=0.9, linewidth=0)
                bottom += smooth
                legend_patches.append(Patch(facecolor=color, label=label))
            if data.get("frame_total") and len(data["frame_total"]) >= min_len:
                total_smooth = rolling_mean(np.array(data["frame_total"][:min_len]), roll)
                ax2.plot(frames, total_smooth, color="white", linewidth=1.0, linestyle="--", alpha=0.8)
            ax2.set_title(f"Per-Frame Time Budget — Stacked ({roll}-frame rolling avg)",
                          color=TEXT, fontsize=12, fontweight="bold")
            ax2.set_xlabel("Frame", color=TEXT)
            ax2.set_ylabel("Cumulative time (ms)", color=TEXT)
            ax2.legend(handles=legend_patches, loc="upper right", fontsize=7,
                       facecolor=BG, edgecolor=GRID, labelcolor=TEXT, ncol=2)
            plt.tight_layout()
            fig2.savefig(GRAPH_BUDGET_OUT, dpi=120, bbox_inches="tight", facecolor=BG, edgecolor="none")
            plt.close(fig2)
            print(f"  [GRAPHIQUES] Budget stacked sauvegardé : {GRAPH_BUDGET_OUT}")

    print(f"  [GRAPHIQUES] Thread avg non généré (mc_predict = thread display séparé)")


# ─────────────────────────────────────────────────────────────────────────────
#  IA (OpenVINO) — inchangé
# ─────────────────────────────────────────────────────────────────────────────
class YoloInference:
    def __init__(self, model_path):
        ie = Core()
        model = ie.read_model(model_path)
        device = get_best_device()
        compiled = ie.compile_model(model, device_name=device,
            config={"CACHE_DIR": "/tmp/ov_cache"})
        self.infer_req = compiled.create_infer_request()
        self.input_layer = compiled.input(0)
        self._lock = threading.Lock()
        self._count = 0
        self._sum_ms = 0.0
        self._recent_ms = deque(maxlen=SMOOTH_WIN)

    def infer(self, frame_bgr, profiler: Profiler = None):
        h, w = frame_bgr.shape[:2]
        t_pre = profiler.begin() if profiler else None
        resized = cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(
            np.transpose(rgb.astype(np.float32)/255.0, (2, 0, 1)), 0)
        if profiler: profiler.end("preprocess", t_pre)

        t_inf = profiler.begin() if profiler else None
        t0 = time.perf_counter()
        self.infer_req.infer({self.input_layer: tensor})
        outputs = [self.infer_req.get_output_tensor(0).data]
        ms = (time.perf_counter() - t0) * 1000
        if profiler: profiler.end("inference", t_inf)

        with self._lock:
            self._count  += 1
            self._sum_ms += ms
            self._recent_ms.append(ms)

        t_post = profiler.begin() if profiler else None
        result = self._postprocess(outputs, w, h)
        if profiler: profiler.end("postprocess", t_post)
        return result

    def _postprocess(self, outputs, fw, fh):
        preds = outputs[0][0]
        xs, ys = fw/INPUT_SIZE, fh/INPUT_SIZE
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
        idx = cv2.dnn.NMSBoxes(boxes, confs, CONF_KEEP, NMS_THRESHOLD)
        flat = idx.flatten() if hasattr(idx, 'flatten') else idx
        return [(boxes[i], confs[i], ids[i]) for i in flat]

    def draw(self, frame, detections, profiler: Profiler = None):
        t_draw = profiler.begin() if profiler else None
        for (x, y, w, h), conf, cls in detections:
            color = CLASS_COLORS[cls % len(CLASS_COLORS)]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2, cv2.LINE_AA)
            label = f"{COCO_CLASSES[cls]}: {conf:.0%}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ly = max(y-4, th+4)
            cv2.rectangle(frame, (x, ly-th-baseline-4), (x+tw+4, ly), color, cv2.FILLED)
            cv2.putText(frame, label, (x+2, ly-baseline-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        if profiler: profiler.end("draw", t_draw)

    def _draw_dashed_rect(self, img, pt1, pt2, color, thickness=2, dash_len=10):
        x1, y1 = pt1; x2, y2 = pt2
        for x_a, y_a, x_b, y_b in [(x1,y1,x2,y1),(x2,y1,x2,y2),(x2,y2,x1,y2),(x1,y2,x1,y1)]:
            dx, dy = x_b-x_a, y_b-y_a
            dist = max(abs(dx), abs(dy))
            if dist == 0: continue
            for i in range(0, dist, dash_len*2):
                start = (int(x_a+dx*i/dist), int(y_a+dy*i/dist))
                end   = (int(x_a+dx*min(i+dash_len,dist)/dist),
                         int(y_a+dy*min(i+dash_len,dist)/dist))
                cv2.line(img, start, end, color, thickness, cv2.LINE_AA)

    def avg_fps(self):
        with self._lock:
            return 1000.0/(self._sum_ms/self._count) if self._count > 0 else 0.0

    def recent_fps(self):
        with self._lock:
            if not self._recent_ms: return 0.0
            return 1000.0/(sum(self._recent_ms)/len(self._recent_ms))

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
#  Serveur MJPEG
# ─────────────────────────────────────────────────────────────────────────────
_jpeg_lock  = threading.Lock()
_jpeg_frame = None
_jpeg_seq   = 0
_jpeg_cond  = threading.Condition(_jpeg_lock)


def make_handler():
    class MJPEGHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args): pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                html = (
                    "<!DOCTYPE html><html><head>"
                    "<title>YOLO Stream v6.1 + Forward Prediction</title>"
                    "<style>body{margin:0;background:#111;display:flex;"
                    "justify-content:center;align-items:center;height:100vh}"
                    "img{max-width:100%;max-height:100vh}</style>"
                    "</head><body><img src=\"/stream\"></body></html>"
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
                            jpg, seq = _jpeg_frame, _jpeg_seq
                        if jpg is None or seq == last_sent:
                            continue
                        last_sent = seq
                        try:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                + jpg + b"\r\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            break
                except Exception:
                    pass
            else:
                self.send_response(404); self.end_headers()
    return MJPEGHandler


# ─────────────────────────────────────────────────────────────────────────────
#  Thread enregistrement MP4
# ─────────────────────────────────────────────────────────────────────────────
def record_thread_fn(record_q, output_path, disp_w, disp_h, fps, profiler, stop_evt):
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
        arr = np.frombuffer(item, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
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

    print("=" * 62)
    print("  YOLO — MJPEG Stream v6.1  (Forward Temporal Prediction)")
    print("=" * 62)
    print(f"  Input   : {video_path}")
    print(f"  Source  : {width}x{height} @ {fps:.1f}fps ({total} frames)")
    print(f"  Display : {disp_w}x{disp_h} (scale {DISPLAY_SCALE})")
    print(f"  JPEG    : qualité {JPEG_QUALITY}")
    print(f"  MP4 out : {RECORD_OUTPUT}")
    print(f"  Profil. : {'activé' if PROFILING_ENABLED else 'désactivé'}")
    print(f"  Motion  : {'activé (forward temporal)' if MC_ENABLED else 'désactivé'}")
    print("-" * 62)
    print(f"  >>> http://{local_ip}:{HTTP_PORT}")
    print("-" * 62)
    print("  Chargement modèle ONNX...")

    ai       = YoloInference(model_path)
    profiler = Profiler(enabled=PROFILING_ENABLED)
    mc_tracker = MotionCompensatedTracker(max_age=5, iou_threshold=MC_IOU_MATCH)

    latest_dets = []
    dets_lock   = threading.Lock()
    stop_evt    = threading.Event()
    ai_done     = threading.Event()

    stats = {
        "displayed": 0, "ai_processed": 0, "ai_skipped": 0, "catchup": 0,
        "t0": None,
        "disp_fps_window": deque(maxlen=SMOOTH_WIN),
        "_last_disp_t": None, "_last_frame_t": None,
        "mc_total_tracked": 0, "mc_predicted": 0,
    }

    ai_q     = queue.Queue(maxsize=AI_QUEUE_SIZE)
    encode_q = queue.Queue(maxsize=1)
    record_q = queue.Queue(maxsize=4)

    def signal_handler(sig, frame_):
        print("\n  [SIGINT] Arrêt demandé")
        stop_evt.set()
        with _jpeg_cond:
            _jpeg_cond.notify_all()
    signal.signal(signal.SIGINT, signal_handler)

    server      = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), make_handler())
    http_thread = threading.Thread(target=server.serve_forever, daemon=True)
    http_thread.start()
    print(f"  Serveur MJPEG démarré sur le port {HTTP_PORT}")

    t_rec = threading.Thread(
        target=record_thread_fn,
        args=(record_q, RECORD_OUTPUT, disp_w, disp_h, RECORD_FPS, profiler, stop_evt),
        daemon=True, name="RecordThread")
    t_rec.start()

    # ------------------------------------------------------------------
    def ai_thread():
        while not stop_evt.is_set():
            try:
                item = ai_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break

            frame, t_capture = item   # ← dépaqueter le tuple

            dets = ai.infer(frame, profiler)

            t_track = profiler.begin()
            if MC_ENABLED:
                sx = DISPLAY_SCALE
                sy = DISPLAY_SCALE
                if sx != 1.0 and dets:
                    dets_scaled = [
                        ([int(x*sx), int(y*sy), int(w*sx), int(h*sy)], c, i)
                        for (x, y, w, h), c, i in dets
                    ]
                else:
                    dets_scaled = dets

                # t_capture = quand la frame a été lue → référence temporelle correcte
                mc_tracker.update(dets_scaled, t_capture=t_capture)
                n_tracked, n_pred = mc_tracker.get_stats()
                stats["mc_total_tracked"] = n_tracked
                stats["mc_predicted"]     = n_pred
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
                    _jpeg_seq  += 1
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
        frame_idx = 0

        while cap.isOpened() and not stop_evt.is_set():

            if stats["_last_frame_t"] is not None:
                profiler.end("frame_total", stats["_last_frame_t"])
            stats["_last_frame_t"] = profiler.begin()

            t_dec = profiler.begin()
            ret, frame = cap.read()
            t_capture = time.perf_counter()   # ← timestamp immédiatement après la lecture
            profiler.end("decode", t_dec)
            if not ret:
                break
            cap.read()  # skip frame

            try:
                ai_q.put_nowait((frame, t_capture))   # ← tuple (frame, t_capture)
            except queue.Full:
                try:
                    ai_q.get_nowait()
                    ai_q.put_nowait((frame, t_capture))
                except queue.Empty:
                    pass
                stats["ai_skipped"] += 1

            if DISPLAY_SCALE != 1.0:
                display_frame = cv2.resize(frame, (disp_w, disp_h))
            else:
                display_frame = frame.copy()

            # ── Forward Temporal Prediction ──────────────────────────────
            # t_now = timestamp exact au moment où on va dessiner
            # predict() extrapole chaque objet depuis t_ia_result → t_now
            t_mcp = profiler.begin()
            if MC_ENABLED:
                t_display_now = time.perf_counter()
                dets_for_draw = mc_tracker.predict(display_scale=1.0,
                                                   t_now=t_display_now)
            else:
                with dets_lock:
                    raw = list(latest_dets)
                if DISPLAY_SCALE != 1.0:
                    sx = DISPLAY_SCALE
                    dets_for_draw = [
                        ([int(x*sx),int(y*sx),int(w*sx),int(h*sx)], c, i)
                        for (x,y,w,h), c, i in raw
                    ]
                else:
                    dets_for_draw = raw
            profiler.end("mc_predict", t_mcp)
            # ─────────────────────────────────────────────────────────────

            ai.draw(display_frame, dets_for_draw, profiler)

            now = time.perf_counter()
            if stats["_last_disp_t"] is not None:
                stats["disp_fps_window"].append(
                    1.0 / max(1e-9, now - stats["_last_disp_t"]))
            stats["_last_disp_t"] = now
            disp_fps = (sum(stats["disp_fps_window"]) / len(stats["disp_fps_window"])
                        if stats["disp_fps_window"] else 0.0)
            ai_fps = ai.recent_fps()

            n_tracked, n_pred = mc_tracker.get_stats() if MC_ENABLED else (0, 0)
            mc_info = f"  MC:{n_tracked}({n_pred}P)" if MC_ENABLED else ""

            txt = (f"Disp:{disp_fps:.1f}fps  AI:{ai_fps:.1f}fps"
                   f"  F:{stats['displayed']}  skip:{stats['ai_skipped']}{mc_info}")
            cv2.putText(display_frame, txt, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
            cv2.putText(display_frame, txt, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,80), 2)

            try:
                encode_q.put_nowait(display_frame)
            except queue.Full:
                try:
                    encode_q.get_nowait()
                    encode_q.put_nowait(display_frame)
                except queue.Empty:
                    pass

            stats["displayed"] += 1

            pct = stats["displayed"] / max(1, total//2) * 100
            bar = "#"*int(25*pct/100) + "-"*(25-int(25*pct/100))
            print(f"\r  [{bar}] {pct:5.1f}%  "
                  f"Disp:{disp_fps:5.1f}fps  AI:{ai_fps:5.1f}fps  "
                  f"IA:{stats['ai_processed']:4d}  skip:{stats['ai_skipped']:3d}"
                  f"{mc_info}",
                  end="", flush=True)

            frame_idx += 1
            target_t   = t_video_start + frame_idx * (2.0/fps)
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
    if MC_ENABLED:
        print(f"  Objets MC max    : {stats['mc_total_tracked']}")

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