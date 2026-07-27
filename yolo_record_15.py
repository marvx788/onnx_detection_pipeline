#!/usr/bin/env python3
"""
YOLO Detection — Real-time Display Priority v10 (Pure OpenCV)
=============================================================

AMÉLIORATIONS v10 :
  1. Resize display fait dans le thread IA (pas dans la boucle principale)
     → le main thread reçoit directement une frame prête à afficher
  2. Suppression du double calcul de coordonnées (scale appliqué une seule fois)
  3. Labels de classe + score affichés sur les boîtes
  4. Couleurs par classe (palette fixe, cohérente entre frames)
  5. frame_time unifié (suppression du double sleep qui décalait le timing)
  6. DISPLAY_SCALE peut maintenant monter à 1.0 sans ralentir l'affichage
"""

import sys
import os
import time
import threading
import queue
from collections import deque
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

INPUT_SIZE     = 256
AI_QUEUE_SIZE  = 1

# ── Affichage ────────────────────────────────────────────────────────────────
# Peut maintenant être mis à 0.75 ou 1.0 sans pénalité d'affichage
# car le resize est fait dans le thread IA (en arrière-plan)
DISPLAY_SCALE  = 0.75

# ── Seuils détection ─────────────────────────────────────────────────────────
CONF_APPEAR   = 0.45
CONF_KEEP     = 0.25
NMS_THRESHOLD = 0.45

# ── Fenêtre lissage FPS ───────────────────────────────────────────────────────
SMOOTH_WIN = 15

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

# ── Palette de couleurs par classe (BGR, générée une fois) ───────────────────
_rng = np.random.default_rng(42)
CLASS_COLORS = [
    tuple(int(c) for c in color)
    for color in _rng.integers(80, 230, size=(len(COCO_CLASSES), 3))
]


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

    def infer(self, frame_bgr):
        """
        Inférence sur frame_bgr (résolution originale ou display, peu importe).
        Retourne les détections en coordonnées display (déjà scalées).
        """
        h, w    = frame_bgr.shape[:2]
        resized = cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor  = np.expand_dims(
            np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)), 0)
        t0 = time.perf_counter()
        self.infer_req.infer({self.input_layer: tensor})
        outputs = [self.infer_req.get_output_tensor(0).data]
        ms = (time.perf_counter() - t0) * 1000
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
        """Dessine les boîtes + labels (classe + score) sur la frame."""
        for (x, y, w, h), conf, cls in detections:
            color = CLASS_COLORS[cls % len(CLASS_COLORS)]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label = f"{COCO_CLASSES[cls]}: {conf:.0%}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

            # Fond semi-opaque pour le texte
            ly = max(y - 4, th + 4)
            cv2.rectangle(frame,
                          (x, ly - th - baseline - 4),
                          (x + tw + 4, ly),
                          color, cv2.FILLED)
            cv2.putText(frame, label,
                        (x + 2, ly - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)

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


# ─────────────────────────────────────────────────────────────────────────────
#  Info device
# ─────────────────────────────────────────────────────────────────────────────
def get_best_device():
    ie = Core()
    available = ie.available_devices
    print(f"  Devices available: {available}")
    priority = ["GPU", "CPU"]
    for device in priority:
        if device in available:
            print(f"  Device sélected  : {device}")
            return device
    return "CPU"


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────
def run(video_path, model_path):
    # ── Lecture vidéo ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la vidéo : {video_path}")
        return False

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    disp_w = int(width * DISPLAY_SCALE)
    disp_h = int(height * DISPLAY_SCALE)

    print("=" * 60)
    print("  YOLO — Real-time Display Priority v10 (Pure OpenCV)")
    print("=" * 60)
    print(f"  Input   : {video_path}")
    print(f"  Model   : {model_path}")
    print(f"  Source  : {width}x{height} @ {fps:.1f}fps ({total} frames)")
    print(f"  Display : {disp_w}x{disp_h} @ {fps:.1f}fps (scale {DISPLAY_SCALE})")
    print(f"  AI queue: {AI_QUEUE_SIZE} (skip si IA en retard)")
    print("=" * 60)
    print("  Chargement modèle ONNX...")
    ai = YoloInference(model_path)

    # ── Variables partagées ───────────────────────────────────────────────────
    # Le thread IA pousse maintenant un tuple (display_frame, detections)
    # La display_frame est déjà redimensionnée → le main thread n'a plus
    # qu'à dessiner et appeler imshow (opérations légères)
    latest_result = {"frame": None, "dets": []}
    result_lock   = threading.Lock()
    stop_evt      = threading.Event()
    ai_done       = threading.Event()

    stats = {
        "displayed":       0,
        "ai_processed":    0,
        "ai_skipped":      0,
        "t0":              None,
        "disp_fps_window": deque(maxlen=SMOOTH_WIN),
        "_last_disp_t":    None,
    }

    # ── Queue IA ──────────────────────────────────────────────────────────────
    # On envoie la frame originale (full res) ; le resize display se fait
    # dans le thread IA pour décharger le main thread.
    ai_q = queue.Queue(maxsize=AI_QUEUE_SIZE)

    # ── Signal Ctrl+C ─────────────────────────────────────────────────────────
    def signal_handler(sig, frame_):
        print("\n  [SIGINT] Arrêt demandé")
        stop_evt.set()
    signal.signal(signal.SIGINT, signal_handler)

    # ── Thread IA ─────────────────────────────────────────────────────────────
    def ai_thread():
        while not stop_evt.is_set():
            try:
                frame = ai_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is None:
                break

            # --- Resize display ICI (thread IA, pas le main thread) ----------
            if DISPLAY_SCALE != 1.0:
                display_frame = cv2.resize(frame, (disp_w, disp_h))
            else:
                display_frame = frame.copy()

            # --- Inférence sur la frame originale (pleine résolution) --------
            dets = ai.infer(frame)

            # --- Adapter les coordonnées à l'espace display ------------------
            if DISPLAY_SCALE != 1.0 and dets:
                sx, sy = DISPLAY_SCALE, DISPLAY_SCALE
                dets = [
                    ([int(x*sx), int(y*sy), int(w*sx), int(h*sy)], c, i)
                    for (x, y, w, h), c, i in dets
                ]

            with result_lock:
                latest_result["frame"] = display_frame
                latest_result["dets"]  = dets
                stats["ai_processed"] += 1

        ai_done.set()

    t_ai = threading.Thread(target=ai_thread, daemon=True)
    t_ai.start()

    # ═══════════════════════════════════════════════════════════════════════
    #  BOUCLE PRINCIPALE — uniquement : lire cap + imshow + waitKey
    # ═══════════════════════════════════════════════════════════════════════
    print("  Démarrage...\n")
    stats["t0"] = time.monotonic()

    win_name = "YOLO Real-time"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, disp_w, disp_h)

    try:
        t_video_start = time.perf_counter()
        frame_idx     = 0
        # Première frame display de secours (noire) jusqu'à la 1ère réponse IA
        blank = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)

        while cap.isOpened() and not stop_evt.is_set():
            t_start = time.perf_counter()

            # ── Lecture frame ───────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                break
            cap.read()   # skip une frame pour rester à vitesse source/2

            # ── Alimente l'IA (non-bloquant) ────────────────────────────────
            try:
                ai_q.put_nowait(frame)
            except queue.Full:
                try:
                    ai_q.get_nowait()
                    ai_q.put_nowait(frame)
                except queue.Empty:
                    pass
                stats["ai_skipped"] += 1

            # ── Récupère le résultat préparé par le thread IA ────────────────
            with result_lock:
                display_frame = latest_result["frame"]
                dets          = list(latest_result["dets"])

            # Secours : frame noire tant que l'IA n'a rien produit
            if display_frame is None:
                display_frame = blank.copy()

            # ── Dessin des boxes (coordonnées déjà en espace display) ────────
            ai.draw(display_frame, dets)

            # ── OSD ─────────────────────────────────────────────────────────
            ai_fps   = ai.recent_fps()
            now      = time.perf_counter()
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

            # ── Affichage ────────────────────────────────────────────────────
            cv2.imshow(win_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n  [User] Quit demandé")
                stop_evt.set()
                break

            stats["displayed"] += 1

            # ── Barre de progression ─────────────────────────────────────────
            pct = stats["displayed"] / max(1, total // 2) * 100
            bar = "#" * int(25 * pct / 100) + "-" * (25 - int(25 * pct / 100))
            print(f"\r  [{bar}] {pct:5.1f}%  "
                  f"Disp:{disp_fps:5.1f}fps  AI:{ai_fps:5.1f}fps  "
                  f"IA:{stats['ai_processed']:4d}  skip:{stats['ai_skipped']:3d}",
                  end="", flush=True)

            # ── Régulation framerate (timing absolu, un seul sleep) ──────────
            frame_idx += 1
            target_t   = t_video_start + frame_idx * (2.0 / fps)
            sleep_time = target_t - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"\n[ERREUR] {e}")
        import traceback; traceback.print_exc()
        stop_evt.set()

    finally:
        cv2.destroyAllWindows()

    # ── Finalisation ─────────────────────────────────────────────────────────
    print("\n\n  Finalisation...")
    stop_evt.set()

    while not ai_q.empty():
        try:
            ai_q.get_nowait()
        except queue.Empty:
            break
    ai_q.put(None)

    ai_done.wait(timeout=3.0)
    if t_ai.is_alive():
        print("  [WARN] Thread IA toujours actif, arrêt forcé...")
    t_ai.join(timeout=2.0)
    cap.release()

    elapsed = time.monotonic() - stats["t0"]
    print(f"\n  Terminé !")
    print(f"  Frames affichées : {stats['displayed']}")
    print(f"  Frames IA        : {stats['ai_processed']}")
    print(f"  Skip IA          : {stats['ai_skipped']}")
    print(f"  FPS IA moyen     : {ai.avg_fps():.1f}")
    print(f"  Durée totale     : {elapsed:.1f}s")
    print(f"  Vitesse affichage: {stats['displayed']/elapsed:.1f} FPS")
    print(f"  Vitesse source   : {fps:.1f} FPS")
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