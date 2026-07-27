#!/usr/bin/env python3
"""
YOLO Detection — MJPEG Stream v5
==================================
CORRECTION v5 : fluide 12.5fps, pas de saccades

Base : architecture v2 (proven, 12.5fps, ~57s)
Corrections :
  1. Handler HTTP sans sleep(1/15) — synchronisé par Condition sur
     nouvelle frame. Chaque frame est envoyée une seule fois, immédiatement.
  2. Rattrapage de retard : si sleep_time < -0.04s, on rattrape 20ms
     progressivement pour éviter la dérive temporelle.
  3. _jpeg_seq pour éviter les doublons si le handler est plus rapide
     que l'encodeur (impossible mais sécurisé).
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
JPEG_QUALITY  = 80
HTTP_PORT     = 8080
SMOOTH_WIN    = 15

CONF_APPEAR   = 0.45
CONF_KEEP     = 0.25
NMS_THRESHOLD = 0.45

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
_jpeg_seq   = 0      # incrémenté à chaque nouvelle frame
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
                    "<title>YOLO Stream v5</title>"
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
                        # Attendre qu'une NOUVELLE frame soit disponible
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
                        # PAS DE SLEEP — on attend la prochaine frame sur Condition
                except Exception:
                    pass
            else:
                self.send_response(404)
                self.end_headers()

    return MJPEGHandler


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
    print("  YOLO — MJPEG Stream v5 (fluide, sans saccades)")
    print("=" * 60)
    print(f"  Input   : {video_path}")
    print(f"  Source  : {width}x{height} @ {fps:.1f}fps ({total} frames)")
    print(f"  Display : {disp_w}x{disp_h} (scale {DISPLAY_SCALE})")
    print(f"  JPEG    : qualité {JPEG_QUALITY}")
    print("-" * 60)
    print(f"  >>> Ouvrir dans le navigateur :")
    print(f"  >>> http://{local_ip}:{HTTP_PORT}")
    print("-" * 60)
    print("  Chargement modèle ONNX...")

    ai = YoloInference(model_path)

    # ── Variables partagées ───────────────────────────────────────────────────
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
    }

    ai_q      = queue.Queue(maxsize=AI_QUEUE_SIZE)
    encode_q  = queue.Queue(maxsize=1)

    def signal_handler(sig, frame_):
        print("\n  [SIGINT] Arrêt demandé")
        stop_evt.set()
        with _jpeg_cond:
            _jpeg_cond.notify_all()
    signal.signal(signal.SIGINT, signal_handler)

    # ── Démarrage serveur HTTP dans un thread daemon ──────────────────────────
    server = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), make_handler())
    http_thread = threading.Thread(target=server.serve_forever, daemon=True)
    http_thread.start()
    print(f"  Serveur MJPEG démarré sur le port {HTTP_PORT}")

    # ── Thread IA ─────────────────────────────────────────────────────────────
    def ai_thread():
        while not stop_evt.is_set():
            try:
                frame = ai_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is None:
                break
            dets = ai.infer(frame)
            if DISPLAY_SCALE != 1.0 and dets:
                sx, sy = DISPLAY_SCALE, DISPLAY_SCALE
                dets = [
                    ([int(x*sx), int(y*sy), int(w*sx), int(h*sy)], c, i)
                    for (x, y, w, h), c, i in dets
                ]
            with dets_lock:
                latest_dets[:] = dets
                stats["ai_processed"] += 1
        ai_done.set()

    t_ai = threading.Thread(target=ai_thread, daemon=True)
    t_ai.start()

    # ── Thread encodeur JPEG ─────────────────────────────────────────────────
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
            ok, jpg_buf = cv2.imencode(".jpg", frame_to_encode, encode_params)
            if ok:
                with _jpeg_cond:
                    _jpeg_frame = jpg_buf.tobytes()
                    _jpeg_seq += 1
                    _jpeg_cond.notify_all()

    t_enc = threading.Thread(target=encode_thread, daemon=True)
    t_enc.start()

    # ═════════════════════════════════════════════════════════════════════════
    #  BOUCLE PRINCIPALE — identique v2, avec rattrapage de retard
    # ═════════════════════════════════════════════════════════════════════════
    print("  Démarrage...\n")
    stats["t0"] = time.monotonic()

    try:
        t_video_start = time.perf_counter()
        frame_idx     = 0

        while cap.isOpened() and not stop_evt.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            cap.read()  # skip frame pour rester à vitesse source/2

            # ── Alimente l'IA ────────────────────────────────────────────────
            try:
                ai_q.put_nowait(frame)
            except queue.Full:
                try:
                    ai_q.get_nowait()
                    ai_q.put_nowait(frame)
                except queue.Empty:
                    pass
                stats["ai_skipped"] += 1

            # ── Resize local pour l'affichage ────────────────────────────────
            if DISPLAY_SCALE != 1.0:
                display_frame = cv2.resize(frame, (disp_w, disp_h))
            else:
                display_frame = frame.copy()

            # ── Overlay détections ───────────────────────────────────────────
            with dets_lock:
                dets = list(latest_dets)
            ai.draw(display_frame, dets)

            # ── OSD ──────────────────────────────────────────────────────────
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

            # ── Envoi au thread encodeur (non-bloquant) ──────────────────────
            try:
                encode_q.put_nowait(display_frame)
            except queue.Full:
                try:
                    encode_q.get_nowait()
                    encode_q.put_nowait(display_frame)
                except queue.Empty:
                    pass

            stats["displayed"] += 1

            # ── Progression terminal ──────────────────────────────────────────
            pct = stats["displayed"] / max(1, total // 2) * 100
            bar = "#" * int(25 * pct / 100) + "-" * (25 - int(25 * pct / 100))
            print(f"\r  [{bar}] {pct:5.1f}%  "
                  f"Disp:{disp_fps:5.1f}fps  AI:{ai_fps:5.1f}fps  "
                  f"IA:{stats['ai_processed']:4d}  skip:{stats['ai_skipped']:3d}",
                  end="", flush=True)

            # ── Régulation framerate avec rattrapage ────────────────────────
            frame_idx += 1
            target_t   = t_video_start + frame_idx * (2.0 / fps)
            sleep_time = target_t - time.perf_counter()

            # Rattrapage : si on est en retard de plus de 40ms,
            # on avance le repère de 20ms pour progressivement revenir à l'heure
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

    ai_done.wait(timeout=3.0)
    t_ai.join(timeout=2.0)
    t_enc.join(timeout=2.0)
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
