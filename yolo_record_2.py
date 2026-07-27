#!/usr/bin/env python3
"""
YOLO Detection Recorder - avec motion compensation configurable
Usage:
    python3 yolo_record_mc.py
    python3 yolo_record_mc.py output.mp4
    python3 yolo_record_mc.py output.mp4 input.mp4
    python3 yolo_record_mc.py v1.mp4 v2.mp4 v3.mp4

Motion compensation :
    none    → position brute de l'inference (le plus reactif)
    linear  → extrapole la position future entre deux inferences
    forward → compense le mouvement de la camera
"""

import sys
import os
import subprocess
import time
import threading
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GLib

# === CONFIGURATION ===
BASE_DIR       = "/home/comvis/Documents/onnx_detection_pipeline"
DEFAULT_VIDEO  = f"{BASE_DIR}/test-video.mp4"
DEFAULT_MODEL  = f"{BASE_DIR}/yolo11n.onnx"
DEFAULT_OUTPUT = f"{BASE_DIR}/output_mc.mp4"
PLUGIN_PATH    = f"{BASE_DIR}/gst-template/builddir/gst-plugin" 
ONNX_LIB_PATH  = "/opt/onnxruntime/lib"

# ── Motion compensation ───────────────────────────────────────────────────────
# none    → position brute (le plus reactif, pas de decalage)
# linear  → extrapole la position entre deux inferences (lisse le mouvement)
# forward → compense le mouvement de la camera
MOTION_COMPENSATION = "linear"

H264_QP = 20

os.environ["GST_PLUGIN_PATH"] = PLUGIN_PATH
os.environ["LD_LIBRARY_PATH"] = ONNX_LIB_PATH + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["ORT_DISABLE_CUDA"] = "1"

Gst.init(None)


# ─────────────────────────────────────────────────────────────────────────────
#  Compteur FPS inference via pad probe
# ─────────────────────────────────────────────────────────────────────────────
class FpsCounter:
    def __init__(self):
        self._lock  = threading.Lock()
        self._count = 0
        self._fps   = 0.0
        self._t0    = time.monotonic()
        self._total = 0

    def tick(self):
        now = time.monotonic()
        with self._lock:
            self._count += 1
            self._total += 1
            elapsed = now - self._t0
            if elapsed >= 1.0:
                self._fps   = self._count / elapsed
                self._count = 0
                self._t0    = now

    @property
    def fps(self):
        with self._lock:
            return self._fps

    @property
    def total(self):
        with self._lock:
            return self._total


_fps = FpsCounter()

def _probe(pad, info, data):
    _fps.tick()
    return Gst.PadProbeReturn.OK


# ─────────────────────────────────────────────────────────────────────────────
#  Info video
# ─────────────────────────────────────────────────────────────────────────────
def get_video_info(path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10)
        w, h, fps_str = r.stdout.strip().split(",")
        num, den = fps_str.split("/")
        fps = round(int(num) / int(den))
        return int(w), int(h), fps
    except Exception as e:
        print(f"[WARN] fallback 768x432@12fps ({e})")
        return 768, 432, 12


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run(video_path, model_path, output_path):
    width, height, fps = get_video_info(video_path)

    # Verifie outline-color disponible
    _e = Gst.ElementFactory.make("textoverlay", None)
    has_outline = _e is not None and any(
        p.name == "outline-color" for p in _e.list_properties())
    outline = "outline-color=4278190080 " if has_outline else ""
    font_size = max(14, height // 30)

    pipeline_str = (
        f"filesrc location={video_path} ! "
        f"decodebin name=dec "

        # Audio ignore
        f"dec. ! queue ! audioconvert ! fakesink "

        # Video → tee (sans toucher au flux)
        f"dec. ! queue ! videoconvert ! videoscale ! tee name=t "

        # ── Branche 1 : overlay + FPS + encodage ─────────────────────────────
        f"t. ! queue max-size-buffers=10 ! "
        f"videoconvert ! videoscale ! "
        f"video/x-raw,format=RGB,width={width},height={height} ! "
        f"onnxoverlay name=ov motion-compensation={MOTION_COMPENSATION} ! "
        f"videoconvert ! "
        f"textoverlay name=fps_txt "
        f"text=\"IA: -- fps\" "
        f"valignment=top halignment=left "
        f"xpad=10 ypad=10 "
        f"font-desc=\"Monospace Bold {font_size}\" "
        f"color=4278255360 "
        f"{outline}! "
        f"videoconvert ! "
        f"x264enc speed-preset=medium quantizer={H264_QP} ! "
        f"h264parse ! mp4mux ! "
        f"filesink location={output_path} "

        # ── Branche 2 : inference 640x640 → ov.sink_meta ─────────────────────
        f"t. ! queue max-size-buffers=3 leaky=downstream ! "
        f"videoconvert ! videoscale ! "
        f"video/x-raw,format=RGB,width=640,height=640 ! "
        f"onnxinference name=infer model-location={model_path} ! "
        f"onnxpostprocess draw-results=false ! "
        f"ov.sink_meta"
    )

    print("=" * 55)
    print("  YOLO Detection Recorder")
    print("=" * 55)
    print(f"  Entree  : {video_path}")
    print(f"  Modele  : {model_path}")
    print(f"  Sortie  : {output_path}")
    print(f"  Dims    : {width}x{height} @ {fps}fps")
    print(f"  MC      : {MOTION_COMPENSATION}")
    print(f"  H264 QP : {H264_QP}")
    print("=" * 55)
    print("  Traitement...\n")

    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except GLib.Error as e:
        print(f"[ERREUR] Pipeline : {e}")
        return False

    infer = pipeline.get_by_name("infer")
    if infer:
        pad = infer.get_static_pad("src")
        if pad:
            pad.add_probe(Gst.PadProbeType.BUFFER, _probe, None)

    fps_txt = pipeline.get_by_name("fps_txt")

    def _update_overlay():
        if fps_txt and _fps.fps > 0:
            GLib.idle_add(fps_txt.set_property, "text", f"IA: {_fps.fps:.1f} fps")
        return True

    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()
    ok   = [True]

    def on_msg(bus, msg):
        t = msg.type
        if t == Gst.MessageType.EOS:
            print(f"\n\n  Termine ! → {output_path}")
            print(f"  Frames IA : {_fps.total}\n")
            pipeline.set_state(Gst.State.NULL)
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"\n[ERREUR] {err.message}")
            print(f"[DEBUG]  {dbg}")
            ok[0] = False
            pipeline.set_state(Gst.State.NULL)
            loop.quit()

    def on_progress():
        ok_p, pos = pipeline.query_position(Gst.Format.TIME)
        ok_d, dur = pipeline.query_duration(Gst.Format.TIME)
        ia = _fps.fps
        fs = f"{ia:5.1f} fps IA" if ia > 0 else "  --- fps IA"
        if ok_p and ok_d and dur > 0:
            pct = pos / dur * 100
            ps  = pos // Gst.SECOND
            ds  = dur // Gst.SECOND
            f   = int(25 * pct / 100)
            bar = "#" * f + "-" * (25 - f)
            print(f"\r  [{bar}] {pct:5.1f}%  "
                  f"{ps//60}:{ps%60:02d}/{ds//60}:{ds%60:02d}"
                  f"  |  {fs}", end="", flush=True)
        else:
            print(f"\r  demarrage...  |  {fs}", end="", flush=True)
        return True

    bus.connect("message", on_msg)
    GLib.timeout_add(500, on_progress)
    GLib.timeout_add(1000, _update_overlay)
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n  Interrompu.")
        pipeline.set_state(Gst.State.NULL)
        return False

    return ok[0]


# ─────────────────────────────────────────────────────────────────────────────
#  Mode batch
# ─────────────────────────────────────────────────────────────────────────────
def process_batch(video_list, model_path):
    total = len(video_list)
    ok = 0
    for i, vpath in enumerate(video_list):
        if not os.path.exists(vpath):
            print(f"[SKIP] {vpath}")
            continue
        base = os.path.splitext(os.path.basename(vpath))[0]
        out  = os.path.join(os.path.dirname(vpath), f"{base}_yolo.mp4")
        print(f"\n[{i+1}/{total}] {os.path.basename(vpath)}")
        if run(vpath, model_path, out):
            ok += 1
    print(f"\nBatch : {ok}/{total} traitee(s).")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) > 1 and all(a.endswith(".mp4") for a in args):
        process_batch(args, DEFAULT_MODEL)
    else:
        output = args[0] if args else DEFAULT_OUTPUT
        video  = args[1] if len(args) > 1 else DEFAULT_VIDEO

        if not os.path.exists(video):
            print(f"[ERREUR] Video introuvable : {video}")
            sys.exit(1)
        if not os.path.exists(DEFAULT_MODEL):
            print(f"[ERREUR] Modele introuvable : {DEFAULT_MODEL}")
            sys.exit(1)

        run(video, DEFAULT_MODEL, output)