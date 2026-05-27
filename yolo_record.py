#!/usr/bin/env python3
"""
YOLO Detection Recorder - conserve les dimensions de la vidéo source
Usage:
    python3 yolo_record_mc.py                        # vidéo par défaut
    python3 yolo_record_mc.py output.mp4             # sortie personnalisée
    python3 yolo_record_mc.py output.mp4 input.mp4   # entrée + sortie custom
    python3 yolo_record_mc.py v1.mp4 v2.mp4 v3.mp4  # mode batch
"""

import sys
import os
import subprocess
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GLib

# === CONFIGURATION ===
BASE_DIR       = "/home/marvin/Documents/onnx_detection/onnx_detection_gpu"
DEFAULT_VIDEO  = f"{BASE_DIR}/build/test_video.mp4"
DEFAULT_MODEL  = f"{BASE_DIR}/yolo11n.onnx"
DEFAULT_OUTPUT = f"{BASE_DIR}/output_yolo_mc.mp4"
PLUGIN_PATH    = f"{BASE_DIR}/gst-template/builddir/gst-plugin"
ONNX_LIB_PATH  = "/opt/onnxruntime/lib"

MOTION_COMPENSATION = "linear"   # none | forward | linear
TRACKER_ALGORITHM   = "sort"     # iou | sort

os.environ["GST_PLUGIN_PATH"] = PLUGIN_PATH
os.environ["LD_LIBRARY_PATH"] = ONNX_LIB_PATH + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["ORT_DISABLE_CUDA"] = "1"

Gst.init(None)


def get_video_dimensions(video_path):
    """Récupère width et height de la vidéo via ffprobe."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            video_path
        ], capture_output=True, text=True, timeout=10)
        w, h = result.stdout.strip().split(",")
        return int(w), int(h)
    except Exception as e:
        print(f"[WARN] Impossible de lire les dimensions ({e}), fallback 1280x720")
        return 1280, 720


def run(video_path, model_path, output_path):
    # Récupérer les dimensions source
    width, height = get_video_dimensions(video_path)
    print(f"  Dimensions source : {width}x{height}")

    pipeline_str = (
        f"filesrc location={video_path} ! "
        f"decodebin name=dec "
        f"dec. ! queue ! audioconvert ! fakesink "
        f"dec. ! queue ! videoconvert ! videoscale ! videorate ! "
        f"video/x-raw,framerate=25/1 ! tee name=t "

        # Branche vidéo : conserve les dimensions source
        f"t. ! queue max-size-buffers=8 ! "
        f"videoconvert ! videoscale ! "
        f"video/x-raw,format=RGB,width={width},height={height} ! "
        f"onnxoverlay name=ov motion-compensation={MOTION_COMPENSATION} ! "
        f"videoconvert ! "
        f"x264enc speed-preset=ultrafast tune=zerolatency ! "
        f"h264parse ! mp4mux ! filesink location={output_path} "

        # Branche inférence : toujours 640x640 pour YOLO
        f"t. ! queue max-size-buffers=3 leaky=downstream ! "
        f"videoconvert ! videoscale ! "
        f"video/x-raw,format=RGB,width=640,height=640 ! "
        f"onnxinference model-location={model_path} ! "
        f"onnxpostprocess draw-results=false ! "
        f"onnxtracker tracker-algorithm={TRACKER_ALGORITHM} ! "
        f"ov.sink_meta"
    )

    print("=" * 55)
    print("  YOLO Detection Recorder — Motion Compensation")
    print("=" * 55)
    print(f"  Entree   : {video_path}")
    print(f"  Modele   : {model_path}")
    print(f"  Sortie   : {output_path}")
    print(f"  Dims     : {width}x{height} (conservees)")
    print(f"  MC       : {MOTION_COMPENSATION}")
    print(f"  Tracker  : {TRACKER_ALGORITHM}")
    print("=" * 55)
    print("  Traitement en cours...\n")

    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except GLib.Error as e:
        print(f"[ERREUR] Pipeline : {e}")
        return False

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    success = [True]

    def on_message(bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            print(f"\n\n  Termine ! Fichier : {output_path}\n")
            pipeline.set_state(Gst.State.NULL)
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"\n[ERREUR] {err.message}")
            print(f"[DEBUG]  {debug}")
            success[0] = False
            pipeline.set_state(Gst.State.NULL)
            loop.quit()

    def update_progress():
        ok_pos, pos = pipeline.query_position(Gst.Format.TIME)
        ok_dur, dur = pipeline.query_duration(Gst.Format.TIME)
        if ok_pos and ok_dur and dur > 0:
            pct   = pos / dur * 100
            pos_s = pos // Gst.SECOND
            dur_s = dur // Gst.SECOND
            filled = int(30 * pct / 100)
            bar = "#" * filled + "-" * (30 - filled)
            print(f"\r  [{bar}] {pct:5.1f}%  "
                  f"{pos_s//60}:{pos_s%60:02d} / {dur_s//60}:{dur_s%60:02d}",
                  end="", flush=True)
        return True

    bus.connect("message", on_message)
    GLib.timeout_add(500, update_progress)
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n  Interrompu.")
        pipeline.set_state(Gst.State.NULL)
        return False

    return success[0]


def process_batch(video_list, model_path):
    total = len(video_list)
    ok = 0
    for i, vpath in enumerate(video_list):
        if not os.path.exists(vpath):
            print(f"[SKIP] Introuvable : {vpath}")
            continue
        base = os.path.splitext(os.path.basename(vpath))[0]
        out  = os.path.join(os.path.dirname(vpath), f"{base}_yolo_mc.mp4")
        print(f"\n[{i+1}/{total}] {os.path.basename(vpath)}")
        if run(vpath, model_path, out):
            ok += 1
    print(f"\nBatch : {ok}/{total} video(s) traitee(s).")


if __name__ == "__main__":
    args = sys.argv[1:]

    # Mode batch : tous les args sont des .mp4
    if len(args) > 1 and all(a.endswith(".mp4") for a in args):
        print(f"Mode batch : {len(args)} video(s)\n")
        process_batch(args, DEFAULT_MODEL)
    else:
        output = args[0] if len(args) > 0 else DEFAULT_OUTPUT
        video  = args[1] if len(args) > 1 else DEFAULT_VIDEO
        if not os.path.exists(video):
            print(f"[ERREUR] Video introuvable : {video}")
            sys.exit(1)
        if not os.path.exists(DEFAULT_MODEL):
            print(f"[ERREUR] Modele introuvable : {DEFAULT_MODEL}")
            sys.exit(1)
        run(video, DEFAULT_MODEL, output)