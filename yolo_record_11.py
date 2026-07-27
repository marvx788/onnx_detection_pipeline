#!/usr/bin/env python3

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
import cv2
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")

from gi.repository import Gst, GLib
from openvino import Core

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "/home/comvis/Documents/onnx_detection_pipeline"

DEFAULT_VIDEO = f"{BASE_DIR}/test_video_3.mp4"
DEFAULT_MODEL = f"{BASE_DIR}/yolo11n_320.onnx"
DEFAULT_OUTPUT = f"{BASE_DIR}/output_25fps.mp4"

PLUGIN_PATH = f"{BASE_DIR}/gst-template/builddir/gst-plugin"
ONNX_LIB_PATH = "/opt/onnxruntime/lib"

INPUT_SIZE = 320

STREAM_PORT = 8080

TARGET_FPS = 25
DISPLAY_SIZE = (960, 540)

CONF_APPEAR = 0.45
CONF_KEEP = 0.25
NMS_THRESHOLD = 0.45
MAX_MISSING_FRAMES = 3

SMOOTH_WIN = 30


os.environ["GST_PLUGIN_PATH"] = PLUGIN_PATH
os.environ["LD_LIBRARY_PATH"] = (
    ONNX_LIB_PATH
    + ":"
    + os.environ.get("LD_LIBRARY_PATH", "")
)

Gst.init(None)


# ============================================================
# MJPEG SERVER
# ============================================================

latest_jpeg_frame = None
jpeg_lock = threading.Lock()


class StreamingHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        global latest_jpeg_frame

        if self.path != "/stream.mjpg":
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame"
        )

        self.end_headers()

        try:

            while True:

                with jpeg_lock:
                    frame = latest_jpeg_frame

                if frame:

                    self.wfile.write(
                        b"--frame\r\n"
                    )

                    self.send_header(
                        "Content-Type",
                        "image/jpeg"
                    )

                    self.send_header(
                        "Content-Length",
                        len(frame)
                    )

                    self.end_headers()

                    self.wfile.write(frame)

                    self.wfile.write(
                        b"\r\n"
                    )

                time.sleep(
                    1 / TARGET_FPS
                )

        except Exception:
            pass


class StreamingServer(
    socketserver.ThreadingMixIn,
    HTTPServer
):
    daemon_threads = True
    allow_reuse_address = True




# ============================================================
# YOLO INFERENCE
# ============================================================

class YoloInference:

    def __init__(self, model_path):

        ie = Core()

        model = ie.read_model(
            model_path
        )

        available = ie.available_devices

        device = (
            "GPU"
            if "GPU" in available
            else "CPU"
        )

        print(
            f"[OpenVINO] Device = {device}"
        )

        compiled = ie.compile_model(
            model,
            device_name=device,
            config={
                "CACHE_DIR": "/tmp/ov_cache"
            }
        )

        self.req = (
            compiled
            .create_infer_request()
        )

        self.input_layer = (
            compiled.input(0)
        )

        self.lock = (
            threading.Lock()
        )

        self.recent = deque(
            maxlen=SMOOTH_WIN
        )

        self.total = 0
        self.total_ms = 0

    def infer(
        self,
        frame
    ):

        h, w = frame.shape[:2]

        img = cv2.resize(
            frame,
            (
                INPUT_SIZE,
                INPUT_SIZE
            )
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        tensor = (
            img
            .astype(
                np.float32
            )
            / 255.0
        )

        tensor = np.transpose(
            tensor,
            (
                2,
                0,
                1
            )
        )[None]

        t0 = time.perf_counter()

        self.req.infer(
            {
                self.input_layer:
                tensor
            }
        )

        outputs = (
            self.req
            .get_output_tensor(0)
            .data
        )

        dt = (
            time.perf_counter()
            - t0
        ) * 1000

        with self.lock:

            self.total += 1

            self.total_ms += dt

            self.recent.append(
                dt
            )

        return self.postprocess(
            outputs,
            w,
            h
        )

    def postprocess(
        self,
        outputs,
        fw,
        fh
    ):

        preds = outputs[0]

        boxes = []
        confs = []
        classes = []

        sx = fw / INPUT_SIZE
        sy = fh / INPUT_SIZE

        for i in range(
            preds.shape[1]
        ):

            scores = preds[4:, i]

            cid = int(
                np.argmax(
                    scores
                )
            )

            conf = float(
                scores[cid]
            )

            if conf < CONF_KEEP:
                continue

            cx = preds[0, i]
            cy = preds[1, i]
            bw = preds[2, i]
            bh = preds[3, i]

            x = int(
                (
                    cx
                    - bw / 2
                )
                * sx
            )

            y = int(
                (
                    cy
                    - bh / 2
                )
                * sy
            )

            boxes.append(
                [
                    x,
                    y,
                    int(
                        bw
                        * sx
                    ),
                    int(
                        bh
                        * sy
                    )
                ]
            )

            confs.append(
                conf
            )

            classes.append(
                cid
            )

        if not boxes:
            return []

        idx = cv2.dnn.NMSBoxes(
            boxes,
            confs,
            CONF_KEEP,
            NMS_THRESHOLD
        )

        if len(idx) == 0:
            return []

        return [
            (
                boxes[i],
                confs[i],
                classes[i]
            )
            for i in idx.flatten()
        ]

    def recent_fps(
        self
    ):

        with self.lock:

            if not self.recent:
                return 0

            return (
                1000
                /
                (
                    sum(
                        self.recent
                    )
                    /
                    len(
                        self.recent
                    )
                )
            )


# ============================================================
# TRACKER
# ============================================================

def iou(
    a,
    b
):

    x1 = max(
        a[0],
        b[0]
    )

    y1 = max(
        a[1],
        b[1]
    )

    x2 = min(
        a[2],
        b[2]
    )

    y2 = min(
        a[3],
        b[3]
    )

    inter = (
        max(
            0,
            x2 - x1
        )
        *
        max(
            0,
            y2 - y1
        )
    )

    if inter == 0:
        return 0

    area1 = (
        a[2]
        -
        a[0]
    ) * (
        a[3]
        -
        a[1]
    )

    area2 = (
        b[2]
        -
        b[0]
    ) * (
        b[3]
        -
        b[1]
    )

    return inter / (
        area1
        + area2
        - inter
        + 1e-6
    )


class HysteresisTracker:

    def __init__(self):

        self.tracks = []

    def update(
        self,
        detections
    ):

        out = []

        for (
            box,
            conf,
            cls
        ) in detections:

            if conf < CONF_APPEAR:
                continue

            x, y, w, h = box

            out.append(
                (
                    (
                        x,
                        y,
                        w,
                        h
                    ),
                    conf,
                    cls
                )
            )

        self.tracks = out

        return out


tracker = HysteresisTracker()

# ============================================================
# VIDEO INFO
# ============================================================

def get_video_info(path):

    cap = cv2.VideoCapture(path)

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = int(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    return (
        width,
        height,
        fps,
        total
    )


# ============================================================
# GLOBAL BUFFERS
# ============================================================

frame_lock = threading.Lock()

latest_frame = None
latest_idx = -1

infer_queue = queue.Queue(
    maxsize=2
)

encode_queue = queue.Queue(
    maxsize=8
)

detections = {}

det_lock = (
    threading.Lock()
)


# ============================================================
# APPSINK CALLBACK
# ============================================================

def on_frame(
    sink
):

    sample = (
        sink.emit(
            "pull-sample"
        )
    )

    if not sample:
        return Gst.FlowReturn.ERROR

    buf = sample.get_buffer()

    success, info = buf.map(
        Gst.MapFlags.READ
    )

    if not success:
        return Gst.FlowReturn.ERROR

    frame = (
        np.frombuffer(
            info.data,
            dtype=np.uint8
        )
    )

    caps = (
        sample
        .get_caps()
    )

    s = (
        caps.get_structure(
            0
        )
    )

    width = s.get_value(
        "width"
    )

    height = s.get_value(
        "height"
    )

    frame = frame.reshape(
        (
            height,
            width,
            3
        )
    ).copy()

    buf.unmap(
        info
    )

    idx = (
        on_frame.counter
    )

    on_frame.counter += 1

    global latest_frame
    global latest_idx

    with frame_lock:

        latest_frame = frame
        latest_idx = idx

    if not infer_queue.full():

        infer_queue.put(
            (
                frame.copy(),
                idx
            )
        )

    return Gst.FlowReturn.OK


on_frame.counter = 0


# ============================================================
# AI THREAD
# ============================================================

def ai_thread(
    ai
):

    while True:

        item = (
            infer_queue.get()
        )

        if item is None:
            break

        frame, idx = item

        det = ai.infer(
            frame
        )

        with det_lock:

            detections[
                idx
            ] = det

            while (
                len(
                    detections
                )
                > 10
            ):
                oldest = min(
                    detections
                )

                del detections[
                    oldest
                ]


# ============================================================
# DISPLAY THREAD
# ============================================================

def display_thread(ai):

    global latest_jpeg_frame

    target_dt = 1 / TARGET_FPS

    prev_gray = None

    last_idx = -1
    last_frame = None

    while True:

        tick_start = time.perf_counter()

        with frame_lock:

            idx = latest_idx

            if latest_frame is not None:
                frame = latest_frame.copy()
            else:
                frame = None

        if frame is None:

            time.sleep(target_dt)
            continue

        det = None

        if idx != last_idx:

            with det_lock:

                det = detections.pop(
                    idx,
                    None
                )

            last_idx = idx

            last_frame = frame.copy()

        elif last_frame is not None:

            frame = last_frame.copy()

        ai_frame = frame

        previous = (
            ai_frame.copy()
        )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        if prev_gray is not None and not det:

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                0.5,
                1,
                9,
                2,
                5,
                1.1,
                0
            )

            moved=[]

            for ((x,y,w,h),conf,cls) in tracker.tracks:

                cx=min(
                    flow.shape[1]-1,
                    max(0,x+w//2)
                )

                cy=min(
                    flow.shape[0]-1,
                    max(0,y+h//2)
                )

                dx,dy=flow[cy,cx]

                moved.append(
                    (
                        (
                            int(x+dx),
                            int(y+dy),
                            w,
                            h
                        ),
                        conf,
                        cls
                    )
                )

            tracker.tracks=moved

        prev_gray=gray

        for (
            (
                x,
                y,
                w,
                h
            ),
            conf,
            cls
        ) in tracker.tracks:

            cv2.rectangle(
                ai_frame,
                (
                    x,
                    y
                ),
                (
                    x + w,
                    y + h
                ),
                (
                    0,
                    255,
                    0
                ),
                2
            )

        fps = (
            ai.recent_fps()
        )

        cv2.putText(
            ai_frame,
            f"AI {fps:.1f} FPS",
            (
                20,
                40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (
                0,
                255,
                0
            ),
            2
        )

        jpg = cv2.resize(
            ai_frame,
            DISPLAY_SIZE
        )

        ok, jpg = (
            cv2.imencode(
                ".jpg",
                jpg
            )
        )

        if ok:

            with jpeg_lock:

                latest_jpeg_frame = (
                    jpg.tobytes()
                )

        encode_queue.put(
            ai_frame
        )

        elapsed = (
            time.perf_counter()
            - tick_start
        )

        time.sleep(
            max(
                0,
                target_dt - elapsed
            )
        )


# ============================================================
# ENCODER THREAD
# ============================================================

def encode_thread(
    appsrc
):

    pts = 0

    frame_duration = int(
        Gst.SECOND
        /
        TARGET_FPS
    )

    while True:

        frame = (
            encode_queue.get()
        )

        if frame is None:

            appsrc.emit(
                "end-of-stream"
            )

            break

        buf = (
            Gst.Buffer.new_wrapped(
                frame.tobytes()
            )
        )

        buf.pts = pts
        buf.dts = pts

        buf.duration = (
            frame_duration
        )

        pts += (
            frame_duration
        )

        appsrc.emit(
            "push-buffer",
            buf
        )


# ============================================================
# MAIN
# ============================================================

def run(
    video_path,
    model_path,
    output_path
):

    global latest_jpeg_frame

    width, height, source_fps, total = (
        get_video_info(
            video_path
        )
    )

    print()

    print("=" * 60)

    print(
        "YOLO 25FPS DISPLAY + MOTION COMP"
    )

    print()

    print(
        f"Input : {video_path}"
    )

    print(
        f"Output : {output_path}"
    )

    print(
        f"Stream : http://localhost:{STREAM_PORT}/stream.mjpg"
    )

    print(
        "=" * 60
    )

    ai = YoloInference(
        model_path
    )

    # ====================================================
    # MJPEG SERVER
    # ====================================================

    server = StreamingServer(
        (
            "",
            STREAM_PORT
        ),
        StreamingHandler
    )

    threading.Thread(
        target=server.serve_forever,
        daemon=True
    ).start()

    # ====================================================
    # DESTINATION PIPELINE
    # ====================================================

    dst = Gst.parse_launch(

        (
            "appsrc name=src "
            "format=time "
            "is-live=false "

            "! videoconvert "

            "! x264enc "

            "speed-preset=veryfast "

            "quantizer=20 "

            "! h264parse "

            "! mp4mux faststart=true "

            f"! filesink location={output_path}"
        )

    )

    appsrc = (
        dst.get_by_name(
            "src"
        )
    )

    appsrc.set_property(

        "caps",

        Gst.Caps.from_string(

            (
                f"video/x-raw,"
                f"format=BGR,"
                f"width={width},"
                f"height={height},"
                f"framerate={TARGET_FPS}/1"
            )

        )
    )

    # ====================================================
    # SOURCE PIPELINE
    # ====================================================

    src = Gst.parse_launch(

        (
            f"filesrc location={video_path}"

            " ! decodebin"

            " ! videoconvert"

            f" ! video/x-raw,"
            f"format=BGR"

            " ! appsink"

            " name=sink"

            " emit-signals=true"

            " sync=false"

            " drop=true"

            " max-buffers=4"
        )

    )

    sink = (
        src.get_by_name(
            "sink"
        )
    )

    sink.connect(
        "new-sample",
        on_frame
    )

    # ====================================================
    # THREADS
    # ====================================================

    ai_worker = (
        threading.Thread(
            target=ai_thread,
            args=(ai,),
            daemon=True
        )
    )

    display_worker = (
        threading.Thread(
            target=display_thread,
            args=(ai,),
            daemon=True
        )
    )

    encode_worker = (
        threading.Thread(
            target=encode_thread,
            args=(appsrc,),
            daemon=True
        )
    )

    ai_worker.start()

    display_worker.start()

    encode_worker.start()

    # ====================================================
    # GST LOOP
    # ====================================================

    loop = (
        GLib.MainLoop()
    )

    threading.Thread(
        target=loop.run,
        daemon=True
    ).start()

    # ====================================================
    # START
    # ====================================================

    t0 = time.time()

    dst.set_state(
        Gst.State.PLAYING
    )

    src.set_state(
        Gst.State.PLAYING
    )

    try:

        while True:

            elapsed = (
                time.time()
                - t0
            )

            processed = (
                on_frame.counter
            )

            pct = (
                processed
                /
                max(
                    1,
                    total
                )
                * 100
            )

            print(

                f"\r"

                f"{pct:5.1f}% "

                f"| "

                f"Frames "

                f"{processed}"

                f"/"

                f"{total}"

                f" | "

                f"AI "

                f"{ai.recent_fps():.1f}"

                f" FPS",

                end=""

            )

            if (
                total
                and
                processed
                >= total
            ):
                break

            time.sleep(
                0.25
            )

    except KeyboardInterrupt:

        print(
            "\nStopping..."
        )

    # ====================================================
    # CLEANUP
    # ====================================================

    display_queue.put(
        None
    )

    infer_queue.put(
        None
    )

    ai_worker.join()

    display_worker.join()

    encode_worker.join()

    src.send_event(
        Gst.Event.new_eos()
    )

    dst.send_event(
        Gst.Event.new_eos()
    )

    src.set_state(
        Gst.State.NULL
    )

    dst.set_state(
        Gst.State.NULL
    )

    server.shutdown()

    loop.quit()

    print()

    print()

    print(
        "Done."
    )

    print(
        output_path
    )


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    run(

        DEFAULT_VIDEO,

        DEFAULT_MODEL,

        DEFAULT_OUTPUT

    )