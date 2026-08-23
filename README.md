# YOLO Detection — MJPEG Stream (C++)

Real-time object detection (YOLO + OpenVINO + GStreamer + OpenCV) with a
frame-by-frame kinematic predictor and MJPEG streaming via a built-in HTTP
server.

> **⚠️ Migration**: this project previously used a Python script
> (`yolo_detection_recorder.py`, generating an annotated video in
> post-processing). It has now been replaced by an optimized **C++**
> implementation for the Raspberry Pi 4, with live streaming instead of
> video file generation. The Python script is no longer maintained.

---

## Overview

- The AI (OpenVINO) runs in the background at a variable rate (adaptive skip
  based on the measured inference time).
- An internal tracker matches each new AI detection to a tracked object
  (matching via IoU + center distance).
- Once an object has received ≥ 2 AI detections, its velocity is computed
  and each display frame recomputes its predicted position through
  kinematic extrapolation — no AI detection is required on every displayed
  frame.
- The annotated stream is JPEG-encoded and served as MJPEG through a
  built-in HTTP server (no dependency on an external server).

---

## Requirements

### System
```bash
sudo apt install build-essential pkg-config \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-libav \
    libopencv-dev \
    ffmpeg
```

### OpenVINO
The OpenVINO runtime (C API) must be installed and accessible at both
compile time and runtime (headers `openvino/c/openvino.h` + libraries).
See the official Intel documentation for installation on ARM64 /
Raspberry Pi.

---

## Build

```bash
g++ -O3 -std=c++17 yolo_detection_stream.cpp -o yolo_detection_stream \
    $(pkg-config --cflags --libs opencv4) \
    $(pkg-config --cflags --libs gstreamer-1.0) \
    -lopenvino_c -lpthread
```

> Adjust the source filename and OpenVINO include/lib paths according to
> your installation (`-I`, `-L` as needed).

---

## Project Structure

```
onnx_detection_pipeline/
├── yolo_detection_stream.cpp     # main program (C++)
├── yolo11n_256.onnx              # YOLO model (git-ignored)
├── test-video.mp4                # test video (git-ignored)
└── README.md
```


---

## Usage

### Default video (paths defined in the code)
```bash
./yolo_detection_stream
```

### Specify a video or a camera
```bash
./yolo_detection_stream /path/to/video.mp4 /path/to/model.onnx
./yolo_detection_stream /dev/video0 /path/to/model.onnx
```

The program automatically detects whether the input is a camera (path
starting with `/dev/video`) or a video file, and adapts the GStreamer
pipeline accordingly (`v4l2src` vs `filesrc + decodebin`).

### View the stream
Once running, the MJPEG stream is available at: http://<raspberry-ip>:8080/stream

or directly at `http://<raspberry-ip>:8080/` for a minimal HTML page that
displays it.

---

## Configuration

Main parameters at the top of the source file (`static const`, require a
recompilation to change):

| Variable | Default | Description |
|---|---|---|
| `INPUT_SIZE` | `256` | YOLO model input resolution |
| `DISPLAY_SCALE` | `1.0` | Display scaling factor (1.0 = no resize) |
| `JPEG_QUALITY` | `70` | JPEG encoding quality of the MJPEG stream |
| `HTTP_PORT` | `8080` | Built-in MJPEG server port |
| `CONF_KEEP` | `0.25` | Minimum confidence threshold to keep a detection |
| `NMS_THRESHOLD` | `0.45` | IoU threshold for non-maximum suppression |
| `DISPLAY_FPS` | `25.0` | Fixed display/streaming FPS (see note below) |
| `MC_IOU_MATCH` | `0.25` | IoU threshold for AI→object tracker matching |
| `TRACKER_MAX_AGE` | `3` | Missed AI frames before a tracked object is removed |
| `SIZE_EMA_ALPHA` | `0.3` | Exponential smoothing of box sizes |
| `MAX_OBJECTS` | `64` | Max number of simultaneously tracked objects |
| `TARGET_AI_MS` | `120.0` | Target inference time for the adaptive skip |
| `MAX_SKIP` | `8` | Maximum skip between two AI inferences |
| `AI_EMA_ALPHA` | `0.2` | Exponential smoothing of the measured inference time |

> **`DISPLAY_FPS`** drives both the display/streaming rate **and** the
> accuracy of the kinematic predictor (frame ↔ real-time conversion for
> velocity computation). It must match the actual FPS of the source
> (camera or video) for trajectory predictions to remain accurate — this
> is not a purely cosmetic value.

### Profiling

Profiling (average time per stage: capture, resize+AI enqueue, prediction,
drawing, JPEG encoding) is enabled by default via the `ENABLE_PROFILING`
macro at the top of the file. It is printed to `stderr` every ~5 seconds.
Set `#define ENABLE_PROFILING 0` to disable it (recompilation required).

---

## Testing with a simulated camera (no physical camera)

To test the pipeline through the same code path used in production with a
real camera (the `v4l2src` branch of the GStreamer pipeline), without
having a physical camera available, **v4l2loopback** can be used to create
a fake `/dev/videoX` device and feed it an mp4 video in real time.

### 1. Installation

```bash
sudo apt install v4l2loopback-dkms v4l2loopback-utils raspberrypi-kernel-headers
```

> On Raspberry Pi OS, you must use the **Raspberry Pi-specific** headers
> (`raspberrypi-kernel-headers`), not the generic Debian headers
> (`linux-headers-*`), or the DKMS build will fail. Check the version
> matches with `uname -r` and `ls /lib/modules/$(uname -r)/build`.

### 2. Load the module (persistent across reboots)

To avoid having to run `modprobe` on every boot:

```bash
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo 'options v4l2loopback video_nr=30 card_label="FakeCam" exclusive_caps=1' \
    | sudo tee /etc/modprobe.d/v4l2loopback.conf
sudo reboot
```

After rebooting, verify with:
```bash
v4l2-ctl --list-devices
# → should show "FakeCam" on /dev/video30
```

> The device number (`video_nr`) must be chosen freely: on the Raspberry
> Pi, the `bcm2835-codec` and `bcm2835-isp` devices already occupy several
> low numbers (typically 10-23) — check with `v4l2-ctl --list-devices`
> before setting `video_nr`.

### 3. Feed a video into the fake device

```bash
ffmpeg -re -stream_loop -1 -i test-video.mp4 \
    -vf "scale=640:360" \
    -pix_fmt yuyv422 -f v4l2 /dev/video30
```

- `-re`: read at the video's native rate (simulates a live source,
  otherwise ffmpeg pushes frames faster than the real framerate).
- `-stream_loop -1`: infinite loop.
- Leave this command running in a separate terminal during testing.

### 4. Run the program on the fake device

```bash
./yolo_detection_stream /dev/video30 yolo11n_256.onnx
```

### Limitation of this test

The simulated stream does not have the physical constraints of a real
camera sensor: depending on the pipeline's buffering
(`queue max-size-buffers=N`, `sync=false`), bursts of frames can
occasionally push the measured display FPS above the actual source FPS
(25fps here) before stabilizing again. This is not representative of a
real camera (whose throughput is physically capped) — reducing buffering
(`max-size-buffers=1 leaky=downstream`, `sync=true`) brings the behavior
closer to reality if stricter validation is needed.

---

## Getting the YOLO model

```bash
pip install ultralytics --break-system-packages
python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=256)"
```
Place the generated `.onnx` file in the project folder (adjust
`DEFAULT_MODEL` in the code if needed, or pass the path as an argument).