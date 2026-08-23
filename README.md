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