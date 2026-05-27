# YOLO Detection Recorder

Real-time object detection on video using YOLO + OpenVINO + GStreamer.  
Each frame waits for its own AI result before being encoded — bounding boxes are perfectly synchronized.

---

## Requirements

### System
```bash
sudo apt install python3-gi python3-gi-cairo gir1.2-gst-1.0 \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-libav \
    ffmpeg
```

### Python
```bash
pip install opencv-python openvino matplotlib --break-system-packages
```

---

## Project Structure

```
onnx_detection_gpu/
├── yolo_detection_recorder.py   # main script
├── yolo11n.onnx                 # YOLO model (ignored by git)
├── build/
│   └── test_video2.mp4          # source video (ignored by git)
└── output_vino.mp4              # generated output video
```

> `.onnx`, `.pt`, `build/` and `gst-template/` are excluded from the repository (see `.gitignore`).

---

## Usage

### Default video (paths defined inside the script)
```bash
python3 yolo_detection_recorder.py
```

### Specify an input video
```bash
python3 yolo_detection_recorder.py output.mp4 /path/to/video.mp4
```

### Batch processing (multiple videos)
```bash
python3 yolo_detection_recorder.py video1.mp4 video2.mp4 video3.mp4
```
Output files are created next to each source video with the `_yolo.mp4` suffix.

---

## Generated Outputs

| File | Description |
|---|---|
| `output_vino.mp4` | Annotated video with detection bounding boxes |
| `output_vino_fps_graph.png` | FPS graph (AI / Pipeline / Display) |

---

## Configuration

Main parameters at the top of the script:

| Variable | Default | Description |
|---|---|---|
| `INPUT_SIZE` | `640` | YOLO model input resolution |
| `H264_QP` | `20` | H264 encoding quality (0=best, 51=worst) |
| `CONF_APPEAR` | `0.45` | Confidence threshold to show a detection |
| `CONF_KEEP` | `0.25` | Confidence threshold to keep a detection |
| `SYNC_TIMEOUT_MS` | `80` | Per-frame AI wait timeout (ms) |
| `SMOOTH_WIN` | `30` | FPS smoothing window size for the graph |

---

## Getting the YOLO Model

```bash
pip install ultralytics --break-system-packages
python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx')"
```
This generates `yolo11n.onnx` — place it in the project folder.