from ultralytics import YOLO

# Charger le modèle
model = YOLO("yolo11n.pt")

# Exporter en TFLite avec INT8 quantization
model.export(
    format="tflite",
    imgsz=256,
    int8=True,        # Quantization INT8
    data="coco.yaml", # Dataset pour la calibration
    nms=False,        # <--- CHANGER ICI : Mettre à False pour ARM64 Linux
)