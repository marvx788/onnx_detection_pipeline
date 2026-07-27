from ultralytics import YOLO

model = YOLO("yolo11n.pt")   # ou le chemin vers ton .pt custom
model.export(format="openvino", imgsz=256, int8=True, data="coco128.yaml")  
# → crée un dossier  yolo11n_ncnn_model/