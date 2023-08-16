"""https://docs.ultralytics.com/modes/predict/#inference-arguments"""
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run inference on 'bus.jpg' with arguments
model.predict("/Users/sarit/study/try_openai/image_to_text/references/IMG_6127.JPG", save=True, imgsz=320, conf=0.5)
