"""Try train the YOLO from scratch."""

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

model.train(data="config.yaml", epochs=100)
metrics = model.val()
