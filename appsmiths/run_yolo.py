"""Try train the YOLO from scratch."""

import torch
from ultralytics import YOLO

device: str = "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    """Run main function."""
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model.train(data="coco.yaml", epochs=5, device="mps")
    metrics = model.val(data="coco.yaml", device="mps")
    print(metrics)


if __name__ == "__main__":
    main()
