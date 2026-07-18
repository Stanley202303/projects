from pathlib import Path
from ultralytics import YOLO

# Train a one-class tennis-ball segmentation model.
# Dataset layout and YAML are described in README_AI_TRACKER.txt.

ROOT = Path(__file__).resolve().parent
DATASET_YAML = ROOT / "tennis_ball_dataset.yaml"

# A segmentation model learns the ball outline, not only a bounding box.
# Use the smallest model first for speed. Increase to "yolo26s-seg.pt"
# if your GPU and runtime can support it.
BASE_MODEL = "yolo26n-seg.pt"

model = YOLO(BASE_MODEL)

model.train(
    data=str(DATASET_YAML),
    epochs=150,
    imgsz=960,
    batch=-1,
    patience=35,
    cache=True,
    close_mosaic=20,
    degrees=8.0,
    translate=0.15,
    scale=0.60,
    shear=3.0,
    perspective=0.0005,
    flipud=0.15,
    fliplr=0.50,
    hsv_h=0.025,
    hsv_s=0.55,
    hsv_v=0.45,
    mosaic=0.80,
    mixup=0.10,
    copy_paste=0.20,
    project=str(ROOT / "runs"),
    name="tennis_ball_seg",
)

print()
print("Training complete.")
print("Copy runs/tennis_ball_seg/weights/best.pt")
print("to tennis_ball_best.pt beside the runtime tracker.")