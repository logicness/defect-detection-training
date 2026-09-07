"""MVIT 三子集 v8s 基线对比训练"""
import os
os.environ["YOLO_AMPCHECK"] = "0"
from ultralytics import YOLO

JOBS = [
    ("casting_billet", "datasets/MVIT/ready/casting_billet.yaml", "runs/mvit_casting"),
    ("steel_pipe", "datasets/MVIT/ready/steel_pipe.yaml", "runs/mvit_pipe"),
    ("mhpsds", "datasets/MVIT/ready/mhpsds.yaml", "runs/mvit_plate"),
]

for name, yaml_path, project in JOBS:
    print(f"\n===== 训练 {name} =====", flush=True)
    m = YOLO("yolov8s.pt")
    m.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=8,
        workers=0,
        project=project,
        name="train",
        patience=30,
        seed=42,
        plots=False,  # 避免 val 阶段字体绘图崩溃(字体已修复,双保险)
    )
    print(f"===== {name} 训练完成 =====", flush=True)

print("\n全部训练完成", flush=True)
