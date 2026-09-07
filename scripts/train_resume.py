#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8s × NEU 缺陷数据集 续训（从 e1003/last.pt resume）
用法: python scripts/train_resume.py
产物: runs/neu_yolov8s_e1003/weights/best.pt
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    last_pt = ROOT / "runs" / "neu_yolov8s_e1003" / "weights" / "last.pt"
    model = YOLO(str(last_pt))
    model.train(
        data=str(ROOT / "neu.yaml"),
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,
        project=str(ROOT / "runs"),
        name="neu_yolov8s_e1003",
        seed=42,
        patience=20,
        amp=False,
        resume=True,
        verbose=True,
    )
    metrics = model.val()
    print(f"[RESULT] mAP@0.5 = {metrics.box.map50:.4f}")
    print(f"[RESULT] mAP@0.5:0.95 = {metrics.box.map:.4f}")
    print(f"[RESULT] best.pt = {ROOT / 'runs' / 'neu_yolov8s_e1003' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
