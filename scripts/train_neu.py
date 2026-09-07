#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8s × NEU 缺陷数据集 — 基线训练（生产模型）
指标: mAP50=0.772 / Recall=0.709（100 epoch, patience 提前停于 84）
用法: E:\Anaconda\envs\yolov11\python.exe scripts\train_neu.py
产物: runs/neu_yolov8s_e1003/weights/best.pt
"""
import os
from pathlib import Path

os.environ['YOLO_AMPCHECK'] = '0'

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    model = YOLO(str(ROOT / "yolov8s.pt"))
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
        verbose=True,
    )
    metrics = model.val()
    print(f"[RESULT] mAP@0.5 = {metrics.box.map50:.4f}")
    print(f"[RESULT] mAP@0.5:0.95 = {metrics.box.map:.4f}")
    print(f"[RESULT] best.pt = {ROOT / 'runs' / 'neu_yolov8s_e1003' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
