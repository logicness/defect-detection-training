#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEU_FIXED 修复数据集训练(标注补标实验)
与 e1003 基线完全同配置(epochs=100/batch=16/imgsz=640/seed=42/patience=20),
仅数据集换成 NEU_FIXED(conf>=0.9 补标 213 框),保证可比

产物: runs/neu_fixed_v8s/weights/best.pt
"""
import os
from pathlib import Path

os.environ['YOLO_AMPCHECK'] = '0'

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    model = YOLO(str(ROOT / "yolov8s.pt"))
    model.train(
        data=str(ROOT / "datasets" / "NEU_FIXED" / "data.yaml"),
        epochs=100,
        imgsz=640,
        batch=16,
        workers=2,
        project=str(ROOT / "runs"),
        name="neu_fixed_v8s",
        seed=42,
        patience=20,
        amp=False,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.0,
        verbose=True,
    )
    metrics = model.val()
    print(f"\n{'='*60}")
    print(f"[RESULT] mAP@0.5 = {metrics.box.map50:.4f}")
    print(f"[RESULT] mAP@0.5:0.95 = {metrics.box.map:.4f}")
    print(f"[RESULT] best.pt = {ROOT / 'runs' / 'neu_fixed_v8s' / 'weights' / 'best.pt'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
