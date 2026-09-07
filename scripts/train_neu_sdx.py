#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2 实验（2026-08-14）：SDX 预训练 → NEU 冻结微调
=================================================
思路：SDX（钢材 25类，7764张）与 NEU（钢材 6类，1260张）同域，
用 SDX best.pt 作预训练权重，冻结 backbone（freeze=10）微调 NEU，
对比 NEU 基线 neu_yolov8s_e1003（test mAP50=0.784）。

用法: E:\Anaconda\envs\yolov11\python.exe scripts\train_neu_sdx.py
产物: runs/neu_sdx_finetune/weights/best.pt
"""
import os
from pathlib import Path

os.environ['YOLO_AMPCHECK'] = '0'

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
SDX_BEST = ROOT / "runs" / "detect" / "sdx_v8s" / "weights" / "best.pt"


def main():
    assert SDX_BEST.exists(), f"SDX 预训练权重不存在: {SDX_BEST}"
    model = YOLO(str(SDX_BEST))
    model.train(
        data=str(ROOT / "neu.yaml"),
        epochs=100,
        imgsz=640,
        batch=16,
        workers=0,          # 8-14 首次启动 DataLoader worker 内存分配失败,改 0 串行更稳
        project=str(ROOT / "runs"),
        name="neu_sdx_finetune",
        seed=42,
        patience=20,
        freeze=10,          # 冻结 backbone（前 10 层）
        lr0=0.001,          # 微调用低学习率
        lrf=0.01,
        cos_lr=True,
        amp=True,
        plots=False,
        exist_ok=True,
        verbose=True,
    )
    print("DONE", model)


if __name__ == "__main__":
    main()
