# -*- coding: utf-8 -*-
"""训练通用金属缺陷模型（SDX 25类 + MVIT 10类 = 35类）

策略：sdx_v8s/best.pt（25类）微调 → 35类
- 用 ultralytics 自动调整输出头（nc 变化时自动重建 detect head）
- imgsz=256（与 SDX 训练一致，MVIT 数据为 640 会 letterbox）
- batch=16, 100 epochs
"""
from ultralytics import YOLO
import os

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    pretrained = r"<LOCAL_MODEL_PATH>/runs/detect/sdx_v8s/weights/best.pt"
    data = r"<LOCAL_MODEL_PATH>/datasets/Universal_Metal/data.yaml"
    out = r"<LOCAL_MODEL_PATH>/runs/detect/universal_metal"

    os.makedirs(out, exist_ok=True)
    print(f"预训练权重: {pretrained}")
    print(f"数据集: {data}")
    print("开始训练 35类通用金属模型 (imgsz=256, 100ep, batch=16)...")

    model = YOLO(pretrained)
    results = model.train(
        data=data,
        epochs=100,
        imgsz=256,
        batch=16,
        device=0,
        project="runs/detect",
        name="universal_metal",
        patience=25,
        workers=4,
        pretrained=True,
        exist_ok=True,
        lr0=0.001,        # 微调用小学习率
        lrf=0.01,
        warmup_epochs=3,
        verbose=True,
    )
    print("训练完成:", results)
