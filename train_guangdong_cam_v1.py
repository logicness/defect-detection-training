# -*- coding: utf-8 -*-
"""YOLOv8n 检测训练: 伪框(GradCAM 弱监督) + 真实数据.
启动方式: PYTHONPATH=...site-packages python -S train_guangdong_cam_v1.py
注意: 不重定向日志, 让输出进 TaskOutput, 崩溃信息不丢失.
"""
import time
from multiprocessing import freeze_support

DATA = r'<LOCAL_ROOT>/ORIN NANO/dataset/guangdong_detect/data.yaml'
PROJECT = r'<LOCAL_MODEL_PATH>/runs/detect'
NAME = 'guangdong_cam_v1'


def main():
    from ultralytics import YOLO
    print(f'[det训练] model=yolov8n  data={DATA}', flush=True)
    print(f'         epochs=60  imgsz=1024  batch=12  workers=2', flush=True)

    model = YOLO('yolov8n.pt')
    t0 = time.time()
    results = model.train(
        data=DATA,
        epochs=60,
        imgsz=640,          # 伪框是大尺度热区, 640 足够; 1024 太慢(1 epoch 8min)
        batch=16,
        workers=2,
        device=0,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        patience=10,
        cos_lr=True,
        verbose=True,
        seed=42,
    )
    print(f'\n[完成] 耗时 {time.time()-t0:.0f}s', flush=True)
    print(f'权重: {PROJECT}/{NAME}/weights/best.pt', flush=True)


if __name__ == '__main__':
    freeze_support()
    main()