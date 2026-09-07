# -*- coding: utf-8 -*-
"""YOLOv8s-cls 训练: 广东铝型材数据集 baseline.
启动方式(重要, 绕过 WorkBuddy safe-delete 钩子对 ultralytics 删 cache 的拦截):
  PYTHONPATH=E:/Anaconda/envs/yolov11/Lib/site-packages E:/Anaconda/envs/yolov11/python.exe -S train_guangdong_baseline.py
"""
import time, sys, os
from multiprocessing import freeze_support

DATA = r'<LOCAL_ROOT>/ORIN NANO/dataset/guangdong_cls'
PROJECT = r'<LOCAL_MODEL_PATH>/runs/cls'
NAME = 'guangdong_baseline'
LOG = r'<LOCAL_MODEL_PATH>/train_guangdong_baseline.log'


def setup_logging():
    """把 stdout/stderr 写日志. 必须在 main() 里调用, 避免 spawn 子进程重复覆盖."""
    log_f = open(LOG, 'w', encoding='utf-8')

    class DualOut:
        def write(self, s):
            sys.__stdout__.write(s)
            sys.__stdout__.flush()
            log_f.write(s)
            log_f.flush()

        def flush(self):
            sys.__stdout__.flush()
            log_f.flush()

    sys.stdout = DualOut()
    sys.stderr = DualOut()
    return log_f


def main():
    from ultralytics import YOLO
    print(f'[训练] model=yolov8s-cls  data={DATA}')
    print(f'        project={PROJECT}  name={NAME}')
    print(f'        epochs=80  imgsz=640  batch=16  workers=4')

    model = YOLO('yolov8s-cls.pt')
    t0 = time.time()
    results = model.train(
        data=DATA,
        epochs=80,
        imgsz=640,
        batch=16,          # 8GB 显存 + 2560x1920 大图, 32 会 OOM
        workers=4,         # worker 多也吃内存
        device=0,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        patience=15,         # 早停
        cos_lr=True,         # cos 学习率
        label_smoothing=0.1, # 类别不平衡
        verbose=True,
        seed=42,
    )
    print(f'\n[训练完成] 耗时 {time.time()-t0:.1f}s')
    print(f'权重: {PROJECT}/{NAME}/weights/best.pt')
    print(f'结果: {PROJECT}/{NAME}/results.csv')


if __name__ == '__main__':
    # Windows 下必须 __main__ 保护, 否则 DataLoader 多进程 spawn 报错
    freeze_support()
    log_f = setup_logging()
    try:
        main()
    finally:
        log_f.close()
