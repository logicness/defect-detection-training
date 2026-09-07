# -*- coding: utf-8 -*-
"""对 best.pt 跑 val, 拿 per-class 指标 + 看样本尺度."""
import sys
from ultralytics import YOLO
from pathlib import Path
from multiprocessing import freeze_support

DATA = r'<LOCAL_ROOT>/ORIN NANO/dataset/guangdong_cls'
MODEL = r'<LOCAL_MODEL_PATH>/runs/cls/guangdong_baseline/weights/best.pt'


def main():
    print(f'[val] model={MODEL}')
    model = YOLO(MODEL)
    metrics = model.val(data=DATA, imgsz=640, batch=16, device=0, verbose=True)
    print(f'\n=== 整体 ===')
    print(f'top1_acc: {metrics.top1:.4f}')
    print(f'top5_acc: {metrics.top5:.4f}')

    # per-class
    import numpy as np
    cm = metrics.confusion_matrix.matrix  # NxN
    names = list(metrics.names.values())
    print(f'\n=== 各类 ===')
    print(f'{"class":12s} {"correct":>8s} {"total":>6s} {"recall":>8s}')
    for i, n in enumerate(names):
        total = cm.sum(axis=1)[i]
        correct = cm[i][i]
        rec = correct / total if total > 0 else 0
        print(f'{n:12s} {int(correct):8d} {int(total):6d} {rec:8.3f}')
    print(f'\n各类 val 样本数(列):')
    for i, n in enumerate(names):
        n_val = cm[:, i].sum()
        print(f'  {n:12s} {int(n_val):4d}')


if __name__ == '__main__':
    freeze_support()
    main()