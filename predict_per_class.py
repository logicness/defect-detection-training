# -*- coding: utf-8 -*-
"""对 val 集逐张 predict, 统计 per-class recall."""
import sys
from pathlib import Path
from multiprocessing import freeze_support

VAL_DIR = Path(r'<LOCAL_ROOT>/ORIN NANO/dataset/guangdong_cls/val')
MODEL = r'<LOCAL_MODEL_PATH>/runs/cls/guangdong_baseline/weights/best.pt'


def main():
    from ultralytics import YOLO
    import collections

    model = YOLO(MODEL)

    # 收集所有 val 图片(扁平列表)
    img_paths = sorted(VAL_DIR.glob('*/*.jpg'))  # 子目录/图片
    if not img_paths:
        img_paths = sorted(VAL_DIR.rglob('*.jpg'))
    print(f'找到 {len(img_paths)} 张val图')

    # 真实类别
    true_labels = [p.parent.name for p in img_paths]

    # 用文件列表 predict
    results = model.predict(source=[str(p) for p in img_paths], imgsz=640, batch=16, device=0, verbose=False)
    pred_labels = [model.names[r.probs.top1] for r in results]

    # per-class recall
    print(f'总样本: {len(true_labels)}')
    by_cls = collections.defaultdict(lambda: [0, 0])  # [correct, total]
    for t, p in zip(true_labels, pred_labels):
        by_cls[t][1] += 1
        if t == p:
            by_cls[t][0] += 1

    # 排序
    items = sorted(by_cls.items(), key=lambda x: -x[1][1])
    print(f'\n{"class":12s} {"correct":>8s} {"total":>6s} {"recall":>8s}')
    for cls, (c, t) in items:
        rec = c / t if t > 0 else 0
        print(f'{cls:12s} {c:8d} {t:6d} {rec:8.3f}')

    # 整体
    total_c = sum(c for c, t in by_cls.values())
    total_n = sum(t for c, t in by_cls.values())
    print(f'\n总体 top1: {total_c/total_n:.4f}  ({total_c}/{total_n})')


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()