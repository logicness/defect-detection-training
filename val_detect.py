# -*- coding: utf-8 -*-
"""对伪框检测模型 best.pt 做最终 val 评估."""
from multiprocessing import freeze_support

MODEL = r'<LOCAL_MODEL_PATH>/runs/detect/guangdong_cam_v1/weights/best.pt'
DATA = r'<LOCAL_ROOT>/ORIN NANO/dataset/guangdong_detect/data.yaml'


def main():
    from ultralytics import YOLO
    model = YOLO(MODEL)
    m = model.val(data=DATA, imgsz=640, batch=16, device=0, verbose=False, plots=False)
    print(f'\n=== 伪框检测模型 best.pt 最终评估 ===')
    print(f'mAP50:   {m.box.map50:.4f}')
    print(f'mAP50-95: {m.box.map:.4f}')
    print(f'Precision: {m.box.mp:.4f}')
    print(f'Recall:   {m.box.mr:.4f}')
    # per class
    print(f'\n=== 各类别 (AP50) ===')
    names = list(m.names.values())
    for i, n in enumerate(names):
        ap = m.box.ap50[i]
        if not (ap is None):
            print(f'  {n:12s}  AP50={ap:.3f}')


if __name__ == '__main__':
    freeze_support()
    main()