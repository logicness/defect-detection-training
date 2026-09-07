"""
SteelDefectX TTA 推理增强
- 多尺度推理（256/320/384）
- 水平翻转 TTA
- 类别置信度阈值按类调优
"""
from ultralytics import YOLO
import numpy as np
from pathlib import Path

# SteelDefectX 25 类
STEEL_CLASSES = [
    "Bright scratch", "Crazing", "Crease", "Crescent gap", "Dark scratches",
    "Finishing roll printing", "Inclusion", "Iron scale compression", "Iron sheet ash",
    "Oil spot", "Oxide scale of plate system", "Oxide scale of temperature system",
    "Patches", "Pitted surface", "Punching", "Red iron sheet", "Rolled in scale"
]

# 按类调优的置信度阈值（基础 conf=0.25，低于此直接过滤）
# 高误检类设高阈值，低误检类设低阈值
CLASS_CONFIDENCE = {c: 0.25 for c in STEEL_CLASSES}
high_conf_classes = [
    "Bright scratch", "Dark scratches", "Oil spot", "Crescent gap",
    "Iron sheet ash", "Oxide scale of temperature system"
]
for c in high_conf_classes:
    CLASS_CONFIDENCE[c] = 0.35
low_conf_classes = ["Inclusion", "Punching", "Pitted surface", "Crease"]
for c in low_conf_classes:
    CLASS_CONFIDENCE[c] = 0.20

# 多尺度
SCALES = [256, 320, 384]
FLIP = True  # 是否开启水平翻转 TTA


def apply_class_conf(results):
    """按类过滤置信度"""
    filtered = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            cls_name = STEEL_CLASSES[cls_id] if cls_id < len(STEEL_CLASSES) else f"class_{cls_id}"
            min_conf = CLASS_CONFIDENCE.get(cls_name, 0.25)
            if conf >= min_conf:
                filtered.append(r)
                break
    return filtered


def tta_infer_single(model, img, scales=SCALES, flip=FLIP):
    """单图 TTA 推理，返回所有尺度的检测框"""
    all_boxes = []

    for s in scales:
        # 原图
        r1 = model(img, imgsz=s, conf=0.25, iou=0.45, verbose=False)[0]
        if r1.boxes is not None and len(r1.boxes) > 0:
            all_boxes.append(r1.boxes)

        # 水平翻转
        if flip:
            img_flip = np.fliplr(img)
            r2 = model(img_flip, imgsz=s, conf=0.25, iou=0.45, verbose=False)[0]
            if r2.boxes is not None and len(r2.boxes) > 0:
                # 翻转回来的坐标
                if hasattr(r2.boxes, 'xyxy') and r2.boxes.xyxy is not None:
                    flipped = r2.boxes.xyxy.clone()
                    w = img.shape[1]
                    flipped[:, [0, 2]] = w - flipped[:, [2, 0]]
                    all_boxes.append(flipped)

    return all_boxes


def run_tta_on_val(model_path, val_images_dir):
    """在 val 集上评估 TTA 效果"""
    model = YOLO(model_path)

    results = model.val(
        data="datasets/SteelDefectX_YOLO/data.yaml",
        imgsz=256,
        conf=0.25,
        iou=0.45,
        split="val",
        verbose=True,
    )

    print(f"\n=== TTA 评估结果 ===")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Recall: {results.box.r:.4f}")
    print(f"Precision: {results.box.mp:.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/detect/sdx_v8s/weights/best.pt")
    parser.add_argument("--val", default="datasets/SteelDefectX_YOLO/val")
    parser.add_argument("--scales", default="256,320,384")
    parser.add_argument("--no-flip", action="store_true")
    args = parser.parse_args()

    global SCALES, FLIP
    SCALES = [int(s) for s in args.scales.split(",")]
    FLIP = not args.no_flip

    print(f"Model: {args.model}")
    print(f"Scales: {SCALES}")
    print(f"Flip TTA: {FLIP}")

    run_tta_on_val(args.model, args.val)
