#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEU 数据集标注质量检查脚本
用 e1003 best.pt 跑 val 集，找出预测与标注不一致的可疑样本
重点检查 background 与 rolled-in_scale 类的标注问题

用法（VS Code 本地跑）:
  cd "D:\RK3568&Orin Nano\ORIN NANO\Model Training"
  E:\Anaconda\envs\yolov11\python.exe scripts\check_labels.py

产物: runs/label_check/suspicious.csv
"""
import os
import csv
from pathlib import Path
from collections import Counter, defaultdict

os.environ['YOLO_AMPCHECK'] = '0'

from ultralytics import YOLO
import cv2

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "runs" / "neu_yolov8s_e1003" / "weights" / "best.pt"
DATA_YAML = ROOT / "neu.yaml"
VAL_IMG_DIR = ROOT / "datasets" / "NEU" / "images" / "val"
VAL_LBL_DIR = ROOT / "datasets" / "NEU" / "labels" / "val"
OUT_DIR = ROOT / "runs" / "label_check"
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
CONF_THRESH = 0.25  # 用默认推理阈值
IOU_THRESH = 0.45


def iou(box1, box2):
    """计算两个 xyxy 框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def load_label(lbl_path, img_w, img_h):
    """加载 YOLO 格式标签，返回 xyxy + cls 列表"""
    boxes = []
    if not lbl_path.exists():
        return boxes
    with open(lbl_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 加载模型: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # 跑验证集推理
    print(f"[INFO] 运行验证集推理 (conf={CONF_THRESH}, iou={IOU_THRESH})...")
    results = model.predict(
        source=str(VAL_IMG_DIR),
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        save=False,
        save_txt=False,
        verbose=False,
    )

    suspicious_records = []
    stats = {
        'total_images': 0,
        'total_gt_boxes': 0,
        'total_pred_boxes': 0,
        'matched': 0,
        'gt_missed': 0,       # GT 框未被匹配（漏检）
        'pred_extra': 0,      # 预测框无匹配 GT（误检）
        'cls_mismatch': 0,    # 类别不匹配
    }
    cls_confusion = defaultdict(int)  # (gt_cls, pred_cls) -> count
    per_image_issues = []

    for r in results:
        img_path = Path(r.path)
        img_name = img_path.name
        img_w = r.orig_shape[1]
        img_h = r.orig_shape[0]
        stats['total_images'] += 1

        # GT 标签
        lbl_path = VAL_LBL_DIR / (img_path.stem + ".txt")
        gt_boxes = load_label(lbl_path, img_w, img_h)
        stats['total_gt_boxes'] += len(gt_boxes)

        # 预测框
        pred_boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            for i in range(len(r.boxes)):
                cls = int(r.boxes.cls[i])
                conf = float(r.boxes.conf[i])
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                pred_boxes.append((cls, conf, x1, y1, x2, y2))
        stats['total_pred_boxes'] += len(pred_boxes)

        # 匹配：贪心最大 IoU
        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)

        for pi, (p_cls, p_conf, p_x1, p_y1, p_x2, p_y2) in enumerate(pred_boxes):
            best_iou = 0
            best_gi = -1
            for gi, (g_cls, g_x1, g_y1, g_x2, g_y2) in enumerate(gt_boxes):
                if gt_matched[gi]:
                    continue
                v = iou((p_x1, p_y1, p_x2, p_y2), (g_x1, g_y1, g_x2, g_y2))
                if v > best_iou:
                    best_iou = v
                    best_gi = gi

            if best_gi >= 0 and best_iou >= 0.3:
                gt_matched[best_gi] = True
                pred_matched[pi] = True
                stats['matched'] += 1
                g_cls = gt_boxes[best_gi][0]
                if p_cls != g_cls:
                    stats['cls_mismatch'] += 1
                    cls_confusion[(CLASS_NAMES[g_cls], CLASS_NAMES[p_cls])] += 1
                    suspicious_records.append({
                        'image': img_name,
                        'issue': '类别不匹配',
                        'gt_class': CLASS_NAMES[g_cls],
                        'pred_class': CLASS_NAMES[p_cls],
                        'conf': f'{p_conf:.3f}',
                        'iou': f'{best_iou:.3f}',
                    })
            # pred_matched[pi] = True  # 已在上面设置

        # 漏检 GT
        for gi, matched in enumerate(gt_matched):
            if not matched:
                stats['gt_missed'] += 1
                g_cls = CLASS_NAMES[gt_boxes[gi][0]]
                suspicious_records.append({
                    'image': img_name,
                    'issue': '漏检(GT未匹配)',
                    'gt_class': g_cls,
                    'pred_class': '-',
                    'conf': '-',
                    'iou': '-',
                })

        # 误检 pred
        for pi, matched in enumerate(pred_matched):
            if not matched:
                stats['pred_extra'] += 1
                p_cls = CLASS_NAMES[pred_boxes[pi][0]]
                p_conf = pred_boxes[pi][1]
                suspicious_records.append({
                    'image': img_name,
                    'issue': '误检(无GT匹配)',
                    'gt_class': '-',
                    'pred_class': p_cls,
                    'conf': f'{p_conf:.3f}',
                    'iou': '-',
                })

        img_issues = sum(1 for r2 in suspicious_records if r2['image'] == img_name)
        if img_issues > 0:
            per_image_issues.append((img_name, img_issues))

    # 输出 CSV
    csv_path = OUT_DIR / "suspicious.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'issue', 'gt_class', 'pred_class', 'conf', 'iou'])
        writer.writeheader()
        writer.writerows(suspicious_records)

    # 输出摘要
    print(f"\n{'='*60}")
    print(f"NEU 数据集标注质量检查报告")
    print(f"{'='*60}")
    print(f"模型: {MODEL_PATH}")
    print(f"验证集: {VAL_IMG_DIR}")
    print(f"阈值: conf={CONF_THRESH}, iou={IOU_THRESH}")
    print(f"")
    print(f"图片数: {stats['total_images']}")
    print(f"GT 框: {stats['total_gt_boxes']}")
    print(f"预测框: {stats['total_pred_boxes']}")
    print(f"匹配: {stats['matched']}")
    print(f"漏检: {stats['gt_missed']}")
    print(f"误检: {stats['pred_extra']}")
    print(f"类别不匹配: {stats['cls_mismatch']}")
    print(f"")
    print(f"可疑记录总数: {len(suspicious_records)}")
    print(f"CSV 已保存: {csv_path}")

    # 按问题类型统计
    issue_counts = Counter(r['issue'] for r in suspicious_records)
    print(f"\n按问题类型:")
    for issue, count in issue_counts.most_common():
        print(f"  {issue}: {count}")

    # 按类别统计漏检
    miss_by_cls = Counter(r['gt_class'] for r in suspicious_records if r['issue'] == '漏检(GT未匹配)')
    print(f"\n漏检按 GT 类别:")
    for cls, count in miss_by_cls.most_common():
        print(f"  {cls}: {count}")

    # 按类别统计误检
    extra_by_cls = Counter(r['pred_class'] for r in suspicious_records if r['issue'] == '误检(无GT匹配)')
    print(f"\n误检按预测类别:")
    for cls, count in extra_by_cls.most_common():
        print(f"  {cls}: {count}")

    # 类别混淆 top 10
    if cls_confusion:
        print(f"\n类别混淆 top 10 (GT -> Pred: count):")
        for (g, p), count in sorted(cls_confusion.items(), key=lambda x: -x[1])[:10]:
            print(f"  {g} -> {p}: {count}")

    # 问题最多的图片 top 10
    if per_image_issues:
        per_image_issues.sort(key=lambda x: -x[1])
        print(f"\n问题最多的图片 top 10:")
        for name, count in per_image_issues[:10]:
            print(f"  {name}: {count} 个问题")

    print(f"\n{'='*60}")
    print(f"建议: 优先检查问题最多的图片，重点复核 background/rolled-in_scale 标注")


if __name__ == "__main__":
    main()
