#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 train 集标注修正清单(供人工确认)
对 NEU train 集全量分析:e1003 推理 vs GT
- 预测框 conf>=0.5 且与所有 GT IOU<0.45 → 建议补标(疑似漏标)
- 预测框 conf<0.5 与 GT 无匹配 → 忽略(疑似误检)
- GT 框与所有预测 IOU<0.45 → 建议核查(模型检不出,可能是坏标注/过大/漏标)
- GT 框面积占比 >0.5 → 标注"过大"提醒(疑似过度标注)

产物: runs/label_review/fix_list_train.csv + fix_list_train.txt
"""
import csv
import os
from pathlib import Path

os.environ['YOLO_AMPCHECK'] = '0'

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "runs" / "neu_yolov8s_e1003" / "weights" / "best.pt"
IMG_DIR = ROOT / "datasets" / "NEU" / "images" / "train"
LBL_DIR = ROOT / "datasets" / "NEU" / "labels" / "train"
OUT_CSV = ROOT / "runs" / "label_review" / "fix_list_train.csv"
OUT_TXT = ROOT / "runs" / "label_review" / "fix_list_train.txt"
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
CONF_ADD = 0.9     # 补标置信度阈值(只留高置信,避免 train 集自举放大)
IOU_MATCH = 0.45   # 匹配 IOU 阈值
AREA_OVER = 0.5    # 大面积框占比阈值


def iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def load_gt(img_name: str):
    lbl = LBL_DIR / (Path(img_name).stem + ".txt")
    boxes = []
    if lbl.exists():
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                x1, y1 = (cx - w / 2) * 200, (cy - h / 2) * 200
                x2, y2 = (cx + w / 2) * 200, (cy + h / 2) * 200
                boxes.append((cls, x1, y1, x2, y2))
    return boxes


def fmt_box(b):
    return f"[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]"


def main():
    print("[1/2] 推理 train 集...")
    model = YOLO(str(MODEL_PATH))
    imgs = sorted(IMG_DIR.glob('*'))
    preds = {}
    BATCH = 32
    for i in range(0, len(imgs), BATCH):
        chunk = [str(f) for f in imgs[i:i + BATCH]]
        res = model.predict(chunk, conf=0.25, imgsz=640, verbose=False, device=0)
        for img_path, r in zip(imgs[i:i + BATCH], res):
            boxes = []
            if r.boxes is not None:
                for box, cls, cf in zip(r.boxes.xyxy.cpu().numpy(),
                                        r.boxes.cls.cpu().numpy().astype(int),
                                        r.boxes.conf.cpu().numpy()):
                    boxes.append((int(cls), float(cf), tuple(box)))
            preds[img_path.name] = boxes

    print("[2/2] 组装修正清单...")
    rows = []
    for img_name in imgs:
        name = img_name.name
        gt = load_gt(name)
        pred = preds.get(name, [])
        img_area = 200 * 200

        # 1. 补标候选:高置信预测框与所有 GT 不匹配
        for pcls, pcf, pbox in pred:
            max_iou = max((iou(pbox, g) for g in gt), default=0)
            if max_iou < IOU_MATCH:
                rows.append({
                    "image": name, "issue": "误检(无GT匹配)", "action": "补标" if pcf >= CONF_ADD else "忽略",
                    "cls": CLASS_NAMES[pcls], "conf": f"{pcf:.2f}",
                    "gt_box": "-", "pred_box": fmt_box(pbox),
                    "note": f"模型检出 conf={pcf:.2f} GT无 → {'建议补进标注(高置信漏标)' if pcf >= CONF_ADD else '建议忽略(低置信误检)'}"
                })

        # 2. 核查候选:GT 框与所有预测不匹配
        for gcls, gx1, gy1, gx2, gy2 in gt:
            max_iou = max((iou((gx1, gy1, gx2, gy2), p[2]) for p in pred), default=0)
            area_ratio = abs((gx2 - gx1) * (gy2 - gy1)) / img_area
            if max_iou < IOU_MATCH:
                note = f"GT有框但模型未检出 → 看图确认:真实缺陷保留/误标删除/过大缩小"
                if area_ratio > AREA_OVER:
                    note += f" | ⚠️ 面积占比 {area_ratio:.0%} 过大(疑似过度标注)"
                rows.append({
                    "image": name, "issue": "漏检(GT未匹配)", "action": "核查",
                    "cls": CLASS_NAMES[gcls], "conf": "-",
                    "gt_box": fmt_box((gx1, gy1, gx2, gy2)), "pred_box": "-",
                    "note": note
                })

    # 排序:按图片,补标优先
    rows.sort(key=lambda r: (r['image'], 0 if r['action'] == '补标' else 1))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "issue", "action", "cls", "conf", "gt_box", "pred_box", "note"])
        w.writeheader()
        w.writerows(rows)

    stats = {"补标": 0, "忽略": 0, "核查": 0}
    for r in rows:
        stats[r['action']] += 1
    by_img = {}
    for r in rows:
        by_img.setdefault(r['image'], []).append(r)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("NEU train 集标注修正清单(人工确认用)\n")
        f.write(f"共 {len(rows)} 条 | 补标 {stats['补标']} / 忽略 {stats['忽略']} / 核查 {stats['核查']} | 涉及图 {len(by_img)} 张\n")
        f.write("确认方式:对照 runs/label_review/report_train.html 看图\n")
        f.write("  补标 = 模型高置信(≥0.5)检出但GT无 → GT中新增该框\n")
        f.write("  忽略 = 模型低置信误检 → 不改\n")
        f.write("  核查 = GT框模型没检出 → 看图:保留/删除/缩小(面积占比>50%标⚠️)\n")
        f.write("=" * 80 + "\n\n")
        for img_name, items in sorted(by_img.items()):
            f.write(f"■ {img_name}\n")
            for r in items:
                f.write(f"  [{r['action']}] {r['cls']}  conf={r['conf']}  GT:{r['gt_box']}  Pred:{r['pred_box']}\n")
                f.write(f"       {r['note']}\n")
            f.write("\n")

    print(f"清单已生成: {OUT_TXT}")
    print(f"统计: {stats} | 涉及图 {len(by_img)} 张")
    print(f"CSV: {OUT_CSV}")


if __name__ == '__main__':
    main()
