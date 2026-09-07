# -*- coding: utf-8 -*-
"""单类标注质量审计：框密度 / 尺寸 / 重叠 / 边距，train vs val 对比

用法 (以 Waist folding class_id=21 为例):
    python audit_annotation.py --labels-train <labels/train> --labels-val <labels/val> \
        --class-id 21 --names-yaml <data.yaml> [--render N --images-train <images/train> --images-val <images/val> --out-dir <dir>]

产出:
    scripts/diag/annotation_audit_<class>.json  + 控制台统计
    可选 --render N: 渲染 N 张带框样本图供目视复核
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np


def load_labels(labels_dir, class_id):
    """返回 [(stem, [boxes...])]，boxes 为 (cx,cy,w,h) 归一化"""
    root = Path(labels_dir)
    found = []
    for fp in root.glob("*.txt"):
        boxes = []
        try:
            for ln in fp.read_text(encoding="utf-8").strip().splitlines():
                p = ln.split()
                if not p:
                    continue
                if int(float(p[0])) == class_id:
                    boxes.append((float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        except Exception:
            continue
        if boxes:
            found.append((fp.stem, boxes))
    return found


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def stats_for(found, name):
    n_img = len(found)
    if n_img == 0:
        return {"images": 0, "boxes": 0, "per_image": {}, "w": [], "h": [], "area": [],
                "overlapping_frac": [], "tiny_frac": 0.0, "aspect": []}
    all_boxes = [b for _, boxes in found for b in boxes]
    n_box = len(all_boxes)
    ws = np.array([b[2] for b in all_boxes])
    hs = np.array([b[3] for b in all_boxes])
    areas = np.array([b[2] * b[3] for b in all_boxes])
    aspects = ws / np.maximum(hs, 1e-6)

    # 每图框数
    per_img = Counter(len(boxes) for _, boxes in found)

    # 重叠: 每张图内 class 内两两 IoU>0.5 的框占比 (碎片化/重复标注信号)
    overlap_fracs = []
    for _, boxes in found:
        n = len(boxes)
        if n < 2:
            overlap_fracs.append(0.0)
            continue
        touched = set()
        for i in range(n):
            for j in range(i + 1, n):
                if box_iou(boxes[i], boxes[j]) > 0.5:
                    touched.add(i)
                    touched.add(j)
        overlap_fracs.append(len(touched) / n)
    overlap_fracs = np.array(overlap_fracs)

    tiny_frac = float((areas < 0.01).mean())  # 面积<1% 的"极小框"占比

    return {
        "images": n_img,
        "boxes": n_box,
        "boxes_per_image_mean": round(n_box / n_img, 2) if n_img else 0,
        "per_image_histogram": {str(k): v for k, v in sorted(per_img.items())},
        "w_pct": {k: round(float(v), 4) for k, v in zip(["p10", "p25", "p50", "p75", "p90"],
                                                        np.percentile(ws, [10, 25, 50, 75, 90]))},
        "h_pct": {k: round(float(v), 4) for k, v in zip(["p10", "p25", "p50", "p75", "p90"],
                                                        np.percentile(hs, [10, 25, 50, 75, 90]))},
        "area_pct": {k: round(float(v), 4) for k, v in zip(["p10", "p25", "p50", "p75", "p90"],
                                                           np.percentile(areas, [10, 25, 50, 75, 90]))},
        "aspect_pct": {k: round(float(v), 2) for k, v in zip(["p10", "p50", "p90"],
                                                             np.percentile(aspects, [10, 50, 90]))},
        "tiny_area_frac": round(tiny_frac, 4),
        "overlap_gt50_frac_mean": round(float(overlap_fracs.mean()), 4),
        "overlap_gt50_frac_max": round(float(overlap_fracs.max()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-train", required=True)
    ap.add_argument("--labels-val", required=True)
    ap.add_argument("--class-id", type=int, required=True)
    ap.add_argument("--names-yaml", required=True)
    ap.add_argument("--render", type=int, default=0)
    ap.add_argument("--images-train", default=None)
    ap.add_argument("--images-val", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    import yaml
    with open(args.names_yaml, "r", encoding="utf-8") as f:
        names = yaml.safe_load(f).get("names", [])
    cls_name = names[args.class_id] if args.class_id < len(names) else f"cls{args.class_id}"

    tr = load_labels(args.labels_train, args.class_id)
    va = load_labels(args.labels_val, args.class_id)
    st_tr = stats_for(tr, "train")
    st_va = stats_for(va, "val")

    out = {"class_id": args.class_id, "class_name": cls_name,
           "train": st_tr, "val": st_va}
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent
    out_json = out_dir / f"annotation_audit_{cls_name.replace(' ', '_')}.json"
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] {out_json}")

    def fmt(d):
        return (f"图={d['images']} 框={d['boxes']} 框/图={d['boxes_per_image_mean']} | "
                f"宽P50={d['w_pct']['p50']} 高P50={d['h_pct']['p50']} 面积P50={d['area_pct']['p50']} | "
                f"极小框(<1%面积)={d['tiny_area_frac']*100:.1f}% | 重叠>0.5均值={d['overlap_gt50_frac_mean']}")

    print(f"\n== {cls_name} (class {args.class_id}) 标注统计 ==")
    print(f"[train] {fmt(st_tr)}")
    print(f"[val]   {fmt(st_va)}")
    print(f"[train] 每图框数分布: {st_tr['per_image_histogram']}")
    print(f"[val]   每图框数分布: {st_va['per_image_histogram']}")

    if args.render and args.render > 0:
        _render(args, tr, va, cls_name, names)


def _render(args, tr, va, cls_name, names):
    import cv2
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent
    out_dir = out_dir / "annotation_renders"
    out_dir.mkdir(parents=True, exist_ok=True)

    def find_img(stem, img_dir):
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            p = Path(img_dir) / (stem + ext)
            if p.exists():
                return p
        return None

    def render_set(found, img_dir, prefix):
        saved = 0
        # 优先挑"框数多"的图，最能暴露标注问题
        found_sorted = sorted(found, key=lambda x: -len(x[1]))
        for stem, boxes in found_sorted:
            if saved >= args.render:
                break
            img_path = find_img(stem, img_dir)
            if img_path is None:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            for (cx, cy, w, h) in boxes:
                x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            out_p = out_dir / f"{prefix}_{stem}.jpg"
            cv2.imwrite(str(out_p), img)
            saved += 1
        return saved

    n1 = render_set(tr, args.images_train, "train")
    n2 = render_set(va, args.images_val, "val")
    print(f"[RENDER] train={n1} val={n2} -> {out_dir}")


if __name__ == "__main__":
    main()