# -*- coding: utf-8 -*-
"""碎片标注框合并：把某类密集小框合并成粗框（几何连通分量法）

只用于"副本验证"，绝不写回原数据。

用法:
    python merge_fragmented_labels.py --labels-train <labels/train> --labels-val <labels/val> \
        --class-id 21 --margin 0.03 --out-root <dir>

逻辑:
    对每张图内 class-id 的所有框，把"膨胀 margin 后相互重叠"的框归为同一组，
    每组输出一个外接粗框（保持原类别）。其它类别的框原样保留。
"""
import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def merge_boxes(boxes, margin):
    n = len(boxes)
    if n <= 1:
        return boxes
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def overlap(a, b):
        ax1 = a[0] - a[2] / 2 - margin
        ax2 = a[0] + a[2] / 2 + margin
        ay1 = a[1] - a[3] / 2 - margin
        ay2 = a[1] + a[3] / 2 + margin
        bx1 = b[0] - b[2] / 2 - margin
        bx2 = b[0] + b[2] / 2 + margin
        by1 = b[1] - b[3] / 2 - margin
        by2 = b[1] + b[3] / 2 + margin
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    for i in range(n):
        for j in range(i + 1, n):
            if overlap(boxes[i], boxes[j]):
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(boxes[i])

    merged = []
    for g in groups.values():
        x1 = min(b[0] - b[2] / 2 for b in g)
        x2 = max(b[0] + b[2] / 2 for b in g)
        y1 = min(b[1] - b[3] / 2 for b in g)
        y2 = max(b[1] + b[3] / 2 for b in g)
        cx = min(max((x1 + x2) / 2, 0.0), 1.0)
        cy = min(max((y1 + y2) / 2, 0.0), 1.0)
        w = min(x2 - x1, 1.0)
        h = min(y2 - y1, 1.0)
        merged.append((round(cx, 6), round(cy, 6), round(max(w, 1e-4), 6), round(max(h, 1e-4), 6)))
    return merged


def process_file(fp, class_id, margin):
    lines = fp.read_text(encoding="utf-8").strip().splitlines()
    others = []
    targets = []
    for ln in lines:
        p = ln.split()
        if not p:
            continue
        if int(float(p[0])) == class_id:
            targets.append((float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        else:
            others.append(ln)
    if not targets:
        return None  # 无该类的文件不写
    merged = merge_boxes(targets, margin)
    out_lines = list(others)
    for (cx, cy, w, h) in merged:
        out_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(out_lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-train", required=True)
    ap.add_argument("--labels-val", required=True)
    ap.add_argument("--class-id", type=int, required=True)
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    stats = {}
    for split, src in (("train", args.labels_train), ("val", args.labels_val)):
        src_dir = Path(src)
        dst_dir = out_root / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        n_files = 0
        n_orig_boxes = 0
        n_merged_boxes = 0
        for fp in src_dir.glob("*.txt"):
            content = process_file(fp, args.class_id, args.margin)
            if content is None:
                continue
            # 统计
            for ln in fp.read_text(encoding="utf-8").strip().splitlines():
                p = ln.split()
                if p and int(float(p[0])) == args.class_id:
                    n_orig_boxes += 1
            for ln in content.strip().splitlines():
                p = ln.split()
                if p and int(float(p[0])) == args.class_id:
                    n_merged_boxes += 1
            (dst_dir / fp.name).write_text(content, encoding="utf-8")
            n_files += 1
        stats[split] = {"files": n_files, "orig_boxes": n_orig_boxes, "merged_boxes": n_merged_boxes}
        print(f"[{split}] 文件={n_files}  原框={n_orig_boxes}  合并后框={n_merged_boxes}  (压缩 {n_orig_boxes - n_merged_boxes})")

    print(f"\n[SAVE] 合并结果已写入副本: {out_root}")
    print("注意: 这仅是验证副本，原数据未改动。请用 audit_annotation.py 指向该副本复核统计。")


if __name__ == "__main__":
    main()