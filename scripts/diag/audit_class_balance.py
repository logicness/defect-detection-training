# -*- coding: utf-8 -*-
"""逐类样本审计：框数 + 覆盖图片数 + 每图框数分布 (train/val 分开)

目的: 区分"样本真少"与"样本多但仍弱"(后者指向类别混淆/标注噪声)

用法:
    python audit_class_balance.py --labels-root <.../labels> [--names-yaml <data.yaml>] [--out <json>]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def load_names(yaml_path):
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("names", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--names-yaml", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = load_names(args.names_yaml)
    nc = len(names)

    root = Path(args.labels_root)
    stats = {}
    for split in ("train", "val"):
        sp = root / split
        if not sp.exists():
            print(f"[WARN] 缺少 {split} 目录: {sp}")
            continue
        imgs = defaultdict(int)      # 类 -> 覆盖图片数
        boxes = defaultdict(int)     # 类 -> 框数
        files = list(sp.glob("*.txt"))
        for fp in files:
            try:
                lines = fp.read_text(encoding="utf-8").strip().splitlines()
            except Exception:
                continue
            pic_classes = set()
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                except Exception:
                    continue
                if 0 <= cid < nc:
                    boxes[cid] += 1
                    pic_classes.add(cid)
            for c in pic_classes:
                imgs[c] += 1
        stats[split] = {
            "label_files": len(files),
            "per_class": {
                names[i]: {"boxes": boxes.get(i, 0), "images": imgs.get(i, 0),
                           "boxes_per_image": round(boxes.get(i, 0) / imgs.get(i, 1), 2)}
                for i in range(nc)
            },
        }

    # 汇总表
    out = {"names": names, "nc": nc, "splits": stats}
    out_json = args.out or str(Path(__file__).resolve().parent / "class_balance_audit.json")
    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] {out_json}")

    print(f"\n{'ID':>3} {'类名':<28}{'train框':>9}{'train图':>9}{'val框':>8}{'val图':>8}{'框/图':>7}")
    print("-" * 82)
    tr = stats.get("train", {}).get("per_class", {})
    va = stats.get("val", {}).get("per_class", {})
    for i, n in enumerate(names):
        t = tr.get(n, {"boxes": 0, "images": 0})
        v = va.get(n, {"boxes": 0, "images": 0})
        bpi = round(t["boxes"] / t["images"], 1) if t["images"] else 0
        print(f"{i:>3} {n:<28}{t['boxes']:>9}{t['images']:>9}{v['boxes']:>8}{v['images']:>8}{bpi:>7}")


if __name__ == "__main__":
    main()