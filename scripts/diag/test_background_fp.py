# -*- coding: utf-8 -*-
"""纯背景负样本误检专项测试：模型在"无缺陷"图上是否会乱框

用法:
    python test_background_fp.py --model <pt> --images <images/val> --labels <labels/val> \
        --names-yaml <data.yaml> [--conf 0.25]

产出:
    scripts/diag/background_fp_<tag>.json  +  控制台汇总
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ["YOLO_AUTOUPDATE"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO  # noqa: E402


def load_names(yaml_path):
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("names", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--names-yaml", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tag", default="prod_v2_b2")
    args = ap.parse_args()

    names = load_names(args.names_yaml)
    img_dir = Path(args.images)
    lab_dir = Path(args.labels)
    lab_stems = {p.stem for p in lab_dir.glob("*.txt")}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    bg_imgs = [p for p in sorted(img_dir.iterdir())
               if p.suffix.lower() in exts and p.stem not in lab_stems]
    print(f"[INFO] 纯背景(无标签)图片数: {len(bg_imgs)}")

    model = YOLO(args.model)
    results = model.predict(
        source=[str(p) for p in bg_imgs],
        imgsz=256, conf=args.conf, iou=0.45, device=0, verbose=False,
    )

    per_img = []
    cls_counter = Counter()
    total_det = 0
    for r, p in zip(results, bg_imgs):
        boxes = r.boxes
        n = len(boxes) if boxes is not None else 0
        classes = [int(c) for c in boxes.cls.tolist()] if boxes is not None else []
        per_img.append({"image": p.name, "detections": n,
                        "classes": classes,
                        "names": [names[c] if c < len(names) else f"cls{c}" for c in classes]})
        total_det += n
        for c in classes:
            cls_counter[c] += 1

    fired = [d for d in per_img if d["detections"] > 0]
    cls_dist = {names[c] if c < len(names) else f"cls{c}": cnt
                for c, cnt in sorted(cls_counter.items(), key=lambda kv: -kv[1])}

    out = {
        "tag": args.tag,
        "model": args.model,
        "conf": args.conf,
        "bg_images": len(bg_imgs),
        "images_with_detection": len(fired),
        "clean_rate": round(1 - len(fired) / len(bg_imgs), 4) if bg_imgs else None,
        "total_detections": total_det,
        "avg_detections_per_image": round(total_det / len(bg_imgs), 4) if bg_imgs else None,
        "class_distribution": cls_dist,
        "per_image": per_img,
    }

    out_json = Path(__file__).resolve().parent / f"background_fp_{args.tag}.json"
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] {out_json}")

    print(f"\n== 纯背景误检测试 ({args.tag}, conf={args.conf}) ==")
    print(f"背景图总数={len(bg_imgs)}")
    print(f"有误检框的图={len(fired)}  (误检率 {len(fired)/len(bg_imgs)*100:.1f}%)")
    print(f"无误检图={len(bg_imgs)-len(fired)}  (干净率 {(1-len(fired)/len(bg_imgs))*100:.1f}%)")
    print(f"误检框总数={total_det}  (平均每图 {total_det/len(bg_imgs):.3f})")
    print("\n误检框类别分布:")
    for n, c in cls_dist.items():
        print(f"  {n:<28} {c}")


if __name__ == "__main__":
    main()