# -*- coding: utf-8 -*-
"""通用模型评估器：单模型 → val 指标 JSON（含每类 mAP50/mAP50-95/P/R）
用法：python eval_model.py --model <pt> --yaml <yaml> --tag <名称> [--save <json>]
产出：控制台表格 + JSON（供训练前后对比，回答"运行后有提升吗"）
"""
import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os_env = __import__("os")
os_env.environ["YOLO_AUTOUPDATE"] = "0"
os_env.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO  # noqa: E402


def load_names(yaml_path):
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("names", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save", default=None)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()

    names = load_names(args.yaml)
    model = YOLO(args.model)
    metrics = model.val(data=args.yaml, imgsz=args.imgsz, conf=args.conf,
                        iou=args.iou, device=0, batch=32, verbose=False, plots=False)

    box = metrics.box
    map50 = float(box.map50) if hasattr(box, "map50") else float(box.map50())
    map5095 = float(box.map) if hasattr(box, "map") else float(box.map())
    per = {}
    try:
        ap50 = box.ap50          # 每类 AP@50
        ap = box.ap               # 每类 AP@50-95
        p = box.p
        r = box.r
        cls_idx = box.ap_class_index
        for ci, cid in enumerate(cls_idx):
            name = names[int(cid)] if int(cid) < len(names) else f"cls{int(cid)}"
            per[name] = {
                "class_id": int(cid),
                "map50": round(float(ap50[ci]), 4),
                "map50_95": round(float(ap[ci]), 4),
                "precision": round(float(p[ci]), 4),
                "recall": round(float(r[ci]), 4),
            }
    except Exception as e:
        print(f"[warn] 每类指标提取失败: {e}")

    out = {"tag": args.tag, "model": args.model, "yaml": args.yaml,
           "map50": round(map50, 4), "map50_95": round(map5095, 4),
           "per_class": per}
    print(f"\n== {args.tag} ==")
    print(f"mAP50={out['map50']}  mAP50-95={out['map50_95']}  类数={len(per)}")
    print(f"{'类名':<28}{'mAP50':>8}{'mAP50-95':>10}{'P':>8}{'R':>8}")
    for n, v in sorted(per.items(), key=lambda kv: kv[1]["class_id"]):
        print(f"{n:<28}{v['map50']:>8}{v['map50_95']:>10}{v['precision']:>8}{v['recall']:>8}")
    save = args.save or f"scripts/eval/result_{args.tag}.json"
    Path(save).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON -> {save}")


if __name__ == "__main__":
    main()
