# -*- coding: utf-8 -*-
"""诊断脚本：数值混淆矩阵 + 三类问题拆解（混淆对 / 漏检 / 误报）

用法:
    python diagnose_confusion.py --model <pt> --yaml <yaml> --tag <名称> [--conf 0.25]

产出 (scripts/diag/):
    confusion_<tag>.json   全量数值混淆矩阵
    confusion_<tag>.md     可读报告：混淆对 / 漏检榜 / 误报榜

重要: Ultralytics 的混淆矩阵 orientation 为
    matrix[预测类, 真实类], 最后一行=漏检(FN), 最后一列=误报(FP)
    即: matrix[dc, gc] 匹配; matrix[dc, nc]=FP; matrix[nc, gc]=FN
"""
import argparse
import json
import os
import sys
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
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    names = load_names(args.yaml)
    model = YOLO(args.model)

    out_root = Path(__file__).resolve().parent
    metrics = model.val(
        data=args.yaml,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.45,
        device=0,
        batch=32,
        verbose=False,
        plots=True,
        project=str(out_root / "val_runs"),
        name=args.tag,
    )

    cm_obj = getattr(metrics, "confusion_matrix", None)
    cm = getattr(cm_obj, "matrix", None) if cm_obj is not None else None
    if cm is None:
        print("[ERROR] 无法提取 metrics.confusion_matrix.matrix")
        return

    import numpy as np
    cm = np.asarray(cm, dtype=int)
    nc = cm.shape[0] - 1
    n = cm.shape[0]
    if nc != len(names):
        print(f"[WARN] 矩阵类别数 nc={nc} 与评估集 names 数 {len(names)} 不一致")

    def nm(i):
        return names[i] if i < len(names) else f"cls{i}"

    # ---- 正确语义 ----
    # 行 = 预测类, 列 = 真实类(GT)
    pred_row = cm[:nc, :].sum(axis=1)          # 每类预测总数(含FP)
    gt_col = cm[:, :nc].sum(axis=0)            # 每类GT总数(含FN)
    tp = np.diag(cm[:nc, :nc])                 # 正确匹配
    fp_count = cm[:nc, nc]                     # 误报: 预测为该类但无GT
    fn_count = cm[nc, :nc]                     # 漏检: GT为该类但无预测

    recall = np.where(gt_col > 0, tp / gt_col, np.nan)
    precision = np.where(pred_row > 0, tp / pred_row, np.nan)

    per_class = {}
    for i in range(nc):
        per_class[nm(i)] = {
            "class_id": i,
            "gt": int(gt_col[i]),
            "pred": int(pred_row[i]),
            "tp": int(tp[i]),
            "fn": int(fn_count[i]),
            "fp": int(fp_count[i]),
            "recall": round(float(recall[i]), 4) if not np.isnan(recall[i]) else None,
            "precision": round(float(precision[i]), 4) if not np.isnan(precision[i]) else None,
        }

    # 混淆对: GT类 gc 被判成其它类 dc (dc != gc)
    confusions = []
    for gc in range(nc):
        for dc in range(nc):
            if gc != dc and cm[dc, gc] > 0:
                confusions.append({
                    "gt": gc, "gt_name": nm(gc),
                    "pred": dc, "pred_name": nm(dc),
                    "count": int(cm[dc, gc]),
                    "share": round(float(cm[dc, gc]) / gt_col[gc], 4) if gt_col[gc] > 0 else None,
                })
    confusions.sort(key=lambda x: -x["count"])

    miss_list = sorted(
        [{"class": nm(i), "class_id": i, "fn": int(fn_count[i]), "gt": int(gt_col[i]),
          "miss_rate": round(float(fn_count[i]) / gt_col[i], 4) if gt_col[i] > 0 else None}
         for i in range(nc)],
        key=lambda x: -x["fn"],
    )
    fp_list = sorted(
        [{"class": nm(i), "class_id": i, "fp": int(fp_count[i])} for i in range(nc)],
        key=lambda x: -x["fp"],
    )

    out = {
        "tag": args.tag,
        "model": args.model,
        "yaml": args.yaml,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "nc": nc,
        "names": [nm(i) for i in range(nc)],
        "total_gt_boxes": int(cm[:, :nc].sum()),
        "total_detections": int(cm[:nc, :].sum()),
        "total_fp": int(fp_count.sum()),
        "total_fn": int(fn_count.sum()),
        "matrix": cm.tolist(),
        "per_class": per_class,
        "top_confusions": confusions[:30],
        "miss_ranking": miss_list,
        "fp_ranking": fp_list,
    }

    js_path = out_root / f"confusion_{args.tag}.json"
    js_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] {js_path}")

    md_path = out_root / f"confusion_{args.tag}.md"
    _write_md(md_path, out)
    print(f"[SAVE] {md_path}")

    print(f"\n== {args.tag} (conf={args.conf}) ==")
    print(f"GT总框={out['total_gt_boxes']}  预测总框={out['total_detections']}  FP={out['total_fp']}  FN={out['total_fn']}")
    print("\n[Top 混淆对 GT→错判]")
    for c in confusions[:15]:
        share = f"{c['share']*100:.1f}%" if c["share"] is not None else "NA"
        print(f"  {c['gt_name']:<26} -> {c['pred_name']:<26} {c['count']:>5}  ({share})")
    print("\n[漏检榜 Top12 GT→无框]")
    for m in miss_list[:12]:
        mr = f"{m['miss_rate']*100:.1f}%" if m["miss_rate"] is not None else "NA"
        print(f"  {m['class']:<26} FN={m['fn']:>4}/{m['gt']:<4} 漏检率={mr}")
    print("\n[误报榜 Top12 无GT→框]")
    for fp in fp_list[:12]:
        print(f"  {fp['class']:<26} FP={fp['fp']:>4}")


def _write_md(path, out):
    lines = []
    lines.append(f"# 混淆矩阵诊断报告：{out['tag']}\n")
    lines.append(f"- 模型: `{out['model']}`")
    lines.append(f"- 评估集: `{out['yaml']}`")
    lines.append(f"- conf={out['conf']}  imgsz={out['imgsz']}  类别数={out['nc']}")
    lines.append(f"- GT总框={out['total_gt_boxes']}  预测总框={out['total_detections']}  FP={out['total_fp']}  FN={out['total_fn']}\n")

    lines.append("## 一、混淆对 (GT类被判成其它类, Top30)\n")
    lines.append("| GT类 | 错判为 | 次数 | 占GT类比例 |")
    lines.append("|---|---|---|---|")
    for c in out["top_confusions"]:
        s = f"{c['share']*100:.1f}%" if c["share"] is not None else "NA"
        lines.append(f"| {c['gt_name']} | {c['pred_name']} | {c['count']} | {s} |")

    lines.append("\n## 二、漏检榜 (GT类 → 无框, 按FN数降序)\n")
    lines.append("| 类名 | FN | GT数 | 漏检率 |")
    lines.append("|---|---|---|---|")
    for m in out["miss_ranking"]:
        mr = f"{m['miss_rate']*100:.1f}%" if m["miss_rate"] is not None else "NA"
        lines.append(f"| {m['class']} | {m['fn']} | {m['gt']} | {mr} |")

    lines.append("\n## 三、误报榜 (无GT → 框, 按FP数降序)\n")
    lines.append("| 类名 | FP数 |")
    lines.append("|---|---|")
    for fp in out["fp_ranking"]:
        lines.append(f"| {fp['class']} | {fp['fp']} |")

    lines.append("\n## 四、逐类混淆矩阵统计\n")
    lines.append("| ID | 类名 | GT | 预测 | TP | FN | FP | recall | precision |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i in range(out["nc"]):
        nm = out["names"][i]
        v = out["per_class"][nm]
        lines.append(f"| {i} | {nm} | {v['gt']} | {v['pred']} | {v['tp']} | {v['fn']} | {v['fp']} | {v['recall']} | {v['precision']} |")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()