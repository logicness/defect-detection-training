#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比评估脚本：基线 vs 增强模型，含 TTA

用法（VS Code 本地跑）：
  cd "D:\RK3568&Orin Nano\ORIN NANO\Model Training"
  E:\Anaconda\envs\yolov11\python.exe scripts/eval_compare.py

会输出各模型的 mAP50 / mAP50-95 / Precision / Recall，以及 TTA 对比。
"""
import os
from pathlib import Path

os.environ['YOLO_AMPCHECK'] = '0'

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = str(ROOT / "datasets" / "NEU_ENHANCED" / "data.yaml")

# 用原始 NEU val 集评估（公平对比）
DATA_ORIG = str(ROOT / "neu.yaml")

MODELS = {
    "baseline_e1003": ROOT / "runs" / "neu_yolov8s_e1003" / "weights" / "best.pt",
    "enhanced_v8s": ROOT / "runs" / "neu_enhanced_v8s" / "weights" / "best.pt",
    "enhanced_v11s": ROOT / "runs" / "neu_enhanced_v11s" / "weights" / "best.pt",
}


def eval_model(name, path, tta=False):
    if not path.exists():
        print(f"  [SKIP] {name}: 模型不存在 {path}")
        return None
    model = YOLO(str(path))
    metrics = model.val(data=DATA_ORIG, augment=tta, verbose=False)
    return {
        'mAP50': metrics.box.map50,
        'mAP50-95': metrics.box.map,
        'precision': metrics.box.mp,
        'recall': metrics.box.mr,
    }


def main():
    print(f"{'='*70}")
    print(f"模型对比评估（原始 NEU val 集）")
    print(f"{'='*70}")

    results = {}
    for name, path in MODELS.items():
        print(f"\n--- {name} ---")
        r = eval_model(name, path, tta=False)
        if r:
            results[name] = r
            print(f"  mAP50={r['mAP50']:.4f}  mAP50-95={r['mAP50-95']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}")

    # 对最优模型跑 TTA
    if results:
        best_name = max(results, key=lambda k: results[k]['mAP50'])
        best_path = MODELS[best_name]
        print(f"\n--- {best_name} + TTA ---")
        r_tta = eval_model(f"{best_name}_tta", best_path, tta=True)
        if r_tta:
            results[f"{best_name}_tta"] = r_tta
            print(f"  mAP50={r_tta['mAP50']:.4f}  mAP50-95={r_tta['mAP50-95']:.4f}  P={r_tta['precision']:.4f}  R={r_tta['recall']:.4f}")

    # 汇总表
    print(f"\n{'='*70}")
    print(f"{'模型':<25} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8}")
    print(f"{'-'*70}")
    baseline = results.get('baseline_e1003', {})
    for name, r in results.items():
        delta = ""
        if baseline and name != 'baseline_e1003' and not name.endswith('_tta'):
            d = r['mAP50'] - baseline.get('mAP50', 0)
            delta = f"  ({'+' if d >= 0 else ''}{d:.4f})"
        print(f"{name:<25} {r['mAP50']:>8.4f} {r['mAP50-95']:>10.4f} {r['precision']:>8.4f} {r['recall']:>8.4f}{delta}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
