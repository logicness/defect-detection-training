#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, csv, glob
from pathlib import Path
os.environ['YOLO_AMPCHECK'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
ROOT = Path(r"D:\RK3568&Orin Nano\ORIN NANO\Model Training")
B20 = ROOT / "runs" / "detect" / "univ45_b20" / "weights" / "best.pt"
M35 = ROOT / "models" / "production" / "Universal_Metal_35类_mAP50_0.80_通用.pt"
DATA35 = str(ROOT / "datasets" / "Universal_Metal" / "data.yaml")
from ultralytics import YOLO
import multiprocessing
multiprocessing.freeze_support()
def main():
    print("=" * 70)
    print("45 class model evaluation")
    print("=" * 70)
    print("\n=== Training mAP50 trend ===")
    bdirs = sorted(glob.glob(str(ROOT / "runs" / "detect" / "univ45_b*")), key=lambda x: int(os.path.basename(x).split('univ45_b')[-1]))
    print(f"{'Block':<15} {'~Ep':>6} {'mAP50':>8} {'mAP50-95':>9} {'P':>7} {'R':>7}")
    print("-" * 55)
    for bd in bdirs:
        cp = os.path.join(bd, "results.csv")
        if not os.path.exists(cp): continue
        bn = os.path.basename(bd)
        bnum = int(bn.replace("univ45_b", ""))
        with open(cp) as f: rows = list(csv.DictReader(f))
        if not rows: continue
        last = rows[-1]
        eib = len(rows)
        tep = (bnum * 5) if bnum <= 11 else (55 + eib if bnum == 12 else 57 + (bnum - 12) * 5)
        m = float(last.get("metrics/mAP50(B)", 0))
        m9 = float(last.get("metrics/mAP50-95(B)", 0))
        p = float(last.get("metrics/precision(B)", 0))
        r = float(last.get("metrics/recall(B)", 0))
        print(f"{bn:<15} {tep:>6} {m:>8.4f} {m9:>9.4f} {p:>7.4f} {r:>7.4f}")
    print(f"\n=== Model comparison on Universal_Metal 35 val (imgsz=256) ===")
    res = {}
    for name, mp in [("45c_b20_best", B20), ("35c_production", M35)]:
        if not mp.exists(): print(f"  SKIP {name}: {mp} not found"); continue
        print(f"\n  Evaluating {name} ...")
        mdl = YOLO(str(mp))
        mt = mdl.val(data=DATA35, verbose=False, imgsz=256)
        res[name] = {"mAP50": mt.box.map50, "mAP50-95": mt.box.map, "P": mt.box.mp, "R": mt.box.mr}
        print(f"  mAP50={res[name]['mAP50']:.4f} mAP50-95={res[name]['mAP50-95']:.4f} P={res[name]['P']:.4f} R={res[name]['R']:.4f}")
    if res:
        print(f"\n{'='*60}")
        print(f"{'Model':<20} {'mAP50':>8} {'mAP50-95':>10} {'P':>7} {'R':>7}")
        print("-" * 55)
        for n, r in res.items(): print(f"{n:<20} {r['mAP50']:>8.4f} {r['mAP50-95']:>10.4f} {r['P']:>7.4f} {r['R']:>7.4f}")
        if "45c_b20_best" in res and "35c_production" in res:
            d = res["45c_b20_best"]["mAP50"] - res["35c_production"]["mAP50"]
            print(f"\nDelta: {d:+.4f}")
    print(f"\nNOTE: 45c on 35c val -> Al classes(35-44) unannotated, mAP lowered.")
    print(f"b20 results.csv mAP50=0.6613 is on full 45c val (more accurate).")
if __name__ == "__main__": main()