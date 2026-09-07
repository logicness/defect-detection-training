# -*- coding: utf-8 -*-
"""训练 #2 分块续训器（production v1.0 数据，v11m 底座）

用法：python train_production_v2.py --block <1..15>
- 每块 5ep，规避 forrtl 崩溃；epoch 权重回拷 weights/last.pt 续训
- 底座：v11m_b20 best.pt（训练 #1 最优）
- 数据：production v1.0（UM39+GC10+DAGM negatives, 39 类, 25689 imgs）
"""
import argparse
import glob
import multiprocessing
import os
import shutil

os.environ["YOLO_AUTOUPDATE"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO  # noqa: E402

DATA = r"D:\RK3588&Orin Nano\ORIN NANO\Model Training\datasets_master\production\v1.0\data.yaml"
PROJ_ROOT = r"D:\RK3588&Orin Nano\ORIN NANO\Model Training\runs\detect"
V11M_BEST = r"D:\RK3588&Orin Nano\ORIN NANO\Model Training\runs\detect\univ39_v11m_b20\weights\best.pt"


def latest_weights(main_dir):
    ws = sorted(glob.glob(os.path.join(main_dir, "weights", "epoch*.pt")))
    if ws:
        return ws[-1]
    last = os.path.join(main_dir, "weights", "last.pt")
    return last if os.path.isfile(last) else None


def main():
    multiprocessing.freeze_support()
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, required=True)
    args = ap.parse_args()

    main_dir = os.path.join(PROJ_ROOT, "prod_v2_v11m")
    os.makedirs(os.path.join(main_dir, "weights"), exist_ok=True)

    start = latest_weights(main_dir) or V11M_BEST
    print("=" * 60)
    print(f"[prod_v2 BLOCK {args.block}] 起点: {start}")
    print("=" * 60)
    m = YOLO(start)
    m.train(
        data=DATA, epochs=5, imgsz=256, batch=16, patience=25,
        device=0, workers=0, optimizer="auto", cos_lr=True, amp=True,
        seed=42, close_mosaic=10, save_period=1, plots=False, verbose=True,
        name=f"prod_v2_b{args.block}", project=PROJ_ROOT, exist_ok=True,
        lr0=0.0005,
    )
    src = os.path.join(PROJ_ROOT, f"prod_v2_b{args.block}", "weights")
    eps = sorted(glob.glob(os.path.join(src, "epoch*.pt")))
    if eps:
        shutil.copy2(eps[-1], os.path.join(main_dir, "weights", "last.pt"))
        print(f"[BLOCK {args.block}] 已回拷续训点: {os.path.basename(eps[-1])}")
    print("BLOCK_DONE_MARKER")


if __name__ == "__main__":
    main()
