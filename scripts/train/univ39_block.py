# -*- coding: utf-8 -*-
"""训练 #1 分块续训器（UM39 39 类，v8m 续训 / v11m COCO 起跑）

用法：python train_univ39_block.py --arch v8m|v11m --block <1..20>
- 每块 5ep，规避 forrtl 崩溃；epoch 权重回拷 runs/detect/univ39_<arch>/weights/last.pt 续训
- v8m 起点：生产 35 类权重（ultralytics 自动重建 detect head 至 39 类）
- v11m 起点：yolo11m.pt（COCO 预训练，首次自动下载）
"""
import argparse
import glob
import multiprocessing
import os
import shutil

os.environ["YOLO_AUTOUPDATE"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO  # noqa: E402

DATA = r"D:\RK3588&Orin Nano\ORIN NANO\Model Training\scripts\train\um39_train.yaml"
PROJ_ROOT = r"D:\RK3588&Orin Nano\ORIN NANO\Model Training\runs\detect"
BASE_35 = r"D:\RK3588&Orin Nano\ORIN NANO\Model Training\models\production\um35c_production.pt"  # ASCII 副本（原名含中文，避免 argv 编码问题）


def latest_weights(main_dir):
    ws = sorted(glob.glob(os.path.join(main_dir, "weights", "epoch*.pt")))
    if ws:
        return ws[-1]
    last = os.path.join(main_dir, "weights", "last.pt")
    return last if os.path.isfile(last) else None


def main():
    multiprocessing.freeze_support()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["v8m", "v11m"])
    ap.add_argument("--block", type=int, required=True)
    args = ap.parse_args()

    main_dir = os.path.join(PROJ_ROOT, "univ39_" + args.arch)
    os.makedirs(os.path.join(main_dir, "weights"), exist_ok=True)

    if args.arch == "v8m":
        base = BASE_35
    else:
        base = "yolo11m.pt"  # COCO 预训练，ultralytics 自动下载

    start = latest_weights(main_dir) or base
    print("=" * 60)
    print(f"[univ39_{args.arch} BLOCK {args.block}] 起点: {start}")
    print("=" * 60)
    m = YOLO(start)
    m.train(
        data=DATA, epochs=5, imgsz=256, batch=16, patience=25,
        device=0, workers=0, optimizer="auto", cos_lr=True, amp=True,
        seed=42, close_mosaic=10, save_period=1, plots=False, verbose=True,
        name=f"univ39_{args.arch}_b{args.block}", project=PROJ_ROOT, exist_ok=True,
        lr0=0.0005,
    )
    src = os.path.join(PROJ_ROOT, f"univ39_{args.arch}_b{args.block}", "weights")
    eps = sorted(glob.glob(os.path.join(src, "epoch*.pt")))
    if eps:
        shutil.copy2(eps[-1], os.path.join(main_dir, "weights", "last.pt"))
        print(f"[BLOCK {args.block}] 已回拷续训点: {os.path.basename(eps[-1])}")
    print("BLOCK_DONE_MARKER")


if __name__ == "__main__":
    main()
