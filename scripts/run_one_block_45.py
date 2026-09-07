# -*- coding: utf-8 -*-
"""45 类通用金属模型训练（Universal_Metal 35类 + APDDD 铝型材 10类）

从 35 类权重继续训练（扩头到 45 类），imgsz=256，分块续训（5ep/块，规避 forrtl）
用法（示例，第 1 块）：
  python run_one_block_45.py <block_id>
"""
import os, sys, shutil, glob
os.environ['YOLO_AUTOUPDATE'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO
import multiprocessing

DATA = r"D:\RK3568&Orin Nano\ORIN NANO\Model Training\datasets\Universal_Metal_45\data.yaml"
PROJ_ROOT = r"D:\RK3568&Orin Nano\ORIN NANO\Model Training\runs\detect"
# 35 类生产权重（起点）
BASE_35 = r"D:\RK3568&Orin Nano\ORIN NANO\Model Training\models\production\Universal_Metal_35类_mAP50_0.80_通用.pt"
MAIN_DIR = os.path.join(PROJ_ROOT, "universal_metal_45")


def latest_weights():
    ws = sorted(glob.glob(os.path.join(MAIN_DIR, "weights", "epoch*.pt")))
    if ws:
        return ws[-1]
    return os.path.join(MAIN_DIR, "weights", "last.pt")


def main():
    multiprocessing.freeze_support()
    bid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sub = "univ45_b%d" % bid
    # 起点：主目录 last.pt（含训练状态），否则 35 类权重
    model_path = latest_weights()
    if not os.path.isfile(model_path):
        model_path = BASE_35
    print("=" * 50)
    print("[BLOCK %d] 加载 %s" % (bid, model_path))
    print("=" * 50)
    m = YOLO(model_path)
    m.train(
        data=DATA, epochs=5, imgsz=256, batch=16, patience=25,
        device=0, workers=0, optimizer='auto', cos_lr=True, amp=True,
        seed=42, close_mosaic=10, save_period=1, plots=False, verbose=True,
        name=sub, project=PROJ_ROOT, exist_ok=True,
        lr0=0.0005,
    )
    # 拷回含优化器状态的 epoch 权重作为续训起点
    src = os.path.join(PROJ_ROOT, sub, "weights")
    os.makedirs(os.path.join(MAIN_DIR, "weights"), exist_ok=True)
    eps = sorted(glob.glob(os.path.join(src, "epoch*.pt")))
    if eps:
        shutil.copy2(eps[-1], os.path.join(MAIN_DIR, "weights", "last.pt"))
        print("[BLOCK %d] 已拷回 %s" % (bid, os.path.basename(eps[-1])))
    print("BLOCK_DONE_MARKER")


if __name__ == "__main__":
    main()
