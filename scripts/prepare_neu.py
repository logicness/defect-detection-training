#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEU-DET 数据集整理脚本
将源仓库的 train/test 合并后按项目约定 7:2:1 重新划分为 train/val/test。
用法: python scripts/prepare_neu.py
"""
import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "NEU-DET-with-yolov8-main" / "data" / "NEU-DET"
DST = ROOT / "datasets" / "NEU"
SEED = 42
RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}


def main():
    pairs = []
    for split in ["train", "test"]:
        for img in sorted((SRC / split / "images").glob("*.jpg")):
            lbl = SRC / split / "labels" / (img.stem + ".txt")
            if lbl.exists():
                pairs.append((img, lbl))
            else:
                print(f"[warn] 缺标签: {img.name}")

    random.seed(SEED)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * RATIOS["train"])
    n_val = int(n * RATIOS["val"])
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    for name, files in splits.items():
        (DST / "images" / name).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / name).mkdir(parents=True, exist_ok=True)
        for img, lbl in files:
            shutil.copy2(img, DST / "images" / name / img.name)
            shutil.copy2(lbl, DST / "labels" / name / lbl.name)

    print(f"总计 {n} 对样本（seed={SEED}）")
    for name, files in splits.items():
        dist = Counter(img.name.rsplit("_", 1)[0] for img, _ in files)
        print(f"  {name}: {len(files)} 张 | 类别分布 {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
