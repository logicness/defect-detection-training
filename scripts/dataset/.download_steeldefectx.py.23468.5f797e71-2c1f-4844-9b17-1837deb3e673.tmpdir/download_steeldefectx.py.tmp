# -*- coding: utf-8 -*-
"""下载 SteelDefectX 数据集 (hf-mirror), 25类与模型0-24类完全一致, 含全部弱类"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

REPO = "Zhaosxian/SteelDefectX"
LOCAL = r"D:\RK3588&Orin Nano\ORIN NANO\Model Training\datasets_master\public_datasets\SteelDefectX"

print(f"Downloading {REPO} -> {LOCAL}")
print("This downloads train/ val/ train_mask/ val_mask/ + metadata json files")

path = snapshot_download(
    repo_id=REPO,
    local_dir=LOCAL,
    repo_type="dataset",
    # 只下载需要的文件，跳过 mask（下载后单独转换）
    # allow_patterns=["train/*", "val/*", "train_mask/*", "val_mask/*", "*.json", "*.md"],
    ignore_patterns=None,
)
print("DONE:", path)