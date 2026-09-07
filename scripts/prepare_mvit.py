"""MVIT 金属数据集准备:组织 YOLO 布局 + 80/20 划分"""
import random
import shutil
from pathlib import Path

random.seed(42)

RAW = Path("datasets/MVIT/raw")
OUT = Path("datasets/MVIT/ready")

SUBSETS = {
    "casting_billet": {
        "src": RAW / "casting_billet" / "casting_billet",
        "names": ["scratch", "weld_slag", "cutting_opening", "water_slag_mark", "slag_skin", "longitudinal_crack"],
    },
    "steel_pipe": {
        "src": RAW / "steel_pipe" / "steel_pipe",
        "names": ["warp", "external_fold", "wrinkle", "scratch"],
    },
    "mhpsds": {
        "src": RAW / "MHPSDS" / "MHPSDS",
        "names": ["inclusion", "blocky_scale", "striated_scale", "foreign_obj"],
    },
}

for sub, cfg in SUBSETS.items():
    src = cfg["src"]
    img_dir = src / "images"
    lbl_dir = src / "labels"

    # 收集 images(递归,兼容 MHPSDS 子目录)
    images = sorted(img_dir.rglob("*.jpg")) + sorted(img_dir.rglob("*.png"))
    items = []
    skipped_empty = 0
    for img in images:
        matches = list(lbl_dir.rglob(img.stem + ".txt"))
        if not matches:
            continue
        txt = matches[0]
        lines = [l for l in txt.read_text().strip().splitlines() if l.strip()]
        if not lines:  # 空标注图跳过(无缺陷图对检测训练无意义且影响划分)
            skipped_empty += 1
            continue
        items.append((img, txt))

    random.shuffle(items)
    n_val = max(1, int(len(items) * 0.2))
    val_set, train_set = items[:n_val], items[n_val:]

    for split, lst in [("train", train_set), ("val", val_set)]:
        o_img = OUT / sub / "images" / split
        o_lbl = OUT / sub / "labels" / split
        o_img.mkdir(parents=True, exist_ok=True)
        o_lbl.mkdir(parents=True, exist_ok=True)
        for img, txt in lst:
            shutil.copy2(img, o_img / img.name)
            shutil.copy2(txt, o_lbl / txt.name)

    print(f"[{sub}] 总 {len(items)} 张(跳空标注 {skipped_empty}) | train {len(train_set)} / val {len(val_set)}")
    print(f"  类别: {cfg['names']}")

# 写 yaml
for sub, cfg in SUBSETS.items():
    yaml_path = OUT / f"{sub}.yaml"
    content = (
        f"path: {str((OUT / sub).resolve())}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(cfg["names"]))
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"yaml: {yaml_path}")
print("全部完成")
