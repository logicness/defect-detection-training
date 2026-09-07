# -*- coding: utf-8 -*-
"""
合并 SteelDefectX(25类学术) + MVIT(14类真实) → 通用金属缺陷数据集(31类)
- SDX 25 类保留原 id 0-24
- MVIT 4 类映射到 SDX 语义最近类（共享类，不新增）
- MVIT 10 类独立保留（id 25-34 中非共享部分）

类别映射（MVIT -> SDX）：
  scratch -> 4  (Dark scratches)
  weld_slag -> 20 (Slag inclusion)
  inclusion -> 6 (Inclusion)
其余 10 类独立新增。

共享类数据并入 SDX 对应类，独立类作为新类追加。
"""
import os, shutil, random
from collections import defaultdict

random.seed(42)

SDX = r"<LOCAL_MODEL_PATH>/_archive_20260811/datasets/SteelDefectX_YOLO_25类_7764张"
MVIT_ROOT = r"<LOCAL_MODEL_PATH>/_archive_20260811/datasets/MVIT_3子集_1810张_真实产线/ready"
OUT = r"<LOCAL_MODEL_PATH>/datasets/Universal_Metal"

SDX_NAMES = ['Bright scratch','Crazing','Crease','Crescent gap','Dark scratches',
'Finishing roll printing','Inclusion','Iron scale compression','Iron sheet ash','Oil spot',
'Oxide scale of plate system','Oxide scale of temperature system','Patches','Pitted surface','Punching',
'Red iron sheet','Rolled in scale','Rolled pit','Secondary rust skin','Silk spot','Slag inclusion',
'Waist folding','Water spot','Welding line','White rust']

# MVIT 14 类 → (目标id, 目标名)
# 映射到 SDX: scratch→4(Dark scratches), weld_slag→20(Slag inclusion), inclusion→6(Inclusion)
MVIT_MAP = {
    # casting_billet (6类)
    "scratch": (4, "Dark scratches"),
    "weld_slag": (20, "Slag inclusion"),
    "cutting_opening": (25, "cutting_opening"),
    "water_slag_mark": (26, "water_slag_mark"),
    "slag_skin": (27, "slag_skin"),
    "longitudinal_crack": (28, "longitudinal_crack"),
    # steel_pipe (4类)
    "warp": (29, "warp"),
    "external_fold": (30, "external_fold"),
    "wrinkle": (31, "wrinkle"),
    # mhpsds (4类)
    "inclusion": (6, "Inclusion"),
    "blocky_scale": (32, "blocky_scale"),
    "striated_scale": (33, "striated_scale"),
    "foreign_obj": (34, "foreign_obj"),
}

# 最终 35 个槽位（25 SDX + 10 MVIT 独立），实际 nc=35
FINAL_NAMES = SDX_NAMES + [None]*10
for mvit_name, (tid, tname) in MVIT_MAP.items():
    if tid >= 25:
        FINAL_NAMES[tid] = tname
NC = len([n for n in FINAL_NAMES if n])
# 紧凑化：去掉空洞（实际 id 从 25 开始连续）
print(f"最终类别数: {NC}")
for i, n in enumerate(FINAL_NAMES):
    if n: print(f"  {i}: {n}")

def build():
    os.makedirs(f"{OUT}/train/images", exist_ok=True)
    os.makedirs(f"{OUT}/train/labels", exist_ok=True)
    os.makedirs(f"{OUT}/val/images", exist_ok=True)
    os.makedirs(f"{OUT}/val/labels", exist_ok=True)

    # 1) SDX 数据：原样拷贝（train/val 分开）
    for split in ["train", "val"]:
        src_img = f"{SDX}/{split}/images"
        src_lbl = f"{SDX}/{split}/labels"
        if not os.path.isdir(src_img):
            print(f"跳过 SDX {split}: 目录不存在")
            continue
        for fn in sorted(os.listdir(src_img)):
            if not fn.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            base = os.path.splitext(fn)[0]
            lbl = f"{src_lbl}/{base}.txt"
            if not os.path.isfile(lbl):
                continue
            shutil.copy2(f"{src_img}/{fn}", f"{OUT}/{split}/images/sdx_{fn}")
            shutil.copy2(lbl, f"{OUT}/{split}/labels/sdx_{base}.txt")
        print(f"  SDX {split}: 拷贝完成")

    # 2) MVIT 三子集：重映射类别 id 后并入
    stats = defaultdict(int)
    for sub, yaml_name in [("casting_billet", "casting_billet"),
                            ("steel_pipe", "steel_pipe"),
                            ("mhpsds", "mhpsds")]:
        # 读取 yaml 拿类别顺序
        import yaml
        y = yaml.safe_load(open(f"{MVIT_ROOT}/{yaml_name}.yaml", encoding="utf-8"))
        names_map = y.get("names", {})
        # names 可能是 dict {0:..} 或 list
        if isinstance(names_map, dict):
            names_map = {int(k): v for k, v in names_map.items()}
        else:
            names_map = {i: v for i, v in enumerate(names_map)}
        for split in ["train", "val"]:
            src_img = f"{MVIT_ROOT}/{sub}/images/{split}"
            src_lbl = f"{MVIT_ROOT}/{sub}/labels/{split}"
            if not os.path.isdir(src_img):
                print(f"跳过 {sub} {split}: {src_img} 不存在")
                continue
            for fn in sorted(os.listdir(src_img)):
                if not fn.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue
                base = os.path.splitext(fn)[0]
                lbl = f"{src_lbl}/{base}.txt"
                if not os.path.isfile(lbl):
                    continue
                # 读标签重映射
                new_lines = []
                for line in open(lbl, encoding="utf-8"):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    old_cls = int(parts[0])
                    mvit_name = names_map.get(old_cls)
                    if mvit_name is None:
                        continue
                    if mvit_name not in MVIT_MAP:
                        print(f"  ⚠ {sub} 未知类别 {old_cls}={mvit_name}，跳过")
                        continue
                    new_cls, _ = MVIT_MAP[mvit_name]
                    new_lines.append(f"{new_cls} " + " ".join(parts[1:]))
                if not new_lines:
                    continue
                stats[mvit_name] += 1
                shutil.copy2(f"{src_img}/{fn}", f"{OUT}/{split}/images/mvit_{sub}_{fn}")
                with open(f"{OUT}/{split}/labels/mvit_{sub}_{base}.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(new_lines) + "\n")
        print(f"  MVIT {sub}: 完成")
    print("\nMVIT 各类别样本数:", dict(stats))

    # 3) 写 data.yaml
    yaml_path = f"{OUT}/data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {OUT}\n")
        f.write("train: train/images\nval: val/images\n")
        f.write(f"nc: {NC}\n")
        names_list = [n for n in FINAL_NAMES if n]
        f.write("names: " + str(names_list).replace("'", '"') + "\n")
    print(f"\n写入 {yaml_path}")

    # 统计
    for split in ["train", "val"]:
        n_img = len([f for f in os.listdir(f"{OUT}/{split}/images") if f.lower().endswith((".jpg",".png"))])
        n_lbl = len(os.listdir(f"{OUT}/{split}/labels"))
        print(f"{split}: {n_img} 图 / {n_lbl} 标签")

if __name__ == "__main__":
    build()
