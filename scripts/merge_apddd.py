# -*- coding: utf-8 -*-
"""
合并 APDDD(铝型材10类检测) → Universal_Metal(35类) → 45 类通用金属数据集
- 现有 35 类保留原 id 0-34
- APDDD 10 类独立新增 id 35-44（铝型材域，与钢材类语义不同不映射）

APDDD 原始格式：
- 图片: APSPC1/APSPC2/*.jpg（2560×1920，共 1885 张）
- 标注: APSPC-Annotations/Annotations/*.xml（VOC 格式，name=英文类名）
"""
import os, shutil, random, glob
from collections import defaultdict

random.seed(42)

UNIV = r"<LOCAL_MODEL_PATH>/datasets/Universal_Metal"
APDDD = r"<LOCAL_MODEL_PATH>/datasets/APDDD_raw"
OUT = r"<LOCAL_MODEL_PATH>/datasets/Universal_Metal_45"

# APDDD 10 类 → id（从 35 开始）
APDDD_CLASSES = {
    'aoxian': 35, 'budaodian': 36, 'cahua': 37, 'jupi': 38, 'loudi': 39,
    'pengshang': 40, 'qikeng': 41, 'tufen': 42, 'tucengkailie': 43, 'zangdian': 44,
}

# 45 类全名（0-34 现有 + 35-44 APDDD）
FINAL_NAMES = ['Bright scratch','Crazing','Crease','Crescent gap','Dark scratches',
'Finishing roll printing','Inclusion','Iron scale compression','Iron sheet ash','Oil spot',
'Oxide scale of plate system','Oxide scale of temperature system','Patches','Pitted surface','Punching',
'Red iron sheet','Rolled in scale','Rolled pit','Secondary rust skin','Silk spot','Slag inclusion',
'Waist folding','Water spot','Welding line','White rust',
'cutting_opening','water_slag_mark','slag_skin','longitudinal_crack','warp','external_fold','wrinkle',
'blocky_scale','striated_scale','foreign_obj',
'aoxian','budaodian','cahua','jupi','loudi','pengshang','qikeng','tufen','tucengkailie','zangdian']


def parse_voc_xml(xml_path):
    """解析 VOC XML → [(cls_name, x1,y1,x2,y2)]"""
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    W, H = int(size.find('width').text), int(size.find('height').text)
    objs = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bb = obj.find('bndbox')
        x1 = float(bb.find('xmin').text)
        y1 = float(bb.find('ymin').text)
        x2 = float(bb.find('xmax').text)
        y2 = float(bb.find('ymax').text)
        objs.append((name, x1, y1, x2, y2))
    return W, H, objs


def main():
    # 1. 复制现有 35 类数据（结构：train/images + train/labels）
    if not os.path.isdir(os.path.join(UNIV, "train", "images")):
        print("未找到 Universal_Metal:", UNIV)
        return
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(OUT, sub, split), exist_ok=True)
        # images
        src_img = os.path.join(UNIV, split, "images")
        if os.path.isdir(src_img):
            for f in os.listdir(src_img):
                if f.lower().endswith((".jpg", ".png", ".bmp")):
                    shutil.copy2(os.path.join(src_img, f), os.path.join(OUT, "images", split, f))
        # labels
        src_lab = os.path.join(UNIV, split, "labels")
        if os.path.isdir(src_lab):
            for f in os.listdir(src_lab):
                if f.endswith(".txt"):
                    shutil.copy2(os.path.join(src_lab, f), os.path.join(OUT, "labels", split, f))

    # 2. APDDD：找图片位置映射
    img_map = {}
    for d in ("APSPC1", "APSPC2"):
        for f in os.listdir(os.path.join(APDDD, d)):
            if f.lower().endswith(".jpg"):
                img_map[f] = os.path.join(APDDD, d, f)
    print(f"APDDD 图片: {len(img_map)} 张")

    # 3. 转换 XML → YOLO，划分 train/val(8:2)
    xmls = sorted(glob.glob(os.path.join(APDDD, "APSPC-Annotations", "Annotations", "*.xml")))
    random.shuffle(xmls)
    n_val = max(1, int(len(xmls) * 0.2))
    val_set = set(os.path.basename(x) for x in xmls[:n_val])
    print(f"APDDD 标注: {len(xmls)} 个, val: {n_val}")

    n_img_copied, n_missing = 0, 0
    for xml_path in xmls:
        fname = os.path.splitext(os.path.basename(xml_path))[0] + ".jpg"
        split = "val" if os.path.basename(xml_path) in val_set else "train"
        W, H, objs = parse_voc_xml(xml_path)
        # 写 YOLO 标注
        out_txt = os.path.join(OUT, "labels", split, fname.replace(".jpg", ".txt"))
        with open(out_txt, "w") as fp:
            for (name, x1, y1, x2, y2) in objs:
                cid = APDDD_CLASSES.get(name)
                if cid is None:
                    print(f"  未知类: {name} in {fname}")
                    continue
                cx = (x1 + x2) / 2 / W
                cy = (y1 + y2) / 2 / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H
                fp.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        # 复制图片
        if fname in img_map:
            shutil.copy2(img_map[fname], os.path.join(OUT, "images", split, fname))
            n_img_copied += 1
        else:
            n_missing += 1
    print(f"图片复制: {n_img_copied}, 缺失: {n_missing}")

    # 4. data.yaml
    assert len(FINAL_NAMES) == 45, f"应为45类: {len(FINAL_NAMES)}"
    yaml = f"path: {OUT}\ntrain: train/images\nval: val/images\n\nnc: 45\nnames:\n"
    for i, n in enumerate(FINAL_NAMES):
        yaml += f"  {i}: {n}\n"
    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml)

    # 5. 统计
    for split in ("train", "val"):
        n_img = len(os.listdir(os.path.join(OUT, "images", split)))
        n_lab = len(os.listdir(os.path.join(OUT, "labels", split)))
        print(f"{split}: {n_img} 图 / {n_lab} 标注")
    print("✅ 45 类数据集构建完成:", OUT)


if __name__ == "__main__":
    main()
