# -*- coding: utf-8 -*-
"""构建 production v2.1 数据集（2026-09-07）
= v1.0 完整副本（images 硬链接，labels 复制，不带 cache）
+ Waist folding(id=21) 碎片框合并 margin=0.06（复用 merge_fragmented_labels.py）
+ 弱类重采样 x3：Crease(id=2) / Rolled pit(id=17) / warp(id=29)（仅 train，硬链接副本）
原 v1.0 不动。
"""
import os
import sys
import glob
import shutil
import subprocess
from pathlib import Path

ROOT = Path(r"D:\RK3588&Orin Nano\ORIN NANO\Model Training")
SRC = ROOT / "datasets_master" / "production" / "v1.0"
DST = ROOT / "datasets_master" / "production" / "v2.1"
MERGE_PY = ROOT / "scripts" / "diag" / "merge_fragmented_labels.py"
WEAK = {2: "Crease", 17: "Rolled pit", 29: "warp"}
DUP = 2  # 额外副本数 → 共 3x

def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)

def main():
    assert SRC.is_dir(), "v1.0 不存在"
    if not DST.exists():
        DST.mkdir(parents=True)

    # 1) images 硬链接
    for split in ("train", "val"):
        sdir, ddir = SRC / "images" / split, DST / "images" / split
        ddir.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sdir.iterdir():
            t = ddir / f.name
            if not t.exists():
                link_or_copy(f, t); n += 1
        print(f"[images/{split}] 链接 {n} 个新文件，总计 {len(list(ddir.iterdir()))}")

    # 2) labels 复制（不带 cache）
    for split in ("train", "val"):
        sdir, ddir = SRC / "labels" / split, DST / "labels" / split
        ddir.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sdir.glob("*.txt"):
            t = ddir / f.name
            if not t.exists():
                shutil.copy2(f, t); n += 1
        print(f"[labels/{split}] 复制 {n} 个新文件，总计 {len(list(ddir.glob('*.txt')))}")

    # 3) data.yaml（改 path 指向 v2.1）
    yaml_txt = (SRC / "data.yaml").read_text(encoding="utf-8")
    yaml_txt = yaml_txt.replace(str(SRC).replace("\\", "/"), str(DST).replace("\\", "/"))
    (DST / "data.yaml").write_text(yaml_txt, encoding="utf-8")
    print("[data.yaml] 已写（path→v2.1）")
    for ref in ("MANIFEST.csv", "BALANCE_REPORT.md"):
        s = SRC / ref
        if s.exists() and not (DST / ref).exists():
            shutil.copy2(s, DST / ref)

    # 4) Waist folding 合并 margin=0.06（复用已验证脚本）
    tmp_out = ROOT / "scripts" / "diag" / "merged_v21_apply"
    if tmp_out.exists():
        shutil.rmtree(tmp_out)
    cmd = [sys.executable, str(MERGE_PY),
           "--labels-train", str(DST / "labels" / "train"),
           "--labels-val", str(DST / "labels" / "val"),
           "--class-id", "21", "--margin", "0.06",
           "--out-root", str(tmp_out)]
    print("[merge] 运行:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr); sys.exit(1)
    applied = 0
    for split in ("train", "val"):
        for f in (tmp_out / split).glob("*.txt"):
            shutil.copy2(f, DST / "labels" / split / f.name)
            applied += 1
    print(f"[merge] 已覆盖 {applied} 个 label 文件到 v2.1")

    # 5) 弱类重采样 x3（仅 train；val 不动）
    for cid, cname in WEAK.items():
        made = 0
        for lf in (DST / "labels" / "train").glob("*.txt"):
            has = any(ln.split() and int(float(ln.split()[0])) == cid
                      for ln in lf.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
            if not has:
                continue
            stem = lf.stem
            img = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                p = DST / "images" / "train" / (stem + ext)
                if p.exists():
                    img = p; break
            if img is None:
                continue
            for k in range(1, DUP + 1):
                li = DST / "labels" / "train" / f"{stem}__w{cid}_{k}.txt"
                ii = DST / "images" / "train" / f"{stem}__w{cid}_{k}{img.suffix}"
                if not li.exists():
                    shutil.copy2(lf, li)
                if not ii.exists():
                    link_or_copy(img, ii)
                made += 1
        print(f"[resample] {cname}(id={cid}) 追加 {made} 副本（共 {DUP}x/图）")

    # 6) 统计复核
    for split in ("train", "val"):
        n_img = len(list((DST / "images" / split).iterdir()))
        n_lbl = len(list((DST / "labels" / split).glob("*.txt")))
        print(f"[v2.1 {split}] images={n_img} labels={n_lbl}")
    print("BUILD_V21_DONE_MARKER")

if __name__ == "__main__":
    main()
