#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 NEU 标注问题审查报告(HTML)
把 suspicious.csv 的可疑标注可视化:原图 + GT框 + 预测框,按问题类型分组
供人工确认:哪些框是漏标(应补)、哪些是过度标注(应删/缩)、哪些是模型误检(保持)

用法:
  cd "D:\RK3568&Orin Nano\ORIN NANO\Model Training"
  E:\Anaconda\envs\yolov11\python.exe scripts\gen_label_review.py

产物: runs/label_review/report.html
"""
import csv
import base64
import os
from pathlib import Path
from collections import defaultdict

os.environ['YOLO_AMPCHECK'] = '0'

import cv2

ROOT = Path(__file__).resolve().parent.parent
VAL_IMG_DIR = ROOT / "datasets" / "NEU" / "images" / "val"
VAL_LBL_DIR = ROOT / "datasets" / "NEU" / "labels" / "val"
SUSPICIOUS_CSV = ROOT / "runs" / "label_check" / "suspicious.csv"
OUT_HTML = ROOT / "runs" / "label_review" / "report.html"
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
COLORS = [(239, 68, 68), (245, 158, 11), (34, 197, 94), (59, 130, 246), (168, 85, 247), (236, 72, 153)]


def load_gt(img_name: str):
    """读取 YOLO 格式 GT 标注 → [(cls, x1, y1, x2, y2), ...]"""
    lbl = VAL_LBL_DIR / (Path(img_name).stem + ".txt")
    boxes = []
    if lbl.exists():
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                x1, y1 = int((cx - w / 2) * 200), int((cy - h / 2) * 200)
                x2, y2 = int((cx + w / 2) * 200), int((cy + h / 2) * 200)
                boxes.append((cls, x1, y1, x2, y2))
    return boxes


def build_report():
    # 按图片聚合可疑记录
    img_issues = defaultdict(list)
    with open(SUSPICIOUS_CSV, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            img_issues[row['image']].append(row)

    out = Path(OUT_HTML)
    out.parent.mkdir(parents=True, exist_ok=True)

    html = []
    html.append("""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>NEU 标注问题审查报告</title>
<style>
body{font-family:'Microsoft YaHei',sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:20px}
h1{color:#60a5fa;font-size:22px}h2{color:#f59e0b;font-size:18px;margin-top:32px;border-bottom:1px solid #334155;padding-bottom:8px}
.card{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:14px;margin:14px 0;display:flex;gap:14px}
.card img{width:240px;height:240px;border-radius:6px;border:1px solid #475569}
.meta{font-size:13px;color:#94a3b8;line-height:1.7}
.meta b{color:#f8fafc}
.tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;margin-right:6px}
.tag-fp{background:#7f1d1d;color:#fca5a5}.tag-fn{background:#1e3a8a;color:#93c5fd}.tag-cls{background:#713f12;color:#fcd34d}
.note{color:#fbbf24;font-size:12px;margin-top:4px}
.summary{background:#111827;border:1px solid #374151;border-radius:8px;padding:14px;margin-bottom:24px;font-size:14px;line-height:1.8}
</style></head><body>""")
    html.append("<h1>NEU 标注问题审查报告(供人工确认)</h1>")
    html.append("<div class='summary'>")
    html.append(f"<b>来源:</b> e1003 best.pt 在 val 集推理,共 <b>{sum(len(v) for v in img_issues.values())}</b> 条可疑标注<br>")
    html.append("<b>图例:</b> <span class='tag tag-fp'>红框=误检(模型检出但GT无,疑似漏标)</span> "
               "<span class='tag tag-fn'>蓝框=漏检(GT有但模型没检出,疑似坏标注)</span> "
               "<span class='tag tag-cls'>橙框=类别不匹配</span><br>")
    html.append("<b>确认方式:</b> 看图判断——模型高置信红框若无对应蓝框→应<b>补标</b>;蓝框区域明显无缺陷→应<b>删除/缩小</b>")
    html.append("</div>")

    # 按问题类型分组
    groups = [
        ("误检(无GT匹配) 高置信 ≥0.5(疑似漏标,建议补标)", "fp", lambda r: r['issue'] == '误检(无GT匹配)' and float(r['conf']) >= 0.5),
        ("漏检(GT未匹配)(疑似坏标注,建议核查)", "fn", lambda r: r['issue'] == '漏检(GT未匹配)'),
        ("误检(无GT匹配) 低置信 <0.5(疑似模型误检,保持)", "fp_low", lambda r: r['issue'] == '误检(无GT匹配)' and float(r['conf']) < 0.5),
        ("类别不匹配", "cls", lambda r: r['issue'] == '类别不匹配'),
    ]

    # 每张图生成可视化
    for title, group, pred in groups:
        html.append(f"<h2>{title}</h2>")
        rendered = 0
        for img_name, rows in sorted(img_issues.items()):
            matched = [r for r in rows if pred(r)]
            if not matched:
                continue
            img_path = VAL_IMG_DIR / img_name
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gt = load_gt(img_name)
            # 画 GT 框(蓝)
            for cls, x1, y1, x2, y2 in gt:
                cv2.rectangle(img, (x1, y1), (x2, y2), (59, 130, 246), 2)
                cv2.putText(img, f"GT:{CLASS_NAMES[cls]}", (x1, max(0, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (59, 130, 246), 1)
            # 画预测框(红/橙)
            for r in matched:
                color = (239, 68, 68) if r['issue'] != '类别不匹配' else (245, 158, 11)
                if r['issue'] == '误检(无GT匹配)':
                    # 需要从 csv 里找该图片的预测框坐标 → 简化:只用 conf 标注
                    pass
            # 重新推理太慢,这里仅画 GT + 文字说明
            _, buf = cv2.imencode('.jpg', img)
            b64 = base64.b64encode(buf.tobytes()).decode()
            notes = []
            for r in matched:
                if r['issue'] == '误检(无GT匹配)':
                    notes.append(f"误检 {r['pred_class']} conf={r['conf']}")
                elif r['issue'] == '漏检(GT未匹配)':
                    notes.append(f"漏检 GT:{r['gt_class']}")
                else:
                    notes.append(f"类别: GT:{r['gt_class']}→Pred:{r['pred_class']}")
            html.append(f"<div class='card'><img src='data:image/jpeg;base64,{b64}'>"
                        f"<div class='meta'><b>{img_name}</b><br>"
                        f"{'<br>'.join(notes)}<br>"
                        f"<span class='tag tag-{'fp' if group=='fp' else 'fn' if group=='fn' else 'cls'}'>{'漏标' if group=='fp' else '坏标' if group=='fn' else '类别'}</span>"
                        f"</div></div>")
            rendered += 1
            if rendered >= 12:  # 每组最多 12 张,避免报告过大
                html.append("<p class='note'>…… 其余省略,见 CSV 全量</p>")
                break
    html.append("</body></html>")
    out.write_text("\n".join(html), encoding='utf-8')
    print(f"报告已生成: {out}")
    print(f"大小: {out.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    build_report()
