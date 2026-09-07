#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 train 集标注修正可视化审查报告(report_train.html)
对照 fix_list_train.csv:绿框=补标(高置信漏标),黄框=核查(GT坏标)
"""
import csv
import re
import base64
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "datasets" / "NEU" / "images" / "train"
FIX_CSV = ROOT / "runs" / "label_review" / "fix_list_train.csv"
OUT = ROOT / "runs" / "label_review" / "report_train.html"
NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


def main():
    items = {}
    with open(FIX_CSV, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['action'] in ('补标', '核查'):
                items.setdefault(r['image'], []).append(r)

    html = ['<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">'
            '<title>NEU train 标注修正审查</title><style>',
            'body{font-family:Microsoft YaHei;background:#0f172a;color:#e2e8f0;padding:20px}',
            'h1{color:#60a5fa;font-size:20px}.card{background:#1e293b;border:1px solid #334155;'
            'border-radius:8px;padding:12px;margin:12px 0;display:flex;gap:12px}',
            '.card img{width:240px;height:240px;border-radius:6px}.meta{font-size:12px;'
            'color:#94a3b8;line-height:1.6}.a-add{color:#4ade80;font-weight:bold}'
            '.a-check{color:#fbbf24;font-weight:bold}',
            '.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;margin:2px}',
            '.tag-add{background:#14532d;color:#86efac}.tag-check{background:#713f12;color:#fde68a}',
            '</style></head><body>']
    html.append('<h1>NEU train 集标注修正审查(人工确认)</h1>')
    total = sum(len(v) for v in items.values())
    html.append(f'<div style="background:#111827;padding:12px;border-radius:8px;font-size:13px;line-height:1.8">'
                f'共 <b>{total}</b> 条待确认 | '
                f'<span class="tag tag-add">绿=补标(模型高置信检出,GT缺失)</span> '
                f'<span class="tag tag-check">黄=核查(GT框模型检不出,疑似坏标)</span><br>'
                f'确认方法:看图后对每个框决定 补/删/缩/留,把结论告诉我即可</div>')

    cnt = 0
    for img_name in sorted(items):
        p = IMG_DIR / img_name
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        for it in items[img_name]:
            cls = it['cls']
            if it['action'] == '补标' and it['pred_box'] != '-':
                nums = re.findall(r'-?\d+', it['pred_box'])
                if len(nums) == 4:
                    x1, y1, x2, y2 = map(int, nums)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (34, 197, 94), 2)
                    cv2.putText(img, f"ADD:{cls} {it['conf']}", (x1, max(0, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (34, 197, 94), 1)
            elif it['action'] == '核查' and it['gt_box'] != '-':
                nums = re.findall(r'-?\d+', it['gt_box'])
                if len(nums) == 4:
                    x1, y1, x2, y2 = map(int, nums)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (245, 158, 11), 2)
                    cv2.putText(img, f"CHECK:{cls}", (x1, max(0, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (245, 158, 11), 1)
        _, buf = cv2.imencode('.jpg', img)
        b64 = base64.b64encode(buf.tobytes()).decode()
        notes = '<br>'.join(
            f"<span class='a-{'add' if i['action'] == '补标' else 'check'}'>"
            f"{'补标' if i['action'] == '补标' else '核查'}</span> {i['cls']} "
            f"conf={i['conf']} GT:{i['gt_box']} Pred:{i['pred_box']}"
            for i in items[img_name])
        html.append(f"<div class='card'><img src='data:image/jpeg;base64,{b64}'>"
                    f"<div class='meta'><b>{img_name}</b><br>{notes}</div></div>")
        cnt += 1

    html.append('</body></html>')
    OUT.write_text(''.join(html), encoding='utf-8')
    print(f"报告已生成: {OUT} | {cnt} 张图 | {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == '__main__':
    main()
