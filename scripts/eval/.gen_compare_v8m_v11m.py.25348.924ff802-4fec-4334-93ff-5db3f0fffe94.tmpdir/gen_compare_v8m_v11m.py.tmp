# -*- coding: utf-8 -*-
"""Generate v8m vs v11m comparison report."""
import json, pathlib, datetime

base = pathlib.Path(__file__).resolve().parent
v8m = json.loads((base / "eval_result_univ39_v8m.json").read_text("utf-8"))
v11m = json.loads((base / "result_univ39_v11m.json").read_text("utf-8"))

lines = []
lines.append("# v8m vs v11m 正式对比报告")
lines.append(f"\n生成时间: {datetime.datetime.now():%Y-%m-%d %H:%M}")
lines.append(f"评估集: um39_val_35c (4018 imgs, 9740 boxes)")
lines.append("")
lines.append("## 整体指标")
lines.append("")
lines.append(f"| 指标 | v8m | v11m | Δ |")
lines.append(f"|---|---|---|---|")
lines.append(f"| mAP50 | {v8m['map50']:.4f} | {v11m['map50']:.4f} | {v11m['map50']-v8m['map50']:+.4f} |")
lines.append(f"| mAP50-95 | {v8m['map50_95']:.4f} | {v11m['map50_95']:.4f} | {v11m['map50_95']-v8m['map50_95']:+.4f} |")
lines.append("")

# per-class
classes = sorted(set(list(v8m["per_class"].keys()) + list(v11m["per_class"].keys())))
up = flat = down = 0
lines.append("## 逐类对比")
lines.append("")
lines.append("| 缺陷类型 | v8m mAP50 | v11m mAP50 | Δ | 趋势 |")
lines.append("|---|---|---|---|---|")
for c in classes:
    a = v8m["per_class"].get(c, {}).get("map50", 0)
    b = v11m["per_class"].get(c, {}).get("map50", 0)
    d = b - a
    if d > 0.02:
        trend = "↑↑"; up += 1
    elif d > 0.005:
        trend = "↑"; up += 1
    elif d < -0.02:
        trend = "↓↓"; down += 1
    elif d < -0.005:
        trend = "↓"; down += 1
    else:
        trend = "="; flat += 1
    lines.append(f"| {c} | {a:.4f} | {b:.4f} | {d:+.4f} | {trend} |")

lines.append("")
lines.append(f"## 总结")
lines.append(f"")
lines.append(f"- 提升类: {up}")
lines.append(f"- 持平类: {flat}")
lines.append(f"- 下降类: {down}")
lines.append(f"")
lines.append(f"## 选型结论")
lines.append(f"")
if v11m["map50"] > v8m["map50"]:
    lines.append(f"**推荐: v11m** — mAP50 高 {v11m['map50']-v8m['map50']:.4f}, mAP50-95 高 {v11m['map50_95']-v8m['map50_95']:.4f}")
else:
    lines.append(f"**推荐: v8m** — 性能持平或更优")

out = base / "v8m_vs_v11m_对比报告.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"报告已保存: {out}")
