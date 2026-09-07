# -*- coding: utf-8 -*-
"""Generate final dual reports: 缺陷种类汇总表 + 各缺陷识别准确率报告."""
import json, pathlib, datetime

base = pathlib.Path(__file__).resolve().parent
baseline = json.loads((base / "eval_result_baseline_35c.json").read_text("utf-8"))
final = json.loads((base / "result_prod_v2_b2.json").read_text("utf-8"))

report_dir = base.parent / "reports_final"
report_dir.mkdir(exist_ok=True)

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

# ── Report 1: 缺陷种类汇总表 ──
classes = sorted(final["per_class"].keys(), key=lambda c: final["per_class"][c]["class_id"])
lines1 = []
lines1.append("# 金属表面缺陷检测模型 — 缺陷种类汇总表")
lines1.append(f"\n生成时间: {now}")
lines1.append(f"模型: prod_v2_b2 (YOLO11m, production v1.0 训练)")
lines1.append(f"评估集: um39_val_35c (4018 images, 9740 boxes)")
lines1.append("")
lines1.append("## 整体性能")
lines1.append("")
lines1.append(f"| 指标 | 值 |")
lines1.append(f"|---|---|")
lines1.append(f"| mAP50 | **{final['map50']:.4f}** |")
lines1.append(f"| mAP50-95 | **{final['map50_95']:.4f}** |")
lines1.append(f"| 相比基线提升 | mAP50 +{final['map50']-baseline['map50']:.4f}, mAP50-95 +{final['map50_95']-baseline['map50_95']:.4f} |")
lines1.append("")
lines1.append("## 缺陷类型列表 (35 类)")
lines1.append("")
lines1.append("| ID | 缺陷名称 | 训练样本数 | mAP50 | 精确率(P) | 召回率(R) | 等级 |")
lines1.append("|---|---|---|---|---|---|---|")
for c in classes:
    pc = final["per_class"][c]
    m = pc["map50"]
    if m >= 0.95: grade = "★★★★★ 优秀"
    elif m >= 0.85: grade = "★★★★ 良好"
    elif m >= 0.70: grade = "★★★ 中等"
    elif m >= 0.50: grade = "★★ 较弱"
    else: grade = "★ 待提升"
    lines1.append(f"| {pc['class_id']} | {c} | - | {m:.4f} | {pc['precision']:.4f} | {pc['recall']:.4f} | {grade} |")
lines1.append("")
lines1.append("## 性能分布")
lines1.append("")
grades = {"★★★★★": 0, "★★★★": 0, "★★★": 0, "★★": 0, "★": 0}
for c in classes:
    m = final["per_class"][c]["map50"]
    if m >= 0.95: grades["★★★★★"] += 1
    elif m >= 0.85: grades["★★★★"] += 1
    elif m >= 0.70: grades["★★★"] += 1
    elif m >= 0.50: grades["★★"] += 1
    else: grades["★"] += 1
for g, cnt in grades.items():
    lines1.append(f"- {g}: {cnt} 类")

out1 = report_dir / "缺陷种类汇总表.md"
out1.write_text("\n".join(lines1), encoding="utf-8")

# ── Report 2: 各缺陷识别准确率报告 ──
lines2 = []
lines2.append("# 金属表面缺陷检测模型 — 各缺陷识别准确率报告")
lines2.append(f"\n生成时间: {now}")
lines2.append(f"模型: prod_v2_b2 (YOLO11m, production v1.0 训练)")
lines2.append(f"评估集: um39_val_35c (4018 images, 9740 boxes)")
lines2.append("")
lines2.append("## 训练历程")
lines2.append("")
lines2.append(f"| 阶段 | 模型 | mAP50 | mAP50-95 | 说明 |")
lines2.append(f"|---|---|---|---|---|")
lines2.append(f"| 基线 | um35c_production.pt | {baseline['map50']:.4f} | {baseline['map50_95']:.4f} | 原始35类模型 |")
lines2.append(f"| 训练#1 v11m | univ39_v11m_b20 | 0.7600 | 0.5390 | UM39数据, 100ep |")
lines2.append(f"| **训练#2 prod** | **prod_v2_b2** | **{final['map50']:.4f}** | **{final['map50_95']:.4f}** | **production v1.0含负样本, 75ep** |")
lines2.append("")
lines2.append("## 逐类准确率对比 (基线 → 最终模型)")
lines2.append("")
lines2.append("| 缺陷类型 | 基线mAP50 | 最终mAP50 | 提升 | 基线P | 最终P | 基线R | 最终R |")
lines2.append("|---|---|---|---|---|---|---|---|")
up = flat = down = 0
for c in classes:
    bl = baseline["per_class"].get(c, {})
    fn = final["per_class"].get(c, {})
    bm = bl.get("map50", 0)
    fm = fn.get("map50", 0)
    bp = bl.get("precision", 0)
    fp = fn.get("precision", 0)
    br = bl.get("recall", 0)
    fr = fn.get("recall", 0)
    delta = fm - bm
    arrow = "↑↑" if delta > 0.05 else ("↑" if delta > 0.01 else ("=" if abs(delta) <= 0.01 else "↓"))
    if delta > 0.01: up += 1
    elif delta < -0.01: down += 1
    else: flat += 1
    lines2.append(f"| {c} | {bm:.4f} | {fm:.4f} | {delta:+.4f} {arrow} | {bp:.4f} | {fp:.4f} | {br:.4f} | {fr:.4f} |")
lines2.append("")
lines2.append("## 总结")
lines2.append("")
lines2.append(f"- 提升类: {up}")
lines2.append(f"- 持平类: {flat}")
lines2.append(f"- 下降类: {down}")
lines2.append("")
lines2.append("## 关键改进")
lines2.append("")
improvements = []
for c in classes:
    bl = baseline["per_class"].get(c, {})
    fn = final["per_class"].get(c, {})
    delta = fn.get("map50", 0) - bl.get("map50", 0)
    if delta > 0.1:
        improvements.append((c, delta))
improvements.sort(key=lambda x: -x[1])
for c, d in improvements:
    lines2.append(f"- **{c}**: +{d:.4f} (" + "{:.4f} → {:.4f}".format(baseline["per_class"][c]["map50"], final["per_class"][c]["map50"]) + ")")
lines2.append("")
lines2.append("## 待提升缺陷")
lines2.append("")
weak = [(c, final["per_class"][c]["map50"]) for c in classes if final["per_class"][c]["map50"] < 0.5]
weak.sort(key=lambda x: x[1])
for c, m in weak:
    lines2.append(f"- **{c}**: mAP50={m:.4f} (样本少/特征不明显)")

out2 = report_dir / "各缺陷识别准确率报告.md"
out2.write_text("\n".join(lines2), encoding="utf-8")

print(f"报告已保存:")
print(f"  {out1}")
print(f"  {out2}")
