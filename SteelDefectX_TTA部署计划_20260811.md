# SteelDefectX 部署与 TTA 增强计划

> **2026-08-13 修订**：任务一部署已完成 ✅（8-12 部署 / 8-13 修复输入尺寸 bug 并验证）；任务二 TTA 结论不变（不采用）。完整部署命令以 `SteelDefectX部署命令_20260811.md`（8-13 修订版）为准。

日期：2026-08-11 18:55（TTA 评估结论）

## 任务一：SteelDefectX 部署到 Nano（优先级：⭐⭐⭐，✅ 已完成 2026-08-13）

### 目标
将 SteelDefectX v8s（mAP50=0.802，25类）部署到 Jetson Orin Nano 替换生产模型。**已达成：Nano 现跑 `SteelDefectX_v8s_25c_fp16.engine`（25类），回归 18/18 通过。**

### 模型信息
- 权重：`runs/detect/sdx_v8s/weights/best.pt`（22.5MB）
- 类别数：25 类（见下方类别表）
- 训练分辨率：256×256
- 架构：YOLOv8s

### 25 类名称
Bright scratch / Crazing / Crease / Crescent gap / Dark scratches / Finishing roll printing / Inclusion / Iron scale compression / Iron sheet ash / Oil spot / Oxide scale of plate system / Oxide scale of temperature system / Patches / Pitted surface / Punching / Red iron sheet / Rolled in scale

### 执行步骤（历史记录，已完成）
1. **导出 ONNX**（PC）✅
   - 输入：best.pt
   - 输出：`SteelDefectX_v8s_25c_256_clean.onnx`（clean 版，44.9MB）
   - 脚本：`scripts/export_sdx_onnx.py`

2. **上传 Nano**（SCP）✅
   - 目标：`/home/nvidia/defect_detection/models/neu/SteelDefectX_v8s_25c_256_clean.onnx`（8-12 10:05）

3. **编译 TensorRT Engine**（Nano）✅
   - 输入：SteelDefectX_v8s_25c_256_clean.onnx
   - 输出：`SteelDefectX_v8s_25c_fp16.engine`（24.9MB，8-12 10:13）
   - 命令：`/usr/src/tensorrt/bin/trtexec --onnx=... --fp16 --saveEngine=...`
   - 预计耗时：5-10 分钟

4. **更新 systemd 服务** ✅
   - `defect-infer.service` ExecStart 指向 `server/infer_server.py --engine .../SteelDefectX_v8s_25c_fp16.engine`
   - 重启服务：`sudo systemctl restart defect-infer.service`
   - **⚠ 8-13 修复**：TRTInfer 输入尺寸原写死 640，与 SDX 静态 256×256 不匹配 → 推理错乱；已改为动态读取 `engine.get_tensor_shape()`，验证通过

5. **验证** ✅
   - 服务 active；NEU 6类×3 张回归 18/18 正确；GPU 6.7ms；30 次压力测试稳定

### 状态
- [x] 导出 ONNX
- [x] 上传 Nano
- [x] 编译 engine
- [x] 更新服务
- [x] 验证

---

## 任务二：TTA 推理增强（优先级：⭐⭐，评估已完成）

### 结论：TTA 无实际价值，不推荐使用

| 方法 | mAP50 | 相对基准 |
|------|-------|---------|
| 基准 256 | **0.8298** | — |
| 320 多尺度 | 0.8039 | ↓0.026 |
| 384 多尺度 | 0.5350 | ↓0.295 |
| 256+TTA 翻转+多尺度 | 0.8324 | ↑0.003 |

**分析**：
- 多尺度推理反而有害：模型训练在 256x256，320/384 尺度超出训练分布，精度下降
- TTA 翻转提升 0.26 个点，可忽略不计，且增加 3 倍推理时间
- **推荐：直接用 256 推理，不做 TTA**

### 状态
- [x] 在 val 集评估（2318张）
- [x] 结论：TTA 无价值，不采用

### 备选方案（仍可尝试）
类别置信度阈值按类调优：不同类别背景噪声不同，按类设不同 conf 阈值。
需手动分析各类 false positive/negative 分布后调参，ROI 较低。

---

## Nano 连接信息
- IP：192.168.1.101
- 用户：nvidia
- 模型目录：`/home/nvidia/defect_detection/models/neu/`
- trtexec 路径：`/usr/src/tensorrt/bin/trtexec`
- 服务：`defect-infer.service`
