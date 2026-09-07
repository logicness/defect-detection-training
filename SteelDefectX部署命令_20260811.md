# SteelDefectX Nano 部署命令

> **2026-08-13 修订**：更正 ONNX 输出布局描述（bbox 在前而非 cls 在前）、Step 3 后处理（实际无需修改，postprocess 已通用化）、Step 4 systemd 路径（server/infer_server.py 而非 trt_infer.py）、补充输入尺寸动态读取要求。**本次部署发现的 640/256 输入尺寸 bug 即源于旧文档误导。**

## ONNX 文件位置
```
D:\RK3568&Orin Nano\ORIN NANO\Model Training\SteelDefectX_v8s_25c_256_clean.onnx
大小：42.8 MB
输入格式：(batch, 3, 256, 256)   ← 静态输入，推理必须用 256×256
输出格式：(batch, 29, 1344) = 前 4 bbox(cx,cy,w,h 已 decode 到 256 尺度) + 后 25 类概率(已 sigmoid)
          ← 与 YOLOv8 标准布局一致（bbox 在前、cls 在后），不是"25类+4bbox"
```

---

## 自动化部署（推荐）

一键脚本（PC 上运行）：
```bash
cd D:\RK3568&Orin Nano\ORIN NANO\Model Training
python scripts/deploy_sdx_to_nano.py --dry-run   # 先预览
python scripts/deploy_sdx_to_nano.py              # 实际执行
```

支持参数：`--nano-host 192.168.1.101` `--nano-user nvidia` `--dry-run`

脚本自动完成：上传 ONNX → 编译 FP16 engine → 备份原配置 → 修改 trt_infer.py → 更新 systemd → 重启验证

---

## 手动部署步骤（192.168.1.101，用户 nvidia）

### Step 1：上传 ONNX
```bash
# Windows PowerShell
scp "D:\RK3568&Orin Nano\ORIN NANO\Model Training\SteelDefectX_v8s_25c_256_clean.onnx" nvidia@192.168.1.101:/home/nvidia/defect_detection/models/neu/
```

### Step 2：编译 TensorRT Engine（Nano 上执行）
```bash
cd /home/nvidia/defect_detection/models/neu/

# FP16 编译（预计 5-10 分钟）
sudo /usr/src/tensorrt/bin/trtexec \
  --onnx=SteelDefectX_v8s_25c_256_clean.onnx \
  --fp16 \
  --saveEngine=SteelDefectX_v8s_25c_256_fp16.engine \
  --workspace=4096

# 验证 engine 性能（Nano 上执行）
sudo /usr/src/tensorrt/bin/trtexec --loadEngine=SteelDefectX_v8s_25c_256_fp16.engine --batch=1
# 预期：GPU latency < 15ms, QPS > 60
```

### Step 3：校验 trt_infer.py（无需修改）
> ⚠️ 旧版本文档要求"修改后处理支持 25 类（cls 在前 + sigmoid）"——**这是错的**，实测 ONNX 布局是 bbox 在前 4 行、cls 在后 25 行（YOLOv8 标准），且概率已 sigmoid。当前 `inference/trt_infer.py` 的 postprocess 已通用化（`output.shape[0] in (84, 10, 29)` 自动转置 + `boxes=output[:,:4], scores=output[:,4:]`），**无需任何修改**。

**必须确认的两点**（否则推理静默错乱）：
1. **输入尺寸动态读取**：`TRTInfer.__init__` 必须从引擎读取真实输入 shape（`self.engine.get_tensor_shape(self.input_name)`），SDX 是 `(1,3,256,256)`。若写死 `input_shape=(1,3,640,640)` 会触发 `setInputShape: Static dimension mismatch`，服务不崩但**结果错乱**（2026-08-13 已实测并修复，备份 `.bak_fix_20260813_174944`）。
2. **postprocess 29 通道分支**：`if output.ndim == 2 and output.shape[0] in (84, 10, 29)` 需含 29。

校验命令（Nano 上执行）：
```bash
grep -c 'get_tensor_shape(self.input_name)' /home/nvidia/defect_detection/inference/trt_infer.py  # 期望 ≥1
grep -c '84, 10, 29' /home/nvidia/defect_detection/inference/trt_infer.py                          # 期望 ≥1
```

### Step 4：更新 systemd 服务
```bash
# Nano 上编辑服务文件（注意：入口是 server/infer_server.py，不是 trt_infer.py）
sudo nano /etc/systemd/system/defect-infer.service

# 修改 ExecStart 中的 --engine 参数：
ExecStart=/usr/bin/python3 /home/nvidia/defect_detection/server/infer_server.py \
  --engine /home/nvidia/defect_detection/models/neu/SteelDefectX_v8s_25c_fp16.engine

# 重载并重启
sudo systemctl daemon-reload
sudo systemctl restart defect-infer.service
sudo systemctl status defect-infer.service
```

### Step 5：验证
```bash
# 测试推理（Nano 上，需 cv2 编码为 JPEG 后发送）
python3 -c "
import socket, struct, json, base64, cv2, numpy as np
img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
x = base64.b64encode(cv2.imencode('.jpg', img)[1].tobytes()).decode()
sock = socket.socket(); sock.settimeout(10)
sock.connect(('127.0.0.1', 8888))
req = {'type': 'detect_request', 'image_base64': x}
data = json.dumps(req).encode()
sock.sendall(struct.pack('>I', len(data)) + data)
lb = sock.recv(4)
L = struct.unpack('>I', lb)[0]
resp = b''
while len(resp) < L: resp += sock.recv(8192)
r = json.loads(resp)
print('ok=%s dets=%d' % (r.get('ok'), len(r.get('detections', []))))
for d in r.get('detections', [])[:5]:
    print('  class_id=%s conf=%.3f' % (d.get('class_id'), d.get('confidence', 0)))
sock.close()
"
```

---

## 上位机兼容性（2026-08-13 已确认）
- ✅ 上位机 `core/class_names.py` 已内置 25 类映射（`STEELDEFECTX_CLASSES`），`resolve_class_names()` 匹配 `SteelDefectX_v8s_25c_fp16.engine` 自动返回 25 类表，越界兜底 `clsN`
- ✅ 检测结果直接用 `class_id` 上报 + 上位机映射，无需改协议
- ⚠️ 上位机「推理源」须选 Nano（模型管理→🖥 本地检测 为本地模型路径，不走 Nano）

---

## 回滚方案
如需回退到 NEU 模型（e1003，6 类，640×640）：
```bash
sudo sed -i 's|--engine [^ ]*|--engine /home/nvidia/defect_detection/models/neu/neu_yolov8s_e1003_fp16.engine|' /etc/systemd/system/defect-infer.service
sudo systemctl daemon-reload
sudo systemctl restart defect-infer.service
```
> 注：e1003 是 640×640 引擎，回滚后 trt_infer.py 动态读取输入 shape 会自动适配，无需其他改动。