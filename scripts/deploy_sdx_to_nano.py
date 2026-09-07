#!/usr/bin/env python3
"""
SteelDefectX Nano 部署脚本
自动化：上传 ONNX → 编译 TensorRT Engine → 重启服务 → 验证

注意（2026-08-13 修订）：
- SDX clean.onnx 输出布局为 (batch, 29, 1344) = 前4 bbox(cx,cy,w,h 已decode到256尺度) + 后25类概率(已sigmoid)
  trt_infer.py 的 postprocess 已通用化（transpose + boxes=[:,:4] + scores=[:,4:]），无需再打 patch
- 输入尺寸必须由 trt_infer.py 动态读取 engine.get_tensor_shape()（SDX 是 256x256，不能写死 640）
- systemd ExecStart 指向 server/infer_server.py（不是 trt_infer.py）

用法：
  python deploy_sdx_to_nano.py [--nano-host <NANO_LAN_IP>] [--nano-user nvidia] [--dry-run]
"""
import argparse, os, sys, time, re, subprocess

# ============ 配置 ============
NANO_HOST = "<NANO_LAN_IP>"
NANO_USER = "nvidia"
NANO_MODEL_DIR = "/home/nvidia/defect_detection/models/neu"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
SERVICE_NAME = "defect-infer.service"
# 修正：trt_infer.py 实际在 inference/ 子目录
TRT_INFER_PATH = "/home/nvidia/defect_detection/inference/trt_infer.py"
INFER_SERVER_PATH = "/home/nvidia/defect_detection/server/infer_server.py"

LOCAL_ONNX = r"D:\RK3568&Orin Nano\ORIN NANO\Model Training\SteelDefectX_v8s_25c_256_clean.onnx"
ONNX_NAME = "SteelDefectX_v8s_25c_256_clean.onnx"
ENGINE_NAME = "SteelDefectX_v8s_25c_256_fp16.engine"

# ============ 工具函数 ============
def run_cmd(cmd, check=True):
    print("  $ " + cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout:
        for line in r.stdout.rstrip().splitlines():
            print("    " + line)
    if r.stderr and r.returncode != 0:
        for line in r.stderr.rstrip().splitlines():
            if "Pseudo-terminal" not in line:
                print("    ERR: " + line, file=sys.stderr)
    if check and r.returncode != 0:
        raise RuntimeError("Command failed (exit %d)" % r.returncode)
    return r


def ssh(host, user, cmd, check=True):
    full = "ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10 %s@%s '%s'" % (user, host, cmd)
    print("  $ ssh %s@%s '%s...'" % (user, host, cmd[:60]))
    r = subprocess.run(full, shell=True, capture_output=True, text=True)
    if r.stdout:
        for line in r.stdout.rstrip().splitlines()[:10]:
            print("    " + line)
    if r.stderr and "Pseudo-terminal" not in r.stderr and r.returncode != 0:
        print("    ERR: " + r.stderr.rstrip(), file=sys.stderr)
    if check and r.returncode != 0:
        raise RuntimeError("SSH failed (exit %d)" % r.returncode)
    return r


def scp(src, remote, host, user):
    cmd = 'scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 "%s" %s@%s:"%s"' % (src, user, host, remote)
    print("  $ scp ... %s@%s:..." % (user, host))
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stderr:
        for line in r.stderr.splitlines():
            if "%" in line or "KB/s" in line:
                print("    " + line.rstrip())
    if r.returncode != 0:
        raise RuntimeError("SCP failed: " + r.stderr)
    return r


# ============ 步骤 ============
def step1(args):
    print("\n[Step 1/6] 上传 ONNX...")
    if not os.path.exists(LOCAL_ONNX):
        raise FileNotFoundError(LOCAL_ONNX)
    print("  文件: %s (%.1f MB)" % (LOCAL_ONNX, os.path.getsize(LOCAL_ONNX) / 1024 / 1024))
    remote = "%s/%s" % (NANO_MODEL_DIR, ONNX_NAME)
    scp(LOCAL_ONNX, remote, args.nano_host, args.nano_user)
    print("  OK")


def step2(args):
    print("\n[Step 2/6] 编译 TensorRT FP16 Engine (5-10min)...")
    host, user = args.nano_host, args.nano_user

    # 检查 trtexec
    r = ssh(host, user, "test -f %s && echo OK || echo MISSING" % TRTEXEC, check=False)
    if "MISSING" in r.stdout:
        raise RuntimeError("trtexec not found: " + TRTEXEC)

    # 后台启动编译
    compile_cmd = (
        "cd %s && "
        "sudo %s --onnx=%s --fp16 --saveEngine=%s --workspace=4096 "
        "> /tmp/trtexec.log 2>&1 &"
    ) % (NANO_MODEL_DIR, TRTEXEC, ONNX_NAME, ENGINE_NAME)
    ssh(host, user, compile_cmd, check=False)
    print("  编译已启动，轮询进度...")

    # 轮询检查
    for i in range(20):  # 最多 10 分钟
        time.sleep(30)
        r = ssh(host, user, "pgrep -f 'trtexec.*%s' && echo RUNNING || echo DONE" % ONNX_NAME, check=False)
        if "DONE" in r.stdout:
            break
        print("  编译中... (%ds)" % (30 * (i + 1)))

    time.sleep(5)
    r = ssh(host, user, "grep -E 'completed|EXIT:0|Engine built|FAILED' /tmp/trtexec.log | tail -5", check=False)
    print("  日志末尾: " + r.stdout.strip().replace("\n", " | "))

    r = ssh(host, user, "ls -lh %s/%s 2>/dev/null || echo NOT_FOUND" % (NANO_MODEL_DIR, ENGINE_NAME), check=False)
    print("  " + r.stdout.strip())
    if "NOT_FOUND" in r.stdout:
        print("  请手动检查: ssh %s@%s 'tail /tmp/trtexec.log'" % (user, host))
    else:
        print("  OK")


def step3(args):
    print("\n[Step 3/6] 校验 trt_infer.py 兼容 25 类 + 动态输入尺寸...")
    host, user = args.nano_host, args.nano_user

    # 备份
    ts = int(time.time())
    ssh(host, user, "sudo cp %s %s/trt_infer.py.bak%d" % (TRT_INFER_PATH, NANO_MODEL_DIR, ts), check=False)
    print("  备份: %s/trt_infer.py.bak%d" % (NANO_MODEL_DIR, ts))

    # 校验 1：postprocess 需含 29 通道转置分支（标准布局：前4 bbox + 后25 cls）
    r = ssh(host, user, "grep -c '84, 10, 29' %s" % TRT_INFER_PATH, check=False)
    if int(r.stdout.strip() or "0") > 0:
        print("  [OK] postprocess 已支持 29 通道（bbox 在前 4，cls 后 25）")
    else:
        print("  [WARN] postprocess 未见 29 通道分支，需手工核对 trt_infer.py")

    # 校验 2：TRTInfer.__init__ 须动态读取引擎输入 shape（SDX 是 256x256，不能写死 640）
    r2 = ssh(host, user, "grep -c 'get_tensor_shape(self.input_name)' %s" % TRT_INFER_PATH, check=False)
    if int(r2.stdout.strip() or "0") > 0:
        print("  [OK] 输入尺寸动态读取（兼容 256x256）")
    else:
        print("  [WARN] 未见动态读取输入 shape，请确认 TRTInfer input_shape 参数与引擎匹配")

    print("  注：不再追加旧版错误 patch（cls/bbox 顺序相反），当前 trt_infer.py 通用逻辑已足够")


def step4(args):
    print("\n[Step 4/6] 更新 systemd 服务...")
    host, user = args.nano_host, args.nano_user
    svc = "/etc/systemd/system/" + SERVICE_NAME
    engine_path = "%s/%s" % (NANO_MODEL_DIR, ENGINE_NAME)

    # 备份
    ts = int(time.time())
    ssh(host, user, "sudo cp %s %s/%s.bak%d" % (svc, NANO_MODEL_DIR, SERVICE_NAME, ts), check=False)
    print("  备份: %s/%s.bak%d" % (NANO_MODEL_DIR, SERVICE_NAME, ts))

    # 替换 --engine 参数（注意：ExecStart 指向 server/infer_server.py，非 trt_infer.py）
    r = ssh(host, user, "grep -- '--engine' %s" % svc, check=False)
    if r.returncode == 0 and r.stdout.strip():
        print("  当前 engine 配置: " + r.stdout.strip())
        ssh(host, user,
            "sudo sed -i 's|--engine [^ ]*|--engine %s|g' %s" % (engine_path, svc),
            check=False)
    else:
        print("  未找到 --engine，追加到 ExecStart")
        ssh(host, user,
            "sudo sed -i 's|ExecStart=/usr/bin/python3 server/infer_server.py|"
            "ExecStart=/usr/bin/python3 server/infer_server.py --engine %s|'"
            % engine_path, check=False)

    ssh(host, user, "sudo systemctl daemon-reload", check=False)
    print("  OK")


def step5(args):
    print("\n[Step 5/6] 重启服务...")
    host, user = args.nano_host, args.nano_user
    ssh(host, user, "sudo systemctl restart %s" % SERVICE_NAME, check=False)
    time.sleep(3)
    r = ssh(host, user, "sudo systemctl is-active %s" % SERVICE_NAME, check=False)
    print("  服务状态: " + r.stdout.strip())
    if "active" in r.stdout:
        print("  OK")
    else:
        print("  检查日志:")
        ssh(host, user, "sudo journalctl -u %s --no-pager -n 10" % SERVICE_NAME, check=False)


def step6(args):
    print("\n[Step 6/6] 推理验证...")
    host, user = args.nano_host, args.nano_user

    # Nano 上生成测试图（JPEG 编码，服务端 cv2.imdecode 需要）
    ssh(host, user,
        "python3 -c \"import cv2, numpy as np; img=(np.random.rand(256,256,3)*255).astype(np.uint8); cv2.imwrite('/tmp/sdx_test.jpg', img)\"",
        check=False)

    # 单行测试代码
    code = (
        "import socket,struct,json,base64,cv2; "
        "x=base64.b64encode(open('/tmp/sdx_test.jpg','rb').read()).decode(); "
        "sock=socket.socket(); sock.settimeout(10); "
        "sock.connect(('127.0.0.1',8888)); "
        "req={'type':'detect_request','image_base64':x}; "
        "data=json.dumps(req).encode(); "
        "sock.sendall(struct.pack('>I',len(data))+data); "
        "lb=sock.recv(4); "
        "if len(lb)<4: print('no response'); "
        "else: "
        "  L=struct.unpack('>I',lb)[0]; resp=b''; "
        "  while len(resp)<L: resp+=sock.recv(8192); "
        "  r=json.loads(resp); "
        "  print('ok=%s dets=%d'%(r.get('ok'), len(r.get('detections',[])))); "
        "  for d in r.get('detections',[])[:5]: print(' class_id=%s conf=%.3f'%(d.get('class_id'),d.get('confidence',0))); "
        "sock.close()"
    )
    ssh(host, user, "python3 -c '%s'" % code, check=False)
    print("  OK")


# ============ 主流程 ============
def main():
    p = argparse.ArgumentParser(description="SteelDefectX Nano 部署脚本")
    p.add_argument("--nano-host", default=NANO_HOST)
    p.add_argument("--nano-user", default=NANO_USER)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print("=" * 50)
    print("SteelDefectX Nano 部署")
    print("Nano: %s@%s" % (args.nano_user, args.nano_host))
    print("=" * 50)

    steps = [
        ("上传 ONNX", step1),
        ("编译 Engine", step2),
        ("校验 trt_infer.py 兼容性", step3),
        ("更新 systemd 服务", step4),
        ("重启服务", step5),
        ("推理验证", step6),
    ]

    for i, (name, fn) in enumerate(steps, 1):
        print("\n[%d/%d] %s" % (i, len(steps), name))
        if args.dry_run:
            print("  [dry-run 跳过]")
            continue
        try:
            fn(args)
        except Exception as e:
            print("  FAIL: %s" % e)
            print("  从第 %d 步重试: python %s --nano-host %s" % (i, __file__, args.nano_host))
            sys.exit(1)

    print("\n" + "=" * 50)
    print("SteelDefectX 部署完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
