# -*- coding: utf-8 -*-
"""Auto-run blocks 2..15 for production v2 training."""
import subprocess, sys, os, datetime

PY = r"E:\Anaconda\envs\yolov11\python.exe"
SCRIPT = r"D:\RK3588&Orin Nano\ORIN NANO\Model Training\scripts\train\train_production_v2.py"
LOG_DIR = r"D:\RK3588&Orin Nano\ORIN NANO\Model Training\logs"

for b in range(2, 16):
    log_out = os.path.join(LOG_DIR, f"train_prod_v2_b{b}.log")
    print(f"[{datetime.datetime.now():%H:%M:%S}] Starting block {b}/15 ...", flush=True)
    with open(log_out, "w") as fout:
        r = subprocess.run([PY, SCRIPT, "--block", str(b)], stdout=fout, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"Block {b} FAILED (exit {r.returncode})", flush=True)
        sys.exit(1)
    print(f"[{datetime.datetime.now():%H:%M:%S}] Block {b} done.", flush=True)

done_marker = os.path.join(LOG_DIR, "train_prod_v2_ALL_DONE.log")
with open(done_marker, "w") as f:
    f.write(f"ALL_DONE {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
print("ALL DONE!", flush=True)
