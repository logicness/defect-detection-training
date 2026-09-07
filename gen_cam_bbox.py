# -*- coding: utf-8 -*-
"""GradCAM 弱监督检测: 用分类模型 last.pt 生成热力图 → 阈值化+连通域 → 伪 bbox.

用法:
  python -S gen_cam_bbox.py --sample N        # 对前 N 张图生成可视化(验证质量)
  python -S gen_cam_bbox.py --full            # 全量 train 集生成伪框数据集
  python -S gen_cam_bbox.py --predict          # (预留) 对任意图预测+可视化
"""
import argparse, time, sys, csv
from pathlib import Path
from multiprocessing import freeze_support

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

CKPT = r'<LOCAL_MODEL_PATH>/runs/cls/guangdong_baseline/weights/last.pt'
DATA = Path(r'<LOCAL_ROOT>/ORIN NANO/dataset/guangdong_cls')
OUT = Path(r'<LOCAL_ROOT>/ORIN NANO/dataset/guangdong_detect')
TARGET_LAYER = 8      # 最后一个 C2f (512ch)
IMGSZ = 640           # 分类模型输入尺寸
CAM_THRESH = 0.35     # 热力图阈值(经 min-max 归一化后)
MIN_AREA_RATIO = 0.0006   # 最小连通域面积占比
NORMAL_CLASSES = {'正常'}


class GradCAM:
    """对 YOLOv8-cls 生成 GradCAM 热力图."""

    def __init__(self, ckpt, target_idx=TARGET_LAYER):
        from ultralytics import YOLO
        self.net = YOLO(ckpt).model
        self.net.eval()
        # 确保参数可求梯度 (加载的推理模型 requires_grad=False)
        for p in self.net.parameters():
            p.requires_grad = True
        self.net.to('cuda')
        self.act = {}
        self.grad = {}
        tgt = self.net.model[target_idx]
        tgt.register_forward_hook(self._fwd)
        tgt.register_full_backward_hook(self._bwd)
        self.names = YOLO(ckpt).names if hasattr(YOLO(ckpt), 'names') else None

    def _fwd(self, m, i, o):
        self.act['t'] = o.detach()

    def _bwd(self, m, gi, go):
        self.grad['t'] = go[0].detach()

    @torch.no_grad()
    def _preprocess(self, img):
        """PIL 图 -> 1,3,640,640 [0,1]"""
        x = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        x = cv2.resize(x, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
        x = x.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]
        return torch.from_numpy(x).to('cuda')

    def forward_logits(self, x):
        return self.net(x)

    def cam(self, img, target_cls=None):
        """返回 (cam_640, target_cls, logits)"""
        x = self._preprocess(img)
        out = self.net(x)
        logits = out[0].detach()  # (1,28)
        if target_cls is None:
            target_cls = int(logits.argmax(1).item())
        self.net.zero_grad()
        onehot = torch.zeros_like(logits)
        onehot[0, target_cls] = 1.0
        out[0].backward(gradient=onehot)
        A = self.act['t']    # 1,512,h,w
        G = self.grad['t']   # 1,512,h,w
        alpha = G.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((A * alpha).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMGSZ, IMGSZ), mode='bilinear', align_corners=False)
        c = cam[0, 0]
        c = (c - c.min()) / (c.max() - c.min() + 1e-8)
        return c.cpu().numpy(), target_cls, logits


def cam_to_boxes(cam, orig_w, orig_h, thr=CAM_THRESH, min_area_ratio=MIN_AREA_RATIO):
    """热力图(IMGSZ 尺寸) -> 伪 bbox 列表 [(cx,cy,w,h) 归一化到原图]"""
    cam_up = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask = (cam_up >= thr).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    min_area = orig_w * orig_h * min_area_ratio
    for i in range(1, n):  # 0 是背景
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        cx = (x + w / 2) / orig_w
        cy = (y + h / 2) / orig_h
        boxes.append((cx, cy, w / orig_w, h / orig_h))
    return boxes


def draw_boxes(img, boxes, out_path):
    """可视化: 原图 + 伪框"""
    vis = np.array(img).copy()
    for cx, cy, w, h in boxes:
        x1 = int((cx - w / 2) * vis.shape[1])
        y1 = int((cy - h / 2) * vis.shape[0])
        x2 = int((cx + w / 2) * vis.shape[1])
        y2 = int((cy + h / 2) * vis.shape[0])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 6)
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def _run_sample(cam, paths, vis_out):
    """通用: 对 paths 列表生成可视化"""
    print(f'[sample] 可视化 {len(paths)} 张')
    for i, p in enumerate(paths):
        img = Image.open(p).convert('RGB')
        hmap, cls_id, logits = cam.cam(img)
        boxes = cam_to_boxes(hmap, img.width, img.height)
        cls_name = p.parent.name
        draw_boxes(img, boxes, vis_out / f'{i:03d}_{cls_name}.jpg')
        print(f'  {p.parent.name:8s} 预测top1={cls_id:2d}  boxes={len(boxes)}  -> {vis_out.name}/{i:03d}_{cls_name}.jpg')
    print(f'[sample] 完成, 查看 {vis_out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=0, help='可视化前 N 张')
    ap.add_argument('--random', type=int, default=0, help='随机抽 N 张(覆盖各类)')
    ap.add_argument('--full', action='store_true', help='全量生成数据集')
    args = ap.parse_args()

    cam = GradCAM(CKPT)
    print(f'模型加载完成, 目标层 idx={TARGET_LAYER}')

    # 收集 train 图(flat)
    train_dir = DATA / 'train'
    img_paths = sorted(train_dir.glob('*/*.jpg'))
    print(f'train 图总数: {len(img_paths)}')

    if args.sample:
        vis_out = OUT / '_vis'
        vis_out.mkdir(parents=True, exist_ok=True)
        paths = img_paths[:args.sample] if args.sample > 0 else []
        _run_sample(cam, paths, vis_out)
        return

    if args.random:
        import random
        random.seed(42)
        vis_out = OUT / '_vis_random'
        vis_out.mkdir(parents=True, exist_ok=True)
        paths = random.sample(img_paths, args.random)
        _run_sample(cam, paths, vis_out)
        return

    if args.full:
        # 组装数据集结构: train + val 都生成伪框
        # 正常类: 空 label (作为负样本降 FP)
        # 瑕疵类: CAM 阈值化生成 bbox
        # 类别: 28 类(含正常)
        splits = {
            'train': sorted((DATA / 'train').glob('*/*.jpg')),
            'val':   sorted((DATA / 'val').glob('*/*.jpg')),
        }
        class_names = sorted([d.name for d in (DATA / 'train').iterdir()])
        cls2id = {c: i for i, c in enumerate(class_names)}
        print(f'类别数: {len(class_names)}')
        print(f'  train: {len(splits["train"])}, val: {len(splits["val"])}')

        t0 = time.time()
        total_box = 0
        stats = {}  # cls -> [n_imgs, n_box, n_empty]
        for split, paths in splits.items():
            imgs_out = OUT / 'images' / split
            labels_out = OUT / 'labels' / split
            imgs_out.mkdir(parents=True, exist_ok=True)
            labels_out.mkdir(parents=True, exist_ok=True)
            print(f'\n=== {split}: {len(paths)} 张 ===')
            for i, p in enumerate(paths):
                cls_name = p.parent.name
                img = Image.open(p).convert('RGB')
                hmap, cls_id, logits = cam.cam(img)
                # 正常类 -> 空 label; 瑕疵类 -> 阈值化框
                if cls_name in NORMAL_CLASSES:
                    boxes = []
                else:
                    boxes = cam_to_boxes(hmap, img.width, img.height)
                # 写图
                dst_img = imgs_out / f'{i:05d}.jpg'
                cv2.imwrite(str(dst_img), cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
                # 写 label
                cid = cls2id[cls_name]
                lbl = labels_out / f'{i:05d}.txt'
                with open(lbl, 'w') as f:
                    for (cx, cy, w, h) in boxes:
                        f.write(f'{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
                total_box += len(boxes)
                stats.setdefault(cls_name, [0, 0, 0])
                stats[cls_name][0] += 1
                stats[cls_name][1] += len(boxes)
                if not boxes:
                    stats[cls_name][2] += 1
                if (i + 1) % 200 == 0:
                    el = time.time() - t0
                    print(f'  {split} {i+1}/{len(paths)}  {el:.0f}s  boxes累计={total_box}')
        el = time.time() - t0
        print(f'\n[full] 完成, 耗时 {el:.0f}s, 总框数 {total_box}')
        print(f'  类别统计 (图/框/空label):')
        for c, (imgs, bx, empty) in sorted(stats.items(), key=lambda x: -x[1][1]):
            print(f'    {c:10s}  图{imgs:4d}  框{bx:5d}  空标签{empty:4d}')
        # 写 data.yaml
        with open(OUT / 'data.yaml', 'w', encoding='utf-8') as f:
            f.write(f'path: {OUT.as_posix()}\n')
            f.write(f'train: images/train\n')
            f.write(f'val: images/val\n')
            f.write(f'nc: {len(class_names)}\n')
            f.write(f'names: {class_names}\n')
        print(f'  data.yaml -> {OUT / "data.yaml"}')
        return


if __name__ == '__main__':
    freeze_support()
    main()