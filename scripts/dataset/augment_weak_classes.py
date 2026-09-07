# -*- coding: utf-8 -*-
"""
弱类数据增强脚本
目的: 通过数据增强提升弱类样本数量

增强策略:
1. 几何变换: 旋转、翻转、缩放、平移
2. 颜色变换: 亮度、对比度、饱和度、色调
3. 噪声添加: 高斯噪声、椒盐噪声
4. 模糊处理: 高斯模糊、运动模糊
5. 裁剪增强: 随机裁剪、中心裁剪
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
from datetime import datetime
import json
import shutil

# 弱类定义
WEAK_CLASSES = {
    "Waist folding": {"class_id": 21, "current_samples": 59, "target_samples": 300},
    "Rolled pit": {"class_id": 17, "current_samples": 15, "target_samples": 200},
    "White rust": {"class_id": 24, "current_samples": 45, "target_samples": 250},
    "Bright scratch": {"class_id": 0, "current_samples": 100, "target_samples": 300},
    "Crazing": {"class_id": 1, "current_samples": 50, "target_samples": 250}
}

class DataAugmentor:
    """数据增强器"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建输出目录
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
    
    def rotate_image(self, image, angle):
        """旋转图片"""
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated, M
    
    def rotate_bbox(self, bbox, M, img_w, img_h):
        """旋转边界框"""
        x_center, y_center, w, h = bbox
        
        # 转换为像素坐标
        x1 = (x_center - w/2) * img_w
        y1 = (y_center - h/2) * img_h
        x2 = (x_center + w/2) * img_w
        y2 = (y_center + h/2) * img_h
        
        # 旋转四个角点
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)
        
        # 添加齐次坐标
        ones = np.ones(shape=(len(corners), 1))
        corners_ones = np.hstack([corners, ones])
        
        # 旋转
        rotated_corners = M.dot(corners_ones.T).T
        
        # 获取新的边界框
        x_coords = rotated_corners[:, 0]
        y_coords = rotated_corners[:, 1]
        
        x_min = max(0, min(x_coords))
        x_max = min(img_w, max(x_coords))
        y_min = max(0, min(y_coords))
        y_max = min(img_h, max(y_coords))
        
        # 转换回YOLO格式
        new_x_center = (x_min + x_max) / 2 / img_w
        new_y_center = (y_min + y_max) / 2 / img_h
        new_w = (x_max - x_min) / img_w
        new_h = (y_max - y_min) / img_h
        
        return [new_x_center, new_y_center, new_w, new_h]
    
    def flip_image(self, image, direction="horizontal"):
        """翻转图片"""
        if direction == "horizontal":
            return cv2.flip(image, 1)
        elif direction == "vertical":
            return cv2.flip(image, 0)
        elif direction == "both":
            return cv2.flip(image, -1)
        return image
    
    def flip_bbox(self, bbox, direction="horizontal", img_w=1, img_h=1):
        """翻转边界框"""
        x_center, y_center, w, h = bbox
        
        if direction == "horizontal":
            new_x_center = 1.0 - x_center
            return [new_x_center, y_center, w, h]
        elif direction == "vertical":
            new_y_center = 1.0 - y_center
            return [x_center, new_y_center, w, h]
        elif direction == "both":
            return [1.0 - x_center, 1.0 - y_center, w, h]
        return bbox
    
    def adjust_brightness(self, image, factor):
        """调整亮度"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def adjust_contrast(self, image, factor):
        """调整对比度"""
        mean = np.mean(image)
        adjusted = np.clip((image - mean) * factor + mean, 0, 255)
        return adjusted.astype(np.uint8)
    
    def add_gaussian_noise(self, image, mean=0, sigma=25):
        """添加高斯噪声"""
        noise = np.random.normal(mean, sigma, image.shape)
        noisy = np.clip(image + noise, 0, 255)
        return noisy.astype(np.uint8)
    
    def gaussian_blur(self, image, kernel_size=5):
        """高斯模糊"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def augment_single(self, image_path, label_path, output_prefix, num_augmentations=5):
        """增强单张图片"""
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not read {image_path}")
            return []
        
        h, w = image.shape[:2]
        
        # 读取标注
        bboxes = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        bboxes.append((class_id, bbox))
        
        augmented = []
        
        for i in range(num_augmentations):
            aug_image = image.copy()
            aug_bboxes = bboxes.copy()
            
            # 随机选择增强方法
            aug_type = random.choice([
                "rotate", "flip_h", "flip_v", 
                "brightness", "contrast", "noise", "blur"
            ])
            
            if aug_type == "rotate":
                angle = random.uniform(-30, 30)
                aug_image, M = self.rotate_image(aug_image, angle)
                aug_bboxes = [(cls, self.rotate_bbox(bbox, M, w, h)) for cls, bbox in aug_bboxes]
            
            elif aug_type == "flip_h":
                aug_image = self.flip_image(aug_image, "horizontal")
                aug_bboxes = [(cls, self.flip_bbox(bbox, "horizontal")) for cls, bbox in aug_bboxes]
            
            elif aug_type == "flip_v":
                aug_image = self.flip_image(aug_image, "vertical")
                aug_bboxes = [(cls, self.flip_bbox(bbox, "vertical")) for cls, bbox in aug_bboxes]
            
            elif aug_type == "brightness":
                factor = random.uniform(0.7, 1.3)
                aug_image = self.adjust_brightness(aug_image, factor)
            
            elif aug_type == "contrast":
                factor = random.uniform(0.7, 1.3)
                aug_image = self.adjust_contrast(aug_image, factor)
            
            elif aug_type == "noise":
                aug_image = self.add_gaussian_noise(aug_image)
            
            elif aug_type == "blur":
                aug_image = self.gaussian_blur(aug_image)
            
            # 保存增强后的图片
            output_image = self.output_dir / "images" / f"{output_prefix}_{aug_type}_{i:03d}.jpg"
            cv2.imwrite(str(output_image), aug_image)
            
            # 保存增强后的标注
            output_label = self.output_dir / "labels" / f"{output_prefix}_{aug_type}_{i:03d}.txt"
            with open(output_label, "w") as f:
                for class_id, bbox in aug_bboxes:
                    f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\\n")
            
            augmented.append({
                "image": str(output_image),
                "label": str(output_label),
                "augmentation": aug_type
            })
        
        return augmented
    
    def augment_class(self, class_name, input_dir, num_augmentations=5):
        """增强整个类别"""
        print(f"\\nAugmenting class: {class_name}")
        
        input_path = Path(input_dir)
        image_dir = input_path / "images"
        label_dir = input_path / "labels"
        
        if not image_dir.exists():
            print(f"Warning: {image_dir} does not exist")
            return []
        
        all_augmented = []
        
        # 遍历所有图片
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        print(f"Found {len(image_files)} images")
        
        for image_file in image_files:
            label_file = label_dir / (image_file.stem + ".txt")
            
            augmented = self.augment_single(
                image_file, 
                label_file, 
                image_file.stem,
                num_augmentations
            )
            all_augmented.extend(augmented)
        
        print(f"Generated {len(all_augmented)} augmented samples")
        return all_augmented

def main():
    """主函数"""
    print("=" * 60)
    print("弱类数据增强")
    print("=" * 60)
    
    # 设置路径
    base_dir = Path(__file__).parent.parent.parent / "datasets_master" / "weak_class_collection"
    output_dir = base_dir / "augmented"
    
    # 创建增强器
    augmentor = DataAugmentor(output_dir)
    
    # 增强每个弱类
    all_results = {}
    
    for class_name, info in WEAK_CLASSES.items():
        safe_name = class_name.replace(" ", "_").lower()
        input_dir = base_dir / "raw" / safe_name
        
        if input_dir.exists():
            augmented = augmentor.augment_class(class_name, input_dir, num_augmentations=5)
            all_results[class_name] = {
                "input_samples": info["current_samples"],
                "augmented_samples": len(augmented),
                "total_samples": info["current_samples"] + len(augmented)
            }
        else:
            print(f"\\nSkipping {class_name}: {input_dir} does not exist")
            all_results[class_name] = {
                "input_samples": info["current_samples"],
                "augmented_samples": 0,
                "total_samples": info["current_samples"]
            }
    
    # 保存结果
    results_file = output_dir / "augmentation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "created_at": datetime.now().isoformat(),
            "results": all_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\\n{'=' * 60}")
    print("数据增强完成!")
    print("=" * 60)
    print(f"\\n结果已保存: {results_file}")
    print(f"\\n增强统计:")
    for class_name, result in all_results.items():
        print(f"  {class_name}: {result['input_samples']} -> {result['total_samples']} (+{result['augmented_samples']})")

if __name__ == "__main__":
    main()
