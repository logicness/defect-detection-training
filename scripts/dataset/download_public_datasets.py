# -*- coding: utf-8 -*-
"""
公开数据集下载脚本
目的: 下载可用于提升弱类性能的公开数据集

可用数据集:
1. NEU-DET (东北大学钢铁缺陷数据集) - 6类热轧钢带缺陷
2. GC10-DET (已收集) - 10类钢铁缺陷
3. DAGM 2007 (已收集) - 10类表面缺陷
4. Severstal Steel Defect Detection - Kaggle竞赛数据集
5. X-SDD - 12类钢铁表面缺陷数据集
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

# 数据集信息
DATASETS = {
    "NEU-DET": {
        "name": "Northeastern University - Surface Defect Database",
        "description": "东北大学热轧钢带表面缺陷数据库",
        "classes": ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"],
        "num_classes": 6,
        "images_per_class": 300,
        "total_images": 1800,
        "format": "BMP",
        "annotation": "VOC (XML)",
        "license": "学术研究",
        "download_urls": [
            "https://github.com/DeepLeetcw/NEU-DET",
            "https://www.kaggle.com/datasets/new-steel-surface-defect-database"
        ],
        "relevance": {
            "crazing": "高 - 直接对应弱类 Crazing",
            "inclusion": "中 - 可补充 Inclusion 样本",
            "patches": "中 - 可补充 Patches 样本",
            "pitted_surface": "中 - 可补充 Pitted surface 样本",
            "rolled_in_scale": "中 - 可补充 Rolled in scale 样本",
            "scratches": "高 - 可补充 Bright scratch 样本"
        }
    },
    "Severstal": {
        "name": "Severstal Steel Defect Detection",
        "description": "Severstal钢铁缺陷检测竞赛数据集",
        "classes": ["defect_1", "defect_2", "defect_3", "defect_4"],
        "num_classes": 4,
        "images_per_class": "varying",
        "total_images": 12568,
        "format": "JPG",
        "annotation": "RLE (需转换)",
        "license": "Kaggle竞赛",
        "download_urls": [
            "https://www.kaggle.com/c/severstal-steel-defect-detection/data"
        ],
        "relevance": {
            "defect_1": "中 - 可能对应划痕类缺陷",
            "defect_2": "中 - 可能对应夹杂类缺陷",
            "defect_3": "中 - 可能对应裂纹类缺陷",
            "defect_4": "中 - 可能对应其他缺陷"
        }
    },
    "X-SDD": {
        "name": "X-SDD Steel Surface Defect Dataset",
        "description": "X-SDD钢铁表面缺陷数据集",
        "classes": ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches", "waist_folding", "welding_line"],
        "num_classes": 8,
        "images_per_class": "varying",
        "total_images": 2000,
        "format": "JPG",
        "annotation": "YOLO",
        "license": "学术研究",
        "download_urls": [
            "https://github.com/X-SDD/X-SDD"
        ],
        "relevance": {
            "waist_folding": "高 - 直接对应弱类 Waist folding",
            "crazing": "高 - 直接对应弱类 Crazing",
            "scratches": "高 - 可补充 Bright scratch 样本",
            "welding_line": "中 - 可补充 Welding line 样本"
        }
    }
}

def create_download_plan():
    """创建下载计划"""
    plan = {
        "created_at": datetime.now().isoformat(),
        "purpose": "下载公开数据集提升弱类性能",
        "datasets": DATASETS,
        "priority": [
            "1. X-SDD - 包含 waist_folding (最弱类)",
            "2. NEU-DET - 包含 crazing 和 scratches",
            "3. Severstal - 大量钢铁缺陷数据"
        ],
        "download_steps": {
            "step1": "访问下载链接",
            "step2": "下载数据集压缩包",
            "step3": "解压到 datasets_master/raw/ 目录",
            "step4": "运行格式转换脚本",
            "step5": "合并到主数据集"
        }
    }
    return plan

def create_conversion_scripts():
    """创建格式转换脚本"""
    scripts = {}
    
    # VOC to YOLO conversion
    voc_to_yolo = '''# -*- coding: utf-8 -*-
"""
VOC (XML) -> YOLO 格式转换
用于转换 NEU-DET 数据集
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

# 类别映射
CLASS_MAP = {
    "crazing": 1,  # 对应我们的 Crazing (class_id=1)
    "inclusion": 6,  # 对应我们的 Inclusion (class_id=6)
    "patches": 12,  # 对应我们的 Patches (class_id=12)
    "pitted_surface": 13,  # 对应我们的 Pitted surface (class_id=13)
    "rolled_in_scale": 16,  # 对应我们的 Rolled in scale (class_id=16)
    "scratches": 0,  # 对应我们的 Bright scratch (class_id=0)
}

def convert_voc_to_yolo(xml_file, output_dir):
    """转换单个VOC XML文件到YOLO格式"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 获取图片尺寸
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    
    # 转换每个目标
    yolo_lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_MAP:
            continue
        
        class_id = CLASS_MAP[name]
        
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        
        # 转换为YOLO格式 (center_x, center_y, width, height)
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        
        yolo_lines.append(f"{class_id} {x_center} {y_center} {w} {h}")
    
    # 保存YOLO格式文件
    output_file = os.path.join(output_dir, Path(xml_file).stem + ".txt")
    with open(output_file, "w") as f:
        f.write("\\n".join(yolo_lines))
    
    return output_file

def batch_convert(input_dir, output_dir):
    """批量转换"""
    os.makedirs(output_dir, exist_ok=True)
    
    xml_files = list(Path(input_dir).glob("*.xml"))
    print(f"Found {len(xml_files)} XML files")
    
    for xml_file in xml_files:
        convert_voc_to_yolo(str(xml_file), output_dir)
    
    print(f"Converted {len(xml_files)} files to {output_dir}")
'''
    scripts["voc_to_yolo.py"] = voc_to_yolo
    
    # RLE to YOLO conversion (for Severstal)
    rle_to_yolo = '''# -*- coding: utf-8 -*-
"""
RLE -> YOLO 格式转换
用于转换 Severstal 数据集
"""

import os
import numpy as np
from pathlib import Path

def rle_to_mask(rle_string, height, width):
    """将 RLE 编码转换为 mask"""
    if rle_string == "":
        return np.zeros((height, width), dtype=np.uint8)
    
    runs = np.array([int(x) for x in rle_string.split()]).reshape(-1, 2)
    mask = np.zeros(height * width, dtype=np.uint8)
    
    for start, length in runs:
        mask[start:start + length] = 1
    
    return mask.reshape(height, width)

def mask_to_yolo_bbox(mask):
    """将 mask 转换为 YOLO 边界框"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any():
        return None
    
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    height, width = mask.shape
    x_center = (xmin + xmax) / 2 / width
    y_center = (ymin + ymax) / 2 / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    
    return x_center, y_center, w, h

def convert_rle_to_yolo(csv_file, output_dir):
    """转换 RLE 格式到 YOLO 格式"""
    import pandas as pd
    
    df = pd.read_csv(csv_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 按图片分组
    for image_id, group in df.groupby("ImageId"):
        yolo_lines = []
        
        for _, row in group.iterrows():
            if pd.isna(row["EncodedPixels"]):
                continue
            
            class_id = int(row["ClassId"]) - 1  # 转换为 0-based
            
            # 这里需要图片尺寸，假设为 256x1600
            # 实际使用时需要根据图片调整
            mask = rle_to_mask(row["EncodedPixels"], 256, 1600)
            bbox = mask_to_yolo_bbox(mask)
            
            if bbox:
                x_center, y_center, w, h = bbox
                yolo_lines.append(f"{class_id} {x_center} {y_center} {w} {h}")
        
        # 保存
        output_file = os.path.join(output_dir, image_id.replace(".jpg", ".txt"))
        with open(output_file, "w") as f:
            f.write("\\n".join(yolo_lines))
    
    print(f"Converted {len(df)} annotations to YOLO format")
'''
    scripts["rle_to_yolo.py"] = rle_to_yolo
    
    return scripts

def main():
    """主函数"""
    print("=" * 60)
    print("公开数据集下载计划")
    print("=" * 60)
    
    # 创建下载计划
    plan = create_download_plan()
    
    # 保存计划
    plan_dir = Path(__file__).parent.parent.parent / "datasets_master" / "public_datasets"
    plan_dir.mkdir(parents=True, exist_ok=True)
    
    plan_file = plan_dir / "download_plan.json"
    with open(plan_file, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    print(f"\n[1/3] 下载计划已保存: {plan_file}")
    
    # 创建转换脚本
    scripts = create_conversion_scripts()
    scripts_dir = plan_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    
    for script_name, script_content in scripts.items():
        script_file = scripts_dir / script_name
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(script_content)
    print(f"\n[2/3] 转换脚本已保存: {scripts_dir}")
    
    # 生成下载指南
    guide = f"""# 公开数据集下载指南

## 数据集优先级

### 1. X-SDD (最高优先级)
**原因**: 包含 waist_folding (当前最弱类 mAP50=0.3112)

- 下载地址: https://github.com/X-SDD/X-SDD
- 包含类别: crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratches, waist_folding, welding_line
- 图片数量: ~2000张
- 格式: JPG + YOLO标注

**下载步骤**:
1. 访问 https://github.com/X-SDD/X-SDD
2. 点击 "Code" -> "Download ZIP"
3. 解压到 datasets_master/public_datasets/X-SDD/
4. 复制图片和标注到对应目录

### 2. NEU-DET (高优先级)
**原因**: 包含 crazing (弱类) 和 scratches (可补充 bright scratch)

- 下载地址: https://github.com/DeepLeetcw/NEU-DET 或 Kaggle
- 包含类别: crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratches
- 图片数量: ~1800张 (每类300张)
- 格式: BMP + VOC标注

**下载步骤**:
1. 访问 https://www.kaggle.com/datasets/new-steel-surface-defect-database
2. 下载数据集
3. 解压到 datasets_master/public_datasets/NEU-DET/
4. 运行转换脚本: python scripts/dataset/convert_neu_det.py

### 3. Severstal (中优先级)
**原因**: 大量钢铁缺陷数据，可用于数据增强

- 下载地址: https://www.kaggle.com/c/severstal-steel-defect-detection/data
- 包含类别: 4类钢铁缺陷
- 图片数量: ~12568张
- 格式: JPG + RLE标注

**下载步骤**:
1. 访问 Kaggle 竞赛页面
2. 注册/登录 Kaggle 账号
3. 下载数据集
4. 解压到 datasets_master/public_datasets/Severstal/
5. 运行转换脚本: python scripts/dataset/convert_severstal.py

## 转换脚本使用

### VOC -> YOLO (NEU-DET)
```bash
python datasets_master/public_datasets/scripts/voc_to_yolo.py \\
  --input datasets_master/public_datasets/NEU-DET/annotations \\
  --output datasets_master/public_datasets/NEU-DET/labels
```

### RLE -> YOLO (Severstal)
```bash
python datasets_master/public_datasets/scripts/rle_to_yolo.py \\
  --csv datasets_master/public_datasets/Severstal/train.csv \\
  --output datasets_master/public_datasets/Severstal/labels
```

## 类别映射

| 原始类别 | 我们的类别ID | 我们的类别名称 |
|---|---|---|
| crazing | 1 | Crazing |
| inclusion | 6 | Inclusion |
| patches | 12 | Patches |
| pitted_surface | 13 | Pitted surface |
| rolled_in_scale | 16 | Rolled in scale |
| scratches | 0 | Bright scratch |
| waist_folding | 21 | Waist folding |
| welding_line | 23 | Welding line |

## 数据合并

下载并转换后，运行以下命令合并到主数据集:

```bash
python scripts/dataset/merge_public_datasets.py \\
  --datasets X-SDD NEU-DET Severstal \\
  --output datasets_master/production/v2.0
```

## 注意事项

1. **许可合规**: 确认数据集许可允许商业使用
2. **数据质量**: 检查图片质量和标注准确性
3. **类别映射**: 确保类别映射正确
4. **数据平衡**: 合并后检查数据分布
5. **负样本**: 适当添加负样本防止误检

## 下一步

1. 按优先级下载数据集
2. 运行转换脚本
3. 检查数据质量
4. 合并到主数据集
5. 启动新一轮训练
"""
    
    guide_file = plan_dir / "README.md"
    with open(guide_file, "w", encoding="utf-8") as f:
        f.write(guide)
    print(f"\n[3/3] 下载指南已保存: {guide_file}")
    
    print("\n" + "=" * 60)
    print("下载计划创建完成!")
    print("=" * 60)
    print(f"\n计划目录: {plan_dir}")
    print(f"\n下一步:")
    print(f"  1. 查看 README.md 了解下载步骤")
    print(f"  2. 按优先级下载数据集")
    print(f"  3. 运行转换脚本")
    print(f"  4. 完成后告诉我，我帮你合并并训练")

if __name__ == "__main__":
    main()
