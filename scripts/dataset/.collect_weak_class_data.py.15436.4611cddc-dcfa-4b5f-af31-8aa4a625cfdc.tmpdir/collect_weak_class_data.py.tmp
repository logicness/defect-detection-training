# -*- coding: utf-8 -*-
"""
弱类增强数据收集脚本
目的: 收集当前模型表现较差的缺陷类型数据

当前弱类 (mAP50 < 0.6):
- Waist folding: 0.3112 (最弱)
- Rolled pit: 0.5779
- White rust: 0.6812
- Bright scratch: 0.5624
- Crazing: 0.6116
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

# 弱类定义
WEAK_CLASSES = {
    "Waist folding": {
        "current_map50": 0.3112,
        "target_map50": 0.60,
        "need_samples": 200,
        "search_keywords": ["steel waist folding defect", "metal folding surface", "waist fold steel"],
        "description": "腰折缺陷 - 钢材腰部折叠"
    },
    "Rolled pit": {
        "current_map50": 0.5779,
        "target_map50": 0.70,
        "need_samples": 100,
        "search_keywords": ["rolled pit steel defect", "rolling pit metal surface", "pit defect steel"],
        "description": "轧制凹坑 - 轧制过程中产生的凹陷"
    },
    "White rust": {
        "current_map50": 0.6812,
        "target_map50": 0.75,
        "need_samples": 150,
        "search_keywords": ["white rust steel", "zinc rust surface", "white corrosion metal"],
        "description": "白锈 - 金属表面白色锈蚀"
    },
    "Bright scratch": {
        "current_map50": 0.5624,
        "target_map50": 0.70,
        "need_samples": 100,
        "search_keywords": ["bright scratch metal", "surface scratch steel", "metal scratch defect"],
        "description": "亮划痕 - 金属表面明亮划痕"
    },
    "Crazing": {
        "current_map50": 0.6116,
        "target_map50": 0.75,
        "need_samples": 100,
        "search_keywords": ["crazing defect metal", "surface crazing steel", "crack network metal"],
        "description": "龟裂 - 金属表面网状裂纹"
    }
}

def create_collection_plan():
    """创建数据收集计划"""
    plan = {
        "created_at": datetime.now().isoformat(),
        "purpose": "提升弱类缺陷识别准确率",
        "weak_classes": WEAK_CLASSES,
        "collection_strategy": {
            "sources": [
                "1. 内部产线采集 (优先)",
                "2. 公开数据集下载",
                "3. 网络图片搜索",
                "4. 数据增强生成"
            ],
            "requirements": {
                "image_size": ">=256x256",
                "format": "JPG/PNG",
                "annotation": "YOLO format (.txt)",
                "quality": "清晰、无模糊、多角度"
            }
        },
        "priority_order": [
            "Waist folding (最弱, 需200+张)",
            "White rust (需150+张)",
            "Bright scratch (需100+张)",
            "Crazing (需100+张)",
            "Rolled pit (需100+张)"
        ]
    }
    
    return plan

def create_directory_structure(base_dir):
    """创建数据收集目录结构"""
    dirs = [
        "raw/images",      # 原始图片
        "raw/labels",      # 原始标注
        "processed/images", # 处理后图片
        "processed/labels", # 处理后标注
        "augmented/images", # 增强图片
        "augmented/labels", # 增强标注
        "negative_samples", # 负样本
    ]
    
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
        print(f"Created: {os.path.join(base_dir, d)}")
    
    # 为每个弱类创建子目录
    for class_name in WEAK_CLASSES.keys():
        safe_name = class_name.replace(" ", "_").lower()
        os.makedirs(os.path.join(base_dir, "raw/images", safe_name), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "raw/labels", safe_name), exist_ok=True)
        print(f"Created class dir: {safe_name}")

def generate_search_queries():
    """生成搜索查询词"""
    queries = {}
    for class_name, info in WEAK_CLASSES.items():
        queries[class_name] = {
            "keywords": info["search_keywords"],
            "need": info["need_samples"],
            "description": info["description"]
        }
    return queries

def create_annotation_template(image_dir, label_dir, class_id):
    """创建标注模板"""
    template = f"""# YOLO 标注格式
# 每行: class_id x_center y_center width height
# 所有值归一化到 0-1

# 示例:
# {class_id} 0.5 0.5 0.3 0.4

# 使用工具标注:
# 1. LabelImg: https://github.com/tzutalin/labelImg
# 2. CVAT: https://cvat.org/
# 3. Roboflow: https://roboflow.com/
"""
    return template

def main():
    """主函数"""
    print("=" * 60)
    print("弱类增强数据收集计划")
    print("=" * 60)
    
    # 1. 创建收集计划
    plan = create_collection_plan()
    
    # 保存计划
    plan_dir = Path(__file__).parent.parent.parent / "datasets_master" / "weak_class_collection"
    plan_dir.mkdir(parents=True, exist_ok=True)
    
    plan_file = plan_dir / "collection_plan.json"
    with open(plan_file, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    print(f"\n[1/4] 收集计划已保存: {plan_file}")
    
    # 2. 创建目录结构
    print(f"\n[2/4] 创建目录结构...")
    create_directory_structure(str(plan_dir))
    
    # 3. 生成搜索查询
    queries = generate_search_queries()
    queries_file = plan_dir / "search_queries.json"
    with open(queries_file, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    print(f"\n[3/4] 搜索查询已保存: {queries_file}")
    
    # 4. 生成收集指南
    guide = f"""# 弱类数据收集指南

## 收集优先级

"""
    for i, (class_name, info) in enumerate(WEAK_CLASSES.items(), 1):
        guide += f"""### {i}. {class_name}
- 当前 mAP50: {info['current_map50']:.4f}
- 目标 mAP50: {info['target_map50']:.4f}
- 需收集: {info['need_samples']} 张
- 描述: {info['description']}

"""
    
    guide += """## 收集方法

### 方法1: 内部产线采集 (优先)
1. 联系产线质检部门
2. 收集历史缺陷图片
3. 安排现场拍摄

### 方法2: 公开数据集
1. NEU-DET (东北大学钢铁缺陷数据集)
2. GC10-DET (已收集)
3. DAGM 2007 (已收集)

### 方法3: 网络搜索
1. Google 图片搜索 (使用 search_queries.json)
2. 学术论文附带数据
3. 技术论坛分享

### 方法4: 数据增强
1. 使用现有图片进行增强
2. 几何变换 (旋转、翻转、缩放)
3. 颜色变换 (亮度、对比度、饱和度)

## 标注工具

1. **LabelImg** (推荐)
   - 安装: pip install labelimg
   - 运行: labelimg
   - 格式选择: YOLO

2. **CVAT** (在线)
   - 地址: https://cvat.org/
   - 支持多人协作

3. **Roboflow** (在线)
   - 地址: https://roboflow.com/
   - 自动标注功能

## 标注规范

1. 边界框紧贴缺陷边缘
2. 遮挡 >50% 不标注
3. 每张图至少 1 个标注
4. 使用统一类别名称

## 文件命名规范

```
{class_name}_{序号}_{日期}.jpg
例: waist_folding_001_20260906.jpg
```

## 目录结构

```
datasets_master/weak_class_collection/
├── raw/
│   ├── images/
│   │   ├── waist_folding/
│   │   ├── rolled_pit/
│   │   ├── white_rust/
│   │   ├── bright_scratch/
│   │   └── crazing/
│   └── labels/
│       ├── waist_folding/
│       ├── rolled_pit/
│       ├── white_rust/
│       ├── bright_scratch/
│       └── crazing/
├── processed/
│   ├── images/
│   └── labels/
├── augmented/
│   ├── images/
│   └── labels/
├── negative_samples/
├── collection_plan.json
├── search_queries.json
└── README.md
```

## 下一步

1. 根据优先级收集数据
2. 使用标注工具进行标注
3. 运行质量检查脚本
4. 合并到主数据集
5. 启动新一轮训练
"""
    
    guide_file = plan_dir / "README.md"
    with open(guide_file, "w", encoding="utf-8") as f:
        f.write(guide)
    print(f"\n[4/4] 收集指南已保存: {guide_file}")
    
    print("\n" + "=" * 60)
    print("数据收集计划创建完成!")
    print("=" * 60)
    print(f"\n计划目录: {plan_dir}")
    print(f"\n下一步:")
    print(f"  1. 查看 README.md 了解收集方法")
    print(f"  2. 根据优先级收集数据")
    print(f"  3. 使用标注工具进行标注")
    print(f"  4. 完成后告诉我，我帮你合并数据集并训练")

if __name__ == "__main__":
    main()
