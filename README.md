# Defect Detection — Model Training

Training scripts, evaluation tools, and experiment records for industrial surface defect detection using Ultralytics YOLOv8 and YOLOv11.

> **Note:** This repository contains training code, configuration, and documentation only. Model weights, datasets, and training artifacts (`runs/`, `logs/`, `models/`, `datasets/`) are excluded due to size. Prepare them locally before training.

## Features

- **Multi-dataset fusion** — training pipelines for NEU-DET, GC10-DET, SDX, SD10, MVIT, and custom production datasets (up to 39 defect classes)
- **Block training** — staged fine-tuning with configurable learning rates per block, supporting resume from any checkpoint
- **Production model training** — dedicated scripts for production-grade models with class balancing, weak-class augmentation, and background false-positive testing
- **Comprehensive evaluation** — per-class mAP analysis, confusion matrix generation, cross-model comparison, and TTA (test-time augmentation) inference
- **Deployment utilities** — ONNX export and TensorRT compilation helpers

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| Python | 3.10+ |
| Framework | Ultralytics (YOLOv8/v11) |
| GPU | NVIDIA GPU with CUDA 11.8+ (recommended) |
| Dataset | Prepare locally — see `scripts/dataset/` for download and merge scripts |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/logicness/defect-detection-training.git
cd defect-detection-training

# Install dependencies
pip install ultralytics

# Download and prepare public datasets
python scripts/dataset/download_public_datasets.py
python scripts/dataset/download_steeldefectx.py

# Train a baseline model
python scripts/train_neu.py

# Evaluate
python scripts/eval_model.py --weights runs/detect/train/weights/best.pt
```

## Project Structure

```
├── scripts/
│   ├── train/                 # Production model training scripts
│   │   ├── train_production_v2.py    # Main production training pipeline
│   │   ├── univ39_block.py           # Block training for 39-class model
│   │   └── um39_train.yaml           # Training configuration
│   ├── eval/                  # Evaluation and comparison tools
│   │   ├── gen_compare_v8m_v11m.py   # YOLOv8m vs v11m comparison
│   │   └── gen_final_reports.py      # Final evaluation reports
│   ├── dataset/               # Dataset download, merge, and augmentation
│   │   ├── download_public_datasets.py
│   │   ├── build_production_v21.py   # Production v2.1 dataset builder
│   │   ├── augment_weak_classes.py   # Weak class oversampling
│   │   └── collect_weak_class_data.py
│   └── diag/                  # Diagnostics and annotation auditing
│       ├── audit_annotation.py       # Label quality audit
│       ├── diagnose_confusion.py     # Confusion matrix analysis
│       └── test_background_fp.py     # Background false-positive test
├── *.py                       # Top-level training and evaluation scripts
├── neu.yaml                   # NEU-DET dataset configuration
├── LICENSE
└── README.md
```

> **Not included:** Experiment reports, diagnostic data (annotation renders, confusion matrices, label audits), evaluation result JSONs, and device deployment scripts are available in the private companion repository.

## Configuration

All paths in scripts use placeholder variables for portability. Replace them with your local paths before running:

| Placeholder | Meaning |
|-------------|---------|
| `[LOCAL_MODEL_PATH]` | Root directory for model weights and checkpoints |
| `[LOCAL_ROOT]` | Root directory for datasets and training data |
| `[proxy-host]:[proxy-port]` | Optional HTTP proxy for downloading pretrained weights |

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
