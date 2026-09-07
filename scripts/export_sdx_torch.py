"""
SteelDefectX ONNX 导出脚本（绕过 ultralytics AutoUpdate）
使用 torch.onnx.export 直接导出
"""
import torch
from ultralytics import YOLO
import sys

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/detect/sdx_v8s/weights/best.pt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "SteelDefectX_v8s_25c_256.onnx"

    print(f"Loading: {model_path}")
    yolo = YOLO(model_path)
    model = yolo.model.fuse()  # 融合后的模型
    model.eval()

    # 动态 batch + 固定 256x256
    img = torch.zeros((1, 3, 256, 256))

    print("Exporting ONNX (torch.onnx.export)...")
    torch.onnx.export(
        model,
        img,
        output_path,
        export_params=True,
        opset_version=13,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={
            "images": {0: "batch"},
            "output": {0: "batch"},
        },
    )

    import os
    size = os.path.getsize(output_path)
    print(f"Done: {output_path} ({size/1024/1024:.1f} MB)")
