"""SteelDefectX 导出 ONNX（用于 Nano TensorRT 部署）"""
from ultralytics import YOLO
import sys

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/detect/sdx_v8s/weights/best.pt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "SteelDefectX_v8s_25c_256.onnx"

    print(f"Loading: {model_path}")
    model = YOLO(model_path)

    print(f"Exporting ONNX (imgsz=256, dynamic batch)...")
    success = model.export(
        format="onnx",
        imgsz=256,
        dynamic=True,     # 动态 batch，方便 Nano 推理时调整
        simplify=True,    # 简化模型
        opset=13,         # ONNX opset 13
    )

    print(f"Done: {success}")
    print(f"Output: {success}")
