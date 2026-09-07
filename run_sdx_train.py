"""SteelDefectX 直接训练 YOLOv8s (M1 实验: imgsz 256→320, 2026-08-14)"""
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")
    results = model.train(
        data=r"D:\RK3568&Orin Nano\ORIN NANO\Model Training\datasets\SteelDefectX_YOLO\data.yaml",
        imgsz=320,
        batch=64,
        epochs=100,
        patience=20,
        seed=42,
        workers=0,
        device=0,
        optimizer="auto",
        amp=True,
        cos_lr=True,
        close_mosaic=10,
        name="sdx_v8s_320",
        project=r"D:\RK3568&Orin Nano\ORIN NANO\Model Training\runs\detect",
        plots=False,
        exist_ok=True,
    )
    print("DONE", results)
