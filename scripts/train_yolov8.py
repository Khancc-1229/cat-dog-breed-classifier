"""
YOLOv8-cls 迁移学习训练脚本
基于 yolov8n-cls.pt (ImageNet 预训练) 微调
使用方法: python scripts/train_yolov8.py
"""

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent

# ========== 配置 ==========
DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"

# YOLOv8 分类模型（ImageNet 预训练）
PRETRAINED = "yolov8n-cls.pt"  # 自动下载

# 训练参数
EPOCHS = 50
IMG_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DEVICE = 0  # CUDA GPU，没有 GPU 改成 "cpu"


def main():
    print("=" * 60)
    print("猫狗品种识别 —— YOLOv8-cls 训练")
    print(f"预训练权重: {PRETRAINED}")
    print(f"数据集: {DATASET_DIR}")
    print("=" * 60)

    # 加载预训练模型
    print("\n[1/3] 加载 ImageNet 预训练权重...")
    model = YOLO(PRETRAINED)

    # 训练
    print("\n[2/3] 开始训练...")
    results = model.train(
        data=str(DATASET_DIR),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        device=DEVICE,
        project=str(MODEL_DIR),
        name="yolov8_cls_breeds",
        exist_ok=True,
        # 数据增强
        augment=True,
        hsv_h=0.015,   # 色调增强（弱，猫狗颜色重要）
        hsv_s=0.3,     # 饱和度
        hsv_v=0.2,     # 亮度
        degrees=10,    # 旋转 ±10 度
        translate=0.1,  # 平移
        scale=0.3,     # 缩放
        shear=2,       # 剪切
        flipud=0.0,    # 上下翻转（关，猫狗不会倒立）
        fliplr=0.5,    # 左右翻转
        erasing=0.1,   # 随机擦除
    )

    # 验证
    print("\n[3/3] 验证集评估...")
    model = YOLO(str(MODEL_DIR / "yolov8_cls_breeds" / "weights" / "best.pt"))
    val_results = model.val(data=str(DATASET_DIR), split="val")
    print(f"\nYOLOv8-cls Top-1: {val_results.top1:.2%}")
    print(f"YOLOv8-cls Top-5: {val_results.top5:.2%}")

    print(f"\n模型已保存到: {MODEL_DIR / 'yolov8_cls_breeds' / 'weights' / 'best.pt'}")
    return val_results


if __name__ == "__main__":
    main()
