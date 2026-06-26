"""
ResNet50 迁移学习训练脚本
基于 torchvision ResNet50_Weights.IMAGENET1K_V2 微调
使用方法: python scripts/train_resnet.py
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import json

PROJECT_ROOT = Path(__file__).parent.parent

# ========== 配置 ==========
DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2

# 早停参数
PATIENCE = 12  # 足够长，让 ResNet 充分收敛

# 数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="   训练", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "acc": f"{correct / total:.3f}"
        })

    return running_loss / len(loader), correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="   验证", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), correct / total


def main():
    print("=" * 60)
    print("猫狗品种识别 —— ResNet50 训练")
    print(f"设备: {DEVICE}")
    print(f"数据集: {DATASET_DIR}")
    print("=" * 60)

    # 加载数据集
    print("\n[1/4] 加载数据集...")
    train_dataset = datasets.ImageFolder(
        str(DATASET_DIR / "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(
        str(DATASET_DIR / "val"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    # 保存类别名称
    class_names = train_dataset.classes
    class_map_path = MODEL_DIR / "resnet50_breeds" / "class_names.json"
    class_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    print(f"  类别数: {len(class_names)}")
    print(f"  类别: {class_names}")
    print(f"  训练集: {len(train_dataset)} 张")
    print(f"  验证集: {len(val_dataset)} 张")

    # 构建模型
    print("\n[2/4] 加载 ImageNet 预训练 ResNet50...")
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)

    # 替换分类头
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(class_names)),
    )
    model = model.to(DEVICE)

    # 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 训练
    print("\n[3/4] 开始训练...")
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        print(f"\n  Epoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_path = MODEL_DIR / "resnet50_breeds" / "best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
            }, save_path)
            print(f"  ✅ 最佳模型已保存 (acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  早停：连续 {PATIENCE} 轮未提升")
                break

    # 结果
    print("\n[4/4] 训练完成")
    print(f"  ResNet50 最佳 Top-1: {best_val_acc:.4f} ({best_val_acc:.2%})")
    print(f"  模型已保存到: {save_path}")

    # 保存训练历史
    history_path = MODEL_DIR / "resnet50_breeds" / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return best_val_acc


if __name__ == "__main__":
    main()
