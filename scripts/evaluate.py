"""
模型对比评估脚本 —— 两个模型的准确率、混淆矩阵、Top-k 对比
使用方法: python scripts/evaluate.py
"""

import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from ultralytics import YOLO
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
font_path = "C:/Windows/Fonts/msyh.ttc"
font_manager.fontManager.addfont(font_path)
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_data():
    """加载验证集"""
    val_dataset = datasets.ImageFolder(
        str(DATASET_DIR / "val"), transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)
    return val_dataset, val_loader


def build_class_path_map(val_dataset):
    """构建 狗/猫/品种 的层级标签名"""
    class_names = val_dataset.classes
    # 从目录结构中提取父类别
    # dataset 目录结构: train/狗/哈士奇, val/猫/英短蓝猫
    # ImageFolder 的 classes 是叶子目录名，我们需要映射父目录
    breed_to_category = {}
    all_samples = list(val_dataset.samples)
    for path_str, label_idx in all_samples:
        breed_name = class_names[label_idx]
        # path 格式: .../val/猫/英短蓝猫/0001.jpg
        parts = Path(path_str).parts
        if "猫" in parts:
            breed_to_category[breed_name] = "猫"
        elif "狗" in parts:
            breed_to_category[breed_name] = "狗"
        else:
            breed_to_category[breed_name] = "未知"

    return class_names, breed_to_category


def evaluate_yolov8(val_dataset):
    """用 YOLOv8-cls 评估"""
    print("\n[YOLOv8-cls] 正在评估...")
    model_path = list((MODEL_DIR / "yolov8_cls_breeds" / "weights").glob("best.pt"))
    if not model_path:
        print("  ❌ 未找到 YOLOv8 模型！请先运行 train_yolov8.py")
        return None
    model = YOLO(str(model_path[0]))
    class_names = val_dataset.classes

    all_preds, all_labels, all_probs = [], [], []

    for idx in tqdm(range(len(val_dataset)), desc="    YOLOv8推理"):
        img_path, label = val_dataset.samples[idx]
        result = model(img_path, verbose=False)
        probs = result[0].probs.data.cpu().numpy()
        yolo_names = result[0].names  # {0: 'name', ...}

        # 对齐 YOLO 类别顺序和 dataset 顺序
        remapped = np.zeros(len(class_names))
        for yi, name in yolo_names.items():
            if name in class_names:
                remapped[class_names.index(name)] = probs[yi]

        all_preds.append(int(remapped.argmax()))
        all_labels.append(label)
        all_probs.append(remapped)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    top1 = (all_preds == all_labels).mean()
    top5 = sum(1 for i in range(len(all_labels))
               if all_labels[i] in np.argsort(all_probs[i])[-5:]) / len(all_labels)

    print(f"  Top-1: {top1:.4f} | Top-5: {top5:.4f}")

    return {
        "name": "YOLOv8-cls",
        "top1": top1,
        "top5": top5,
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
        "class_names": class_names,
    }


def evaluate_resnet(val_dataset, val_loader):
    """用 ResNet50 评估"""
    print("\n[ResNet50] 正在评估...")
    model_path = MODEL_DIR / "resnet50_breeds" / "best.pth"
    if not model_path.exists():
        print("  ❌ 未找到 ResNet50 模型！请先运行 train_resnet.py")
        return None

    # 加载模型
    checkpoint = torch.load(model_path, map_location=DEVICE)
    class_names = checkpoint["class_names"]

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(in_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, len(class_names)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="    批量推理"):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(probs.argmax(axis=1))
            all_labels.extend(labels.numpy())
            all_probs.append(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs)

    correct = (all_preds == all_labels).sum()
    total = len(all_labels)
    top1 = correct / total

    # Top-5 计算
    top5_correct = 0
    for i in range(total):
        top5_idx = np.argsort(all_probs[i])[-5:]
        if all_labels[i] in top5_idx:
            top5_correct += 1
    top5 = top5_correct / total

    print(f"  Top-1: {top1:.4f} | Top-5: {top5:.4f}")

    return {
        "name": "ResNet50",
        "top1": top1,
        "top5": top5,
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
        "class_names": class_names,
    }


def plot_confusion_matrix(result, breed_to_category, save_dir: Path):
    """绘制混淆矩阵"""
    class_names = result["class_names"]
    cm = confusion_matrix(result["labels"], result["preds"],
                          labels=range(len(class_names)))

    # 归一化
    cm_norm = cm.astype("float32") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    # 用 狗🐶品种名 / 猫🐱品种名 作为标签
    display_names = []
    for name in class_names:
        cat = breed_to_category.get(name, "")
        emoji = "🐶" if cat == "狗" else "🐱"
        if len(name) > 5:
            name = name[:5]  # 缩短中文标签
        display_names.append(f"{emoji}{name}")

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=display_names, yticklabels=display_names,
                vmin=0, vmax=1, ax=ax, linewidths=0.5,
                cbar_kws={"label": "准确率"})
    ax.set_title(f"{result['name']} 混淆矩阵 (归一化)", fontsize=16,
                 fontweight="bold")
    ax.set_xlabel("预测类别", fontsize=13)
    ax.set_ylabel("真实类别", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    save_path = save_dir / f"confusion_matrix_{result['name']}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  混淆矩阵已保存: {save_path}")

    # 找出最容易混淆的对子
    print(f"\n  [{result['name']}] Top-5 最易混淆对:")
    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 0:
                errors.append((i, j, cm[i][j], cm[i][j] / cm[i].sum()))
    errors.sort(key=lambda x: x[2], reverse=True)
    for true_i, pred_j, count, rate in errors[:5]:
        print(f"    {class_names[true_i]} → {class_names[pred_j]}: "
              f"{count}次 ({rate:.1%})")


def plot_accuracy_comparison(yolo_result, resnet_result, save_dir: Path):
    """准确率对比柱状图"""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = []
    top1_vals = []
    top5_vals = []

    for r in [yolo_result, resnet_result]:
        if r:
            models.append(r["name"])
            top1_vals.append(r["top1"] * 100)
            top5_vals.append(r["top5"] * 100)

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width / 2, top1_vals, width, label="Top-1",
                   color=["#1f77b4", "#ff7f0e"])
    bars2 = ax.bar(x + width / 2, top5_vals, width, label="Top-5",
                   color=["#4fc3f7", "#ffcc80"])

    # 数值标注
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.5,
                f"{h:.1f}%", ha="center", fontsize=12, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.5,
                f"{h:.1f}%", ha="center", fontsize=11)

    ax.set_ylabel("准确率 (%)", fontsize=13)
    ax.set_title("模型对比：YOLOv8-cls vs ResNet50", fontsize=15,
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=13)
    ax.legend(fontsize=12)
    ax.set_ylim(0, max(top5_vals) + 10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "accuracy_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  对比图已保存: {save_path}")


def export_report(yolo_result, resnet_result, breed_to_category, save_dir: Path):
    """导出文本报告"""
    report_path = save_dir / "evaluation_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("猫狗品种识别 —— 模型评估报告\n")
        f.write("=" * 60 + "\n\n")

        for r in [yolo_result, resnet_result]:
            if not r:
                continue
            f.write(f"## {r['name']}\n")
            f.write(f"  Top-1 Accuracy: {r['top1']:.4f} ({r['top1']:.2%})\n")
            f.write(f"  Top-5 Accuracy: {r['top5']:.4f} ({r['top5']:.2%})\n\n")

            # 每类准确率
            class_names = r["class_names"]
            f.write("  各类别准确率:\n")
            for i, name in enumerate(class_names):
                mask = r["labels"] == i
                cls_acc = (r["preds"][mask] == i).sum() / mask.sum()
                cat = breed_to_category.get(name, "")
                f.write(f"    {cat} {name}: {cls_acc:.2%}\n")
            f.write("\n")

            # 分类报告
            f.write("  分类报告 (sklearn):\n")
            report = classification_report(
                r["labels"], r["preds"],
                target_names=class_names,
                zero_division=0
            )
            f.write(report)
            f.write("\n\n")

        # 对比
        if yolo_result and resnet_result:
            f.write("-" * 40 + "\n")
            f.write("## 模型对比\n")
            f.write(f"  YOLOv8-cls Top-1: {yolo_result['top1']:.2%}\n")
            f.write(f"  ResNet50   Top-1: {resnet_result['top1']:.2%}\n")

            diff_top1 = yolo_result["top1"] - resnet_result["top1"]
            winner = "YOLOv8-cls" if diff_top1 > 0 else "ResNet50"
            f.write(f"  差距: {abs(diff_top1):.2%}\n")
            f.write(f"  胜出: {winner}\n\n")

            yolo_wins = 0
            resnet_wins = 0
            for i, name in enumerate(yolo_result["class_names"]):
                y_mask = yolo_result["labels"] == i
                r_mask = resnet_result["labels"] == i
                y_acc = (yolo_result["preds"][y_mask] == i).sum() / y_mask.sum()
                r_acc = (resnet_result["preds"][r_mask] == i).sum() / r_mask.sum()
                if y_acc > r_acc:
                    yolo_wins += 1
                elif r_acc > y_acc:
                    resnet_wins += 1

            f.write(f"  YOLOv8 各类别胜出数: {yolo_wins}/{len(class_names)}\n")
            f.write(f"  ResNet50 各类别胜出数: {resnet_wins}/{len(class_names)}\n")

    print(f"  评估报告已保存: {report_path}")


def main():
    print("=" * 60)
    print("猫狗品种识别 —— 模型对比评估")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] 加载验证集...")
    val_dataset, val_loader = load_data()
    class_names, breed_to_category = build_class_path_map(val_dataset)
    print(f"  验证集: {len(val_dataset)} 张, {len(class_names)} 类")

    print("\n[2/3] 评估两个模型...")
    yolo_result = evaluate_yolov8(val_dataset)
    resnet_result = evaluate_resnet(val_dataset, val_loader)

    print("\n[3/3] 生成可视化 & 报告...")

    for r in [yolo_result, resnet_result]:
        if r:
            plot_confusion_matrix(r, breed_to_category, OUTPUT_DIR)

    plot_accuracy_comparison(yolo_result, resnet_result, OUTPUT_DIR)
    export_report(yolo_result, resnet_result, breed_to_category, OUTPUT_DIR)

    # 终端汇总
    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    for r in [yolo_result, resnet_result]:
        if r:
            print(f"\n  {r['name']}:")
            print(f"    Top-1: {r['top1']:.2%}")
            print(f"    Top-5: {r['top5']:.2%}")

    if yolo_result and resnet_result:
        diff = yolo_result["top1"] - resnet_result["top1"]
        winner = "YOLOv8-cls" if diff > 0 else "ResNet50"
        print(f"\n  🏆 胜出: {winner} (差距 {abs(diff):.2%})")

    print(f"\n所有输出文件位于: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
