"""
置信度阈值分析 —— 区分"在册品种"和"不在册品种"
1. 对验证集（在册 15 品种）逐张预测，记录最高置信度
2. 对 unknown_test（不在册品种）逐张预测，记录最高置信度
3. 画分布图，找最佳阈值
"""
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from ultralytics import YOLO
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体
font_path = "C:/Windows/Fonts/msyh.ttc"
font_manager.fontManager.addfont(font_path)
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
UNKNOWN_DIR = PROJECT_ROOT / "data" / "unknown_test"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_yolo_confidences(image_paths):
    """YOLOv8 对每张图的最高置信度"""
    model_path = MODEL_DIR / "yolov8_cls_breeds" / "weights" / "best.pt"
    model = YOLO(str(model_path))
    confs = []
    for p in image_paths:
        r = model(p, verbose=False)
        confs.append(float(r[0].probs.data.max()))
    return confs


def get_resnet_confidences(image_paths):
    """ResNet50 对每张图的最高置信度"""
    ckpt = torch.load(MODEL_DIR / "resnet50_breeds" / "best.pth", map_location=DEVICE)
    class_names = ckpt["class_names"]
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(in_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, len(class_names)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    confs = []
    for p in image_paths:
        img = datasets.folder.default_loader(p)
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(model(img_t), dim=1).cpu().numpy()[0]
        confs.append(float(probs.max()))
    return confs


def main():
    print("置信度阈值分析")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. 在册品种的置信度（用 val 集）
    val_dir = PROJECT_ROOT / "data" / "processed" / "dataset" / "val"
    in_images = list(val_dir.rglob("*.jpg"))
    print(f"\n在册图片 (val): {len(in_images)} 张")

    # 2. 不在册品种的置信度
    out_images = list(UNKNOWN_DIR.rglob("*.jpg")) + list(UNKNOWN_DIR.rglob("*.png")) + list(UNKNOWN_DIR.rglob("*.jpeg"))
    print(f"不在册图片: {len(out_images)} 张")

    # 3. 两个模型分别预测
    for model_name, get_confs in [
        ("YOLOv8-cls", get_yolo_confidences),
        ("ResNet50", get_resnet_confidences),
    ]:
        print(f"\n[{model_name}] 正在预测...")
        in_confs = get_confs(in_images)
        out_confs = get_confs(out_images)

        in_mean, in_std = np.mean(in_confs), np.std(in_confs)
        out_mean, out_std = np.mean(out_confs), np.std(out_confs)
        print(f"  在册: mean={in_mean:.3f}, std={in_std:.3f}")
        print(f"  不在册: mean={out_mean:.3f}, std={out_std:.3f}")

        # 打印详细阈值表
        print(f"\n  详细阈值表（0.01步长）:")
        print(f"  {'阈值':<8} {'在册误拒':<12} {'不在册误收':<12} {'总误差':<10}")
        for t in np.arange(0.80, 0.97, 0.01):
            fr = np.mean(in_confs < t)
            fa = np.mean(out_confs >= t)
            print(f"  {t:.2f}    {fr:>8.1%}      {fa:>8.1%}       {fr+fa:>8.1%}")

        # 找最佳阈值（使两类错分率之和最小）
        best_thresh, best_error = 0, 1.0
        for t in np.arange(0.3, 0.95, 0.01):
            false_reject = np.mean(in_confs < t)   # 在册被拒绝
            false_accept = np.mean(out_confs >= t)  # 不在册被接受
            error = false_reject + false_accept
            if error < best_error:
                best_error = error
                best_thresh = t

        print(f"  推荐阈值: {best_thresh:.2f}")
        print(f"    在册被误拒: {np.mean(in_confs < best_thresh):.1%}")
        print(f"    不在册被误收: {np.mean(out_confs >= best_thresh):.1%}")

        # 画图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(in_confs, bins=30, alpha=0.6, label=f"在册品种 (n={len(in_confs)})", color="green", edgecolor="white")
        ax.hist(out_confs, bins=30, alpha=0.6, label=f"不在册品种 (n={len(out_confs)})", color="red", edgecolor="white")
        ax.axvline(best_thresh, color="black", linestyle="--", linewidth=2,
                   label=f"推荐阈值 = {best_thresh:.2f}")
        ax.set_xlabel("最高置信度", fontsize=13)
        ax.set_ylabel("图片数量", fontsize=13)
        ax.set_title(f"{model_name} 置信度分布（在册 vs 不在册）", fontsize=15, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"threshold_{model_name}.png", dpi=150)
        plt.close()
        print(f"  分布图已保存: outputs/threshold_{model_name}.png")


if __name__ == "__main__":
    main()
