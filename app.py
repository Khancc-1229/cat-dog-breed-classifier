"""
猫狗品种识别 —— 两阶段级联：检测 + 双模型分类对比
Stage 1: yolov8n.pt 检测图中所有猫/狗位置
Stage 2: YOLOv8-cls / ResNet50 分别识别品种
使用方法: python app.py
"""

import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from ultralytics import YOLO
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import numpy as np

PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# COCO 类别 ID：猫=15, 狗=16
CAT_DOG_IDS = {15, 16}

# 置信度拒识阈值（find_threshold.py 分析）
YOLO_THRESHOLD = 0.92
RESNET_THRESHOLD = 0.94

# ResNet 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ============================================================
#  模型加载
# ============================================================

def load_detector():
    """加载检测模型 yolov8n.pt（COCO 预训练，无需微调）"""
    try:
        return YOLO("yolov8n.pt")
    except Exception:
        print("  ⚠️ yolov8n.pt 下载失败，将自动重试")
        return YOLO("yolov8n.pt")


def load_yolo_classifier():
    """加载 YOLOv8-cls 品种分类模型"""
    model_path = MODEL_DIR / "yolov8_cls_breeds" / "weights" / "best.pt"
    if model_path.exists():
        return YOLO(str(model_path))
    return None


def load_resnet_classifier():
    """加载 ResNet50 品种分类模型"""
    model_path = MODEL_DIR / "resnet50_breeds" / "best.pth"
    if not model_path.exists():
        return None, None

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

    return model, class_names


print("正在加载模型...")
detector = load_detector()
yolo_cls = load_yolo_classifier()
resnet_model, resnet_classes = load_resnet_classifier()

print(f"  检测器 (yolov8n.pt): ✅ 已加载")
print(f"  YOLOv8 分类器:        {'✅ 已加载' if yolo_cls else '❌ 未找到'}")
print(f"  ResNet50 分类器:     {'✅ 已加载' if resnet_model else '❌ 未找到'}")


# ============================================================
#  两阶段预测
# ============================================================

# 框的颜色：狗=蓝色，猫=橙色
DOG_COLOR = (30, 144, 255)    # DodgerBlue
CAT_COLOR = (255, 165, 0)     # Orange
BOX_WIDTH = 3


def detect_animals(image: Image.Image) -> list[dict]:
    """
    Stage 1: 检测图中所有猫/狗
    返回: [{"bbox": (x1,y1,x2,y2), "is_dog": bool, "conf": float, "crop": PIL.Image}, ...]
    """
    results = detector(image, verbose=False)
    detections = []

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id not in CAT_DOG_IDS:
                continue  # 跳过非猫狗目标

            conf = float(boxes.conf[i].item())
            if conf < 0.3:
                continue  # 置信度太低

            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            crop = image.crop((int(x1), int(y1), int(x2), int(y2)))

            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "is_dog": cls_id == 16,
                "det_conf": conf,
                "crop": crop,
            })

    # 去重：如果两个同类别框 IoU > 0.5，保留置信度更高的
    filtered = []
    detections.sort(key=lambda d: d["det_conf"], reverse=True)
    while detections:
        best = detections.pop(0)
        filtered.append(best)
        detections = [d for d in detections
                      if d["is_dog"] != best["is_dog"]
                      or calc_iou(d["bbox"], best["bbox"]) < 0.5]
    return filtered


def calc_iou(box_a, box_b):
    """计算两个框的 IoU"""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


DOG_BREEDS = {"哈士奇", "柯基", "柴犬", "金毛", "德牧", "萨摩耶", "拉布拉多", "阿拉斯加"}


def classify_crop(crop: Image.Image, is_dog: bool = None) -> tuple[dict, dict]:
    """
    Stage 2: 双模型品种分类（仅从狗/猫子集中选择）
    """
    # YOLOv8 分类
    if yolo_cls:
        r = yolo_cls(crop, verbose=False)
        probs = r[0].probs.data.cpu().numpy()
        names = r[0].names

        # 已知狗/猫时，屏蔽不相关类别
        if is_dog is not None:
            for i, name in names.items():
                if is_dog and name not in DOG_BREEDS:
                    probs[i] = 0
                elif not is_dog and name in DOG_BREEDS:
                    probs[i] = 0
            if probs.sum() > 0:
                probs = probs / probs.sum()

        conf = float(probs.max())
        yolo_result = {
            "top1_name": names[int(probs.argmax())],
            "top1_conf": conf,
            "rejected": conf < YOLO_THRESHOLD,
            "top5": [(names[int(i)], float(probs[int(i)]))
                     for i in np.argsort(probs)[-5:][::-1] if probs[int(i)] > 0],
        }
    else:
        yolo_result = None

    # ResNet50 分类
    if resnet_model:
        img_tensor = transform(crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = resnet_model(img_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

        # 已知狗/猫时，屏蔽不相关类别
        if is_dog is not None:
            for i, name in enumerate(resnet_classes):
                if is_dog and name not in DOG_BREEDS:
                    probs[i] = 0
                elif not is_dog and name in DOG_BREEDS:
                    probs[i] = 0
            if probs.sum() > 0:
                probs = probs / probs.sum()

        top5_idx = np.argsort(probs)[-5:][::-1]
        conf = float(probs[top5_idx[0]])
        resnet_result = {
            "top1_name": resnet_classes[int(top5_idx[0])],
            "top1_conf": conf,
            "rejected": conf < RESNET_THRESHOLD,
            "top5": [(resnet_classes[int(i)], float(probs[i]))
                     for i in top5_idx if probs[int(i)] > 0],
        }
    else:
        resnet_result = None

    return yolo_result, resnet_result


def draw_boxes(image: Image.Image, detections: list[dict],
               yolo_results: list, resnet_results: list) -> Image.Image:
    """在图片上画框 + 品种标签"""
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)

    # 尝试加载中文字体
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
        font_sm = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
        font_sm = ImageFont.load_default()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        color = DOG_COLOR if det["is_dog"] else CAT_COLOR
        icon = "🐶" if det["is_dog"] else "🐱"

        # 画框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_WIDTH)

        # 标签文本
        yolo_r = yolo_results[i] if i < len(yolo_results) else None
        resnet_r = resnet_results[i] if i < len(resnet_results) else None

        lines = [f"{icon} {'狗' if det['is_dog'] else '猫'}"]
        if yolo_r:
            if yolo_r["rejected"]:
                lines.append(f"YOLOv8: ? {yolo_r['top1_name']} {yolo_r['top1_conf']:.0%}")
            else:
                lines.append(f"YOLOv8: {yolo_r['top1_name']} {yolo_r['top1_conf']:.0%}")
        if resnet_r:
            if resnet_r["rejected"]:
                lines.append(f"ResNet: ? {resnet_r['top1_name']} {resnet_r['top1_conf']:.0%}")
            else:
                lines.append(f"ResNet: {resnet_r['top1_name']} {resnet_r['top1_conf']:.0%}")

        # 标签背景
        label_h = 22 * len(lines) + 8
        label_w = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font_sm)
            label_w = max(label_w, bbox[2] - bbox[0] + 12)

        # 标签放框上方（如果框太靠近顶部则放框内）
        label_y = y1 - label_h
        if label_y < 0:
            label_y = y1

        draw.rectangle(
            [x1, label_y, x1 + label_w, label_y + label_h],
            fill=(*color, 200)
        )

        for j, line in enumerate(lines):
            draw.text(
                (x1 + 6, label_y + 4 + j * 22),
                line,
                fill="white",
                font=font_sm,
            )

    return draw_img


def predict(image):
    """完整两阶段推理"""
    if image is None:
        return None, "*请上传一张图片*", "*等待检测结果...*"

    # Stage 1: 检测
    detections = detect_animals(image)

    if not detections:
        # 没检测到猫狗，回退为整图分类
        full_crop_result = classify_crop(image)
        yolo_r, resnet_r = full_crop_result
        table = build_comparison_table([], [yolo_r], [resnet_r])
        return image, (
            "### ⚠️ 未检测到猫/狗\n\n"
            "可能原因：图片中没有猫狗、目标太小、或被遮挡。\n\n"
            "已对整张图片进行分类作为参考：\n"
        ), table

    # Stage 2: 逐个分类
    yolo_results = []
    resnet_results = []
    for det in detections:
        y_r, r_r = classify_crop(det["crop"], det["is_dog"])
        yolo_results.append(y_r)
        resnet_results.append(r_r)

    # 画框
    annotated = draw_boxes(image, detections, yolo_results, resnet_results)

    # 生成文本报告
    text = build_detection_text(detections, yolo_results, resnet_results)

    # 生成对比表
    table = build_comparison_table(detections, yolo_results, resnet_results)

    return annotated, text, table


def build_detection_text(detections, yolo_results, resnet_results) -> str:
    """生成检测详情文本"""
    if not detections:
        return "*无检测结果*"

    text = f"### 📊 检测到 {len(detections)} 个目标\n\n"

    for i, det in enumerate(detections):
        icon = "🐶" if det["is_dog"] else "🐱"
        kind = "狗" if det["is_dog"] else "猫"

        # 裁剪图转 base64 小缩略图
        import base64, io
        thumb = det["crop"].copy()
        thumb.thumbnail((80, 80))
        buf = io.BytesIO()
        thumb.save(buf, "JPEG", quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode()

        text += (
            f"<img src='data:image/jpeg;base64,{b64}' "
            f"style='width:80px;height:80px;border-radius:10px;"
            f"border:3px solid {'#3b82f6' if det['is_dog'] else '#f97316'};"
            f"float:left;margin-right:12px;object-fit:cover;'>"
        )
        text += f"**目标 {i+1}**: {icon} {kind} ({det['det_conf']:.0%})\n\n"
        text += '<br clear="both">'

        y_r = yolo_results[i] if i < len(yolo_results) else None
        r_r = resnet_results[i] if i < len(resnet_results) else None

        text += "| 模型 | 预测品种 | 置信度 |\n"
        text += "|------|----------|--------|\n"
        if y_r:
            name = (f"? {y_r['top1_name']}" if y_r["rejected"] else y_r['top1_name'])
            conf = f"{y_r['top1_conf']:.0%}"
            text += f"| 🔵 YOLOv8-cls | {name} | {conf} {'⚠️' if y_r['rejected'] else ''}|\n"
        if r_r:
            name = (f"? {r_r['top1_name']}" if r_r["rejected"] else r_r['top1_name'])
            conf = f"{r_r['top1_conf']:.0%}"
            text += f"| 🟠 ResNet50   | {name} | {conf} {'⚠️' if r_r['rejected'] else ''}|\n"

        # 一致性
        if y_r and r_r:
            if y_r["rejected"] and r_r["rejected"]:
                text += "\n⚠️ 两个模型都认为**可能不在识别范围内**\n"
            elif y_r["rejected"] or r_r["rejected"]:
                text += "\n⚠️ 一个模型认为可能不在识别范围内\n"
            elif y_r["top1_name"] == r_r["top1_name"]:
                text += f"\n✅ 一致 —— 都认为是 **{y_r['top1_name']}**\n"
            else:
                text += (
                    f"\n⚠️ 分歧：YOLOv8 → {y_r['top1_name']}，"
                    f"ResNet50 → {r_r['top1_name']}\n"
                )

        text += "\n---\n\n"

    return text


def build_comparison_table(detections, yolo_results, resnet_results) -> str:
    """生成双模型对比汇总表"""
    if not detections:
        return "*无数据*"

    table = "### 🆚 双模型对比汇总\n\n"
    table += "| 目标 | YOLOv8 预测 | YOLOv8 置信度 | ResNet50 预测 | ResNet50 置信度 | 是否一致 |\n"
    table += "|------|-------------|---------------|---------------|-----------------|----------|\n"

    for i, det in enumerate(detections):
        icon = "🐶" if det["is_dog"] else "🐱"
        y_r = yolo_results[i] if i < len(yolo_results) else None
        r_r = resnet_results[i] if i < len(resnet_results) else None

        y_name = (f"可能不在范围内 ({y_r['top1_name']})" if (y_r and y_r["rejected"])
                  else (y_r["top1_name"] if y_r else "—"))
        y_conf = f"{y_r['top1_conf']:.0%}" if y_r else "—"
        r_name = (f"可能不在范围内 ({r_r['top1_name']})" if (r_r and r_r["rejected"])
                  else (r_r["top1_name"] if r_r else "—"))
        r_conf = f"{r_r['top1_conf']:.0%}" if r_r else "—"

        agree = ""
        if y_r and r_r:
            if y_r["rejected"] and r_r["rejected"]:
                agree = "✅ 均拒识"
            elif y_r["rejected"] or r_r["rejected"]:
                agree = "⚠️ 一方拒识"
            elif y_r["top1_name"] == r_r["top1_name"]:
                agree = "✅"
            else:
                agree = "⚠️"

        table += f"| {icon}{i+1} | {y_name} | {y_conf} | {r_name} | {r_conf} | {agree} |\n"

    return table


# ============================================================
#  Gradio 界面
# ============================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}
.header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
}
.header h1 { font-size: 2.2rem; margin-bottom: 0.3rem; font-weight: 700; }
.header p { font-size: 1rem; opacity: 0.9; }
.upload-card, .result-card {
    background: white;
    border-radius: 14px;
    padding: 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    transition: box-shadow 0.3s;
}
.upload-card:hover, .result-card:hover {
    box-shadow: 0 6px 32px rgba(0,0,0,0.12);
}
.upload-card label, .result-card label {
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    color: #333 !important;
}
.info-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.5rem;
}
.info-col {
    flex: 1;
    background: white;
    border-radius: 14px;
    padding: 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    min-height: 180px;
}
.info-col h3 {
    margin-top: 0;
    font-size: 1.1rem;
    color: #555;
    border-bottom: 2px solid #667eea;
    padding-bottom: 0.5rem;
    margin-bottom: 0.8rem;
}
.info-col table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
}
.info-col th {
    background: #f5f3ff;
    color: #555;
    padding: 0.5rem 0.4rem;
    text-align: left;
    font-weight: 600;
}
.info-col td {
    padding: 0.5rem 0.4rem;
    border-bottom: 1px solid #f0f0f0;
}
.footer {
    margin-top: 1.5rem;
    padding: 1rem;
    background: white;
    border-radius: 14px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    font-size: 0.9rem;
    color: #888;
    line-height: 1.8;
}
.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.badge-blue { background: #dbeafe; color: #1e40af; }
.badge-orange { background: #ffedd5; color: #9a3412; }
"""

with gr.Blocks(
    title="猫狗品种识别",
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
) as demo:
    gr.HTML("""
    <div class="header">
        <h1>🐱🐶 猫狗品种识别系统</h1>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_classes="upload-card"):
            gr.HTML('<div style="font-weight:600;font-size:1.05rem;color:#333;margin-bottom:0.5rem;">📤 上传图片</div>')
            inp = gr.Image(type="pil", label="", show_label=False, height=380)

        with gr.Column(scale=1, elem_classes="result-card"):
            gr.HTML('<div style="font-weight:600;font-size:1.05rem;color:#333;margin-bottom:0.5rem;">📸 检测结果</div>')
            out_img = gr.Image(type="pil", label="", show_label=False, height=380)

    with gr.Row(elem_classes="info-row"):
        with gr.Column(scale=1, elem_classes="info-col"):
            gr.HTML('<h3>📋 检测详情</h3>')
            detail_output = gr.Markdown("*等待上传图片...*")
        with gr.Column(scale=1, elem_classes="info-col"):
            gr.HTML('<h3>🆚 双模型对比</h3>')
            compare_output = gr.Markdown("*等待上传图片...*")

    inp.change(
        fn=predict,
        inputs=inp,
        outputs=[out_img, detail_output, compare_output],
    )

    gr.HTML("""
    <div class="footer">
        <b>📋 品种</b>：哈士奇 · 柯基 · 柴犬 · 金毛 · 德牧 · 萨摩耶 · 拉布拉多 · 阿拉斯加
        ｜ 英短蓝猫 · 布偶猫 · 暹罗猫 · 美短 · 狸花猫 · 缅因猫 · 异国短毛猫
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
