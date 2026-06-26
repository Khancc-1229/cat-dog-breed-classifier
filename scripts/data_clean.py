"""
数据清洗脚本 —— 去重、去破损图片、统一格式
使用方法: python scripts/data_clean.py
"""

import os
import hashlib
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import shutil

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DIR = PROJECT_ROOT / "data" / "processed" / "cleaned"

MIN_SIZE = 100  # 图片最小边长（像素），只拦极小缩略图
MIN_FILE_SIZE = 1000  # 最小文件大小（字节），只拦空文件


def get_file_hash(filepath: Path) -> str:
    """计算文件 MD5，用于去重"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def is_valid_image(filepath: Path) -> tuple[bool, str]:
    """
    检查图片是否有效
    返回 (是否有效, 原因)
    """
    # 1. 文件大小
    size = filepath.stat().st_size
    if size < MIN_FILE_SIZE:
        return False, f"文件太小 ({size} bytes)"

    # 2. 能否用 PIL 打开
    try:
        img = Image.open(filepath)
        img.verify()  # 验证图片完整性（不加载像素）
    except (UnidentifiedImageError, Exception):
        return False, "无法打开/图片已损坏"

    # 3. 重新加载检查尺寸（verify 后需要重新打开）
    try:
        img = Image.open(filepath)
        w, h = img.size
        if w < MIN_SIZE or h < MIN_SIZE:
            return False, f"尺寸太小 ({w}x{h})"
    except Exception:
        return False, "无法读取尺寸"

    return True, "ok"


def clean_breed(src_dir: Path, dst_dir: Path):
    """清洗一个品种的所有图片"""
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        image_files.extend(src_dir.glob(ext))

    if not image_files:
        print(f"  [{src_dir.name}] 没有图片，跳过")
        return

    print(f"  [{src_dir.name}] 发现 {len(image_files)} 张图片")

    seen_hashes = set()
    kept = 0
    removed = {"broken": 0, "small": 0, "duplicate": 0}

    for img_path in tqdm(image_files, desc=f"    清洗", leave=False):
        # 检查有效性
        valid, reason = is_valid_image(img_path)
        if not valid:
            if "太小" in reason:
                removed["small"] += 1
            else:
                removed["broken"] += 1
            continue

        # 检查重复
        file_hash = get_file_hash(img_path)
        if file_hash in seen_hashes:
            removed["duplicate"] += 1
            continue
        seen_hashes.add(file_hash)

        # 保存
        try:
            img = Image.open(img_path)
            # 转换为 RGB（处理 RGBA / 灰度图）
            if img.mode != "RGB":
                img = img.convert("RGB")
            save_path = dst_dir / f"{kept:04d}.jpg"
            img.save(save_path, "JPEG", quality=90)
            kept += 1
        except Exception:
            removed["broken"] += 1

    print(f"    保留 {kept} 张 | 破损 {removed['broken']} | "
          f"太小 {removed['small']} | 重复 {removed['duplicate']}")


def main():
    print("=" * 60)
    print("猫狗品种识别 —— 数据清洗")
    print("=" * 60)

    for category in ["狗", "猫"]:
        category_dir = RAW_DIR / category
        if not category_dir.exists():
            continue

        print(f"\n[{category}]")
        for breed_dir in sorted(category_dir.iterdir()):
            if breed_dir.is_dir():
                src = breed_dir
                dst = CLEANED_DIR / category / breed_dir.name
                clean_breed(src, dst)

    # 汇总
    print("\n" + "=" * 60)
    print("清洗后数据集统计：")
    total = 0
    for category in ["狗", "猫"]:
        cat_dir = CLEANED_DIR / category
        if cat_dir.exists():
            for breed_dir in sorted(cat_dir.iterdir()):
                if breed_dir.is_dir():
                    count = len(list(breed_dir.glob("*.jpg")))
                    print(f"  {category}/{breed_dir.name}: {count} 张")
                    total += count
    print(f"  总计: {total} 张")
    print("=" * 60)


if __name__ == "__main__":
    main()
