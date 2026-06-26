"""
数据集划分脚本 —— 按品种划分 train/val (80/20)，保持各类别比例一致
使用方法: python scripts/split_data.py
"""

from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(__file__).parent.parent
CLEANED_DIR = PROJECT_ROOT / "data" / "processed" / "cleaned"
DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "dataset"

TRAIN_RATIO = 0.8   # 80% 训练
VAL_RATIO = 0.1     # 10% 验证
TEST_RATIO = 0.1    # 10% 测试
RANDOM_SEED = 42


def split_breed(breed_dir: Path, train_dir: Path, val_dir: Path, test_dir: Path):
    """划分一个品种的图片到 train/val/test (80/10/10)"""
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(breed_dir.glob("*.jpg"))
    if not images:
        print(f"  [{breed_dir.parent.name}/{breed_dir.name}] 没有图片，跳过")
        return

    # 打乱
    random.seed(RANDOM_SEED)
    random.shuffle(images)

    # 划分
    n = len(images)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    # 复制
    for img in train_imgs:
        shutil.copy2(img, train_dir / img.name)
    for img in val_imgs:
        shutil.copy2(img, val_dir / img.name)
    for img in test_imgs:
        shutil.copy2(img, test_dir / img.name)

    print(f"  [{breed_dir.parent.name}/{breed_dir.name}] "
          f"train: {len(train_imgs)} | val: {len(val_imgs)} | test: {len(test_imgs)}")


def main():
    print("=" * 60)
    print("猫狗品种识别 —— 数据集划分 (80/10/10)")
    print("=" * 60)

    # 清空旧数据集
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    total_train, total_val, total_test = 0, 0, 0

    for category in ["狗", "猫"]:
        cat_cleaned = CLEANED_DIR / category
        if not cat_cleaned.exists():
            continue

        print(f"\n[{category}]")
        for breed_dir in sorted(cat_cleaned.iterdir()):
            if breed_dir.is_dir():
                # 品种直接放在 train/val/test 下（ImageFolder 要求）
                train_dst = DATASET_DIR / "train" / breed_dir.name
                val_dst = DATASET_DIR / "val" / breed_dir.name
                test_dst = DATASET_DIR / "test" / breed_dir.name
                split_breed(breed_dir, train_dst, val_dst, test_dst)

                total_train += len(list(train_dst.glob("*.jpg")))
                total_val += len(list(val_dst.glob("*.jpg")))
                total_test += len(list(test_dst.glob("*.jpg")))

    print("\n" + "=" * 60)
    print(f"train: {total_train} | val: {total_val} | test: {total_test}")
    print(f"总计: {total_train + total_val + total_test} 张")
    print("=" * 60)


if __name__ == "__main__":
    main()
