"""
文件名标准化 —— 统一命名为 BreedName_001.jpg 格式，转JPG
"""
from pathlib import Path
from PIL import Image
import os

BASE = Path("d:/Dogs and Cats/data/raw")

# 中文 → 英文名映射
NAME_MAP = {
    "哈士奇":   "Husky",
    "柯基":     "Corgi",
    "柴犬":     "Shiba",
    "金毛":     "Golden_Retriever",
    "德牧":     "German_Shepherd",
    "萨摩耶":   "Samoyed",
    "拉布拉多": "Labrador",
    "阿拉斯加": "Malamute",
    "英短蓝猫": "British_Shorthair",
    "布偶猫":   "Ragdoll",
    "暹罗猫":   "Siamese",
    "美短":     "American_Shorthair",
    "狸花猫":   "Chinese_Tabby",
    "缅因猫":   "Maine_Coon",
    "异国短毛猫": "Exotic_Shorthair",
}

for category in ["狗", "猫"]:
    cat_dir = BASE / category
    if not cat_dir.exists():
        continue
    for breed_dir in sorted(cat_dir.iterdir()):
        if not breed_dir.is_dir():
            continue
        cn_name = breed_dir.name
        en_name = NAME_MAP.get(cn_name)
        if not en_name:
            print(f"⚠️ 未知品种: {cn_name}")
            continue

        # 收集所有图片文件
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.gif"):
            imgs.extend(breed_dir.glob(ext))
        # 按文件名排序保证顺序稳定
        imgs = sorted(imgs)

        if not imgs:
            print(f"{category}/{cn_name}: 无图片, 跳过")
            continue

        print(f"{category}/{cn_name} → {en_name}: {len(imgs)} 张")

        for i, old_path in enumerate(imgs, 1):
            new_path = breed_dir / f"{en_name}_{i:04d}.jpg"

            try:
                img = Image.open(old_path)
                if img.mode in ("RGBA", "P", "L"):
                    img = img.convert("RGB")

                # 如果新路径和旧路径不同，先保存新的再删旧的
                if old_path != new_path:
                    img.save(new_path, "JPEG", quality=92)
                    old_path.unlink()
                else:
                    # 同路径但可能是别的格式→覆盖为JPG
                    img.save(new_path, "JPEG", quality=92)

            except Exception as e:
                print(f"  [ERR] {old_path.name}: {e}")
                continue

        # 清理残留的非jpg文件
        for f in breed_dir.iterdir():
            if f.suffix.lower() != ".jpg":
                f.unlink()

        print(f"  [OK] {len(imgs)} 张已标准化")

print("完成")
