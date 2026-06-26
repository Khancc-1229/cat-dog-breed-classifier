"""
数据收集脚本 —— 百度图片搜索 + 本地整理混合方案

数据来源：
  - Stanford Dogs 数据集（狗品种补充）
  - 百度图片搜索爬取（猫品种 + 狗品种补量）
  - Roboflow 数据集下载（部分品种）
  - 手动筛选整理

使用方法: python scripts/data_collect.py
"""

import sys
import subprocess

# 自动安装 icrawler
subprocess.check_call([sys.executable, "-m", "pip", "install", "icrawler", "-q"])

from icrawler.builtin import BaiduImageCrawler
from pathlib import Path

BASE = Path(__file__).parent.parent / "data" / "raw"
TARGET = 500

BREEDS = {
    # 狗品种 (搜索词针对中文环境优化)
    "哈士奇":   ("狗/哈士奇",   ["哈士奇犬", "西伯利亚哈士奇", "husky 狗"]),
    "柯基":     ("狗/柯基",     ["柯基犬", "彭布罗克柯基", "柯基 短腿狗"]),
    "柴犬":     ("狗/柴犬",     ["柴犬", "日本柴犬", "shiba inu"]),
    "金毛":     ("狗/金毛",     ["金毛寻回犬", "金色寻回犬", "金毛 狗"]),
    "德牧":     ("狗/德牧",     ["德国牧羊犬", "德国黑背", "德牧 警犬"]),
    "萨摩耶":   ("狗/萨摩耶",   ["萨摩耶犬", "萨摩耶 微笑", "samoyed 白狗"]),
    "拉布拉多": ("狗/拉布拉多", ["拉布拉多犬", "拉布拉多寻回犬", "labrador 狗"]),
    "阿拉斯加": ("狗/阿拉斯加", ["阿拉斯加雪橇犬", "阿拉斯加 大型犬", "malamute 狗"]),
    # 猫品种
    "英短蓝猫": ("猫/英短蓝猫", ["英短蓝猫", "英国短毛猫 蓝猫", "british shorthair"]),
    "布偶猫":   ("猫/布偶猫",   ["布偶猫", "布偶猫 蓝眼长毛", "ragdoll cat"]),
    "暹罗猫":   ("猫/暹罗猫",   ["暹罗猫", "暹罗猫 重点色", "siamese cat"]),
    "美短":     ("猫/美短",     ["美国短毛猫", "美短 银虎斑", "american shorthair"]),
    "狸花猫":   ("猫/狸花猫",   ["中国狸花猫", "狸花猫 虎斑", "中华田园狸花猫"]),
    "缅因猫":   ("猫/缅因猫",   ["缅因猫", "缅因猫 大型长毛", "maine coon cat"]),
    "异国短毛猫": ("猫/异国短毛猫", ["异国短毛猫", "加菲猫", "exotic shorthair cat"]),
}


def collect_breed(breed_name, subdir, keywords):
    """用百度和必应图片搜索为一个品种收集图片"""
    save_dir = BASE / subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(save_dir.glob("*.jpg")))
    needed = TARGET - existing

    if needed <= 0:
        print(f"  [{breed_name}] 已有 {existing} >= {TARGET}，跳过")
        return

    print(f"  [{breed_name}] 缺 {needed} 张")

    per_kw = needed // len(keywords) + 30
    for kw in keywords:
        remaining = TARGET - len(list(save_dir.glob("*.jpg")))
        if remaining <= 0:
            break
        print(f"    搜索: \"{kw}\"")
        crawler = BaiduImageCrawler(
            downloader_threads=4,
            storage={"root_dir": str(save_dir)},
        )
        crawler.crawl(
            keyword=kw,
            max_num=min(per_kw, remaining),
            min_size=(300, 300),
            file_idx_offset=len(list(save_dir.glob("*.jpg"))),
        )

    final = len(list(save_dir.glob("*.jpg")))
    print(f"  [{breed_name}] 完成: {existing} → {final}")


def main():
    print("=" * 60)
    print("猫狗品种识别 —— 图片数据收集")
    print(f"目标: {len(BREEDS)} 品种 x {TARGET} = {len(BREEDS) * TARGET} 张")
    print("数据来源: 百度图片搜索 + Stanford Dogs + Roboflow")
    print("=" * 60)

    for breed, (subdir, keywords) in BREEDS.items():
        collect_breed(breed, subdir, keywords)

    # 汇总
    print("\n" + "=" * 60)
    total = 0
    for breed, (subdir, _) in BREEDS.items():
        n = len(list((BASE / subdir).glob("*.jpg")))
        print(f"  {breed}: {n} 张")
        total += n
    print(f"  总计: {total} 张")
    print("=" * 60)


if __name__ == "__main__":
    main()
