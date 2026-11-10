# 离线 CLAHE 缓存：把训练集一次性做 CLAHE 增强 + 转 RGB，存成缓存图。
# 之后训练直接读缓存，省掉每个 epoch 重复的 CLAHE 计算（最大的 CPU 瓶颈）。
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from dataset import records

CACHE_DIR = Path(__file__).parent / "cache_clahe"


def main():
    total = len(records)
    done = 0
    skipped = 0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # CLAHE 对象只创建一次

    for i, r in enumerate(records):
        dst = Path(r["cache_path"])
        if dst.exists():
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(r["path"])                 # 读灰度图（L 模式）
        arr = clahe.apply(np.array(img))            # CLAHE 对比度增强
        img = Image.fromarray(arr).convert("RGB")   # 转 RGB 三通道
        img.save(dst)
        done += 1
        if (i + 1) % 2000 == 0:
            print(f"已处理 {i + 1}/{total}")

    print(f"缓存完成：新增 {done} 张，已存在跳过 {skipped} 张，共 {total} 张")


if __name__ == "__main__":
    main()
