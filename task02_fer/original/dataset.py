# 人脸情感分类数据加载：扫描、分层五折、预处理、类别权重、Dataset/DataLoader
from pathlib import Path
from collections import Counter
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

# 数据路径（数据集在桌面上）
DATA_DIR = Path(r"C:\Users\Lenovo\Desktop\fer_data\fer_data")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# CLAHE 缓存目录：离线一次性增强后存这里，训练直接读，避免每个 epoch 重复计算
CACHE_DIR = Path(__file__).parent / "cache_clahe"

# 情感类别 -> 数字标签（提交时 Emotion 用这个数字）
CLASS_TO_IDX = {
    "Angry": 0,
    "Fear": 1,
    "Happy": 2,
    "Sad": 3,
    "Surprise": 4,
    "Neutral": 5,
}

# 扫描训练集：每张图一条记录
records = []
for class_dir in sorted(TRAIN_DIR.iterdir()):
    if not class_dir.is_dir():
        continue
    class_name = class_dir.name
    class_idx = CLASS_TO_IDX[class_name]
    for img_path in sorted(class_dir.glob("*.jpg")):
        records.append({
            "path": str(img_path),
            "class_idx": class_idx,
            "filename": img_path.name,
            "cache_path": str(CACHE_DIR / str(class_idx) / img_path.name),
        })

print("训练集图片总数:", len(records))

# 分层五折划分（固定种子，可复现），净化数据后需要重新调用
def assign_folds(recs, n_splits=5, seed=2026):
    y = [r["class_idx"] for r in recs]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(recs, y)):
        for i in val_idx:
            recs[i]["fold"] = fold
    return recs


records = assign_folds(records)

# 类别权重：样本少的类别给更高权重，缓解类不平衡
class_counts = Counter(r["class_idx"] for r in records)
total = len(records)
num_classes = len(CLASS_TO_IDX)
class_weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)]

# 预处理：48x48 灰度 -> RGB -> 直接 Resize 到 224 -> 归一化
# 注意：不用 RandomResizedCrop，因为 48x48 太小，随机裁剪会切掉关键表情细节
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),  # 人脸左右对称，水平翻转安全
    transforms.RandAugment(num_ops=2, magnitude=9),  # 自动增强（2026 论文：显著提升 FER 准确率）
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # CutOut：随机抠块
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 验证/测试只做基础处理
val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def enhance(image):
    """CLAHE 对比度增强：局部自适应，比全局直方图均衡化更适合人脸表情"""
    img_np = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_np)
    return Image.fromarray(enhanced)


class FerDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        image = Image.open(r["cache_path"])  # 直接读缓存图（已 CLAHE + RGB）
        if self.transform is not None:
            image = self.transform(image)
        return image, r["class_idx"], r["filename"]  # 额外返回文件名，用于记录 OOF 预测


def split_by_fold(records, val_fold):
    train_records = [r for r in records if r["fold"] != val_fold]
    val_records = [r for r in records if r["fold"] == val_fold]
    return train_records, val_records


def get_dataloaders(records, val_fold, batch_size=64):
    train_records, val_records = split_by_fold(records, val_fold)
    train_dataset = FerDataset(train_records, train_transform)
    val_dataset = FerDataset(val_records, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=8, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=8, persistent_workers=True, pin_memory=True)
    return train_loader, val_loader
