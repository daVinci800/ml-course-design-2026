# 人脸情感分类预测：5 折模型集成 + TTA（测试时增强），生成 submission.csv
import csv
import glob
import tempfile
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from model import create_model
from dataset import TEST_DIR, num_classes, enhance

# 模型与 submission 都放临时目录（与 train.py 一致，沙盒仅允许写这里）
SAVE_DIR = Path(tempfile.gettempdir()) / "fer_models"

# 测试集预处理（和验证集一样，不做增强）
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载所有折 × 种子的最佳模型（多种子集成）
    model_files = sorted(glob.glob(str(SAVE_DIR / "best_model_fold*_seed*.pth")))
    models = []
    for f in model_files:
        model = create_model(num_classes=num_classes)
        model.load_state_dict(torch.load(f, map_location=device))
        model = model.to(device)
        model.eval()
        models.append(model)
    print(f"已加载 {len(models)} 个折模型")

    test_files = sorted(TEST_DIR.glob("*.jpg"))
    results = []
    # 多尺度 TTA：两个尺度 × 水平翻转（共 4 个视图）
    scale_sizes = [(224, 224), (256, 224)]  # (resize, center-crop)
    with torch.no_grad():
        for img_path in test_files:
            image = Image.open(img_path)
            image = enhance(image)   # CLAHE 对比度增强（与训练一致）
            image = image.convert("RGB")

            views = []
            for rs, cs in scale_sizes:
                im = image.resize((rs, rs))
                if rs != cs:
                    im = transforms.CenterCrop(cs)(im)
                t = test_transform(im)          # [3,224,224] 归一化后
                views.append(t)
                views.append(torch.flip(t, dims=[2]))  # 水平翻转

            probs = None
            for v in views:
                v = v.unsqueeze(0).to(device)
                for model in models:
                    logits, _ = model(v)
                    out = torch.softmax(logits, dim=1)
                    probs = out if probs is None else probs + out
            probs = probs / (len(views) * len(models))  # 视图数 × 折数 平均
            pred = probs.argmax(dim=1).item()
            results.append((img_path.name, pred))

    out_path = SAVE_DIR / "submission.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Emotion"])
        for filename, pred in results:
            writer.writerow([filename, pred])

    print(f"已生成 {out_path}，共 {len(results)} 条预测")


if __name__ == "__main__":
    main()
