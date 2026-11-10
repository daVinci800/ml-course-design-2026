from pathlib import Path

import timm
import torch
import torchvision


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "test"


def main() -> None:
    print("=" * 60)
    print("环境检查")
    print("=" * 60)

    print(f"PyTorch版本：{torch.__version__}")
    print(f"TorchVision版本：{torchvision.__version__}")
    print(f"timm版本：{timm.__version__}")
    print(f"CUDA运行库：{torch.version.cuda}")
    print(f"CUDA是否可用：{torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"显卡：{torch.cuda.get_device_name(0)}")
        print(
            f"显存："
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

        test_tensor = torch.randn(1024, 1024, device="cuda")
        test_result = test_tensor @ test_tensor

        print(f"GPU运算设备：{test_result.device}")
        print("GPU运算测试：成功")
    else:
        print("GPU运算测试：未执行，因为CUDA不可用")

    print()
    print("=" * 60)
    print("数据集检查")
    print("=" * 60)

    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"训练集目录：{TRAIN_DIR}")
    print(f"测试集目录：{TEST_DIR}")
    print(f"训练集是否存在：{TRAIN_DIR.exists()}")
    print(f"测试集是否存在：{TEST_DIR.exists()}")

    if TRAIN_DIR.exists():
        class_names = sorted(
            directory.name
            for directory in TRAIN_DIR.iterdir()
            if directory.is_dir()
        )

        print(f"类别数量：{len(class_names)}")

        total_train_images = 0

        for class_name in class_names:
            class_dir = TRAIN_DIR / class_name
            image_count = len(list(class_dir.glob("*.png")))
            total_train_images += image_count
            print(f"  {class_name}：{image_count}张")

        print(f"训练图像总数：{total_train_images}")

    if TEST_DIR.exists():
          test_count = len(list(TEST_DIR.glob("*.png")))
          print(f"测试图像总数：{test_count}")

if __name__ == "__main__":
      main()