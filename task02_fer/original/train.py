# 人脸情感分类训练：Focal Loss + Mixup + CutOut + EMA + 多种子五折集成 + 断点续训
import copy
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from dataset import records, get_dataloaders, num_classes, assign_folds
from model import create_model

# 模型/checkpoint 保存到系统临时目录
SAVE_DIR = Path(tempfile.gettempdir()) / "fer_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = SAVE_DIR / "checkpoint.pth"


def compute_weights(recs):
    counts = Counter(r["class_idx"] for r in recs)
    total = len(recs)
    return [total / (num_classes * counts[i]) for i in range(num_classes)]


class FocalLoss(nn.Module):
    """Focal Loss：降低易分类样本权重，聚焦难样本，缓解类别不平衡与标签噪声"""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class CenterLoss(nn.Module):
    """Center Loss：让同类样本的特征靠近其类别中心，增强类内紧凑性"""
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        centers_batch = self.centers.index_select(0, labels)  # [B, feat_dim]
        loss = (features - centers_batch).pow(2).sum() / features.size(0)
        return loss


class ModelEMA:
    """模型权重指数滑动平均（EMA）：提升泛化、抑制噪声标签"""
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_p, p in zip(self.ema.parameters(), model.parameters()):
                ema_p.mul_(self.decay).add_(p, alpha=1 - self.decay)


def save_ckpt(round_tag, fold, epoch, model, center_loss, optimizer, scheduler,
              best_val_acc, fold_accs, oof_preds):
    torch.save({
        "round": round_tag,
        "fold": fold,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "center_state": center_loss.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "fold_accs": fold_accs,
        "oof_preds": oof_preds,
    }, CKPT_PATH)


def mixup_data(x, y, alpha=0.2):
    """Mixup：按 Beta 分布取混合比例，线性混合两张图与标签，抑制标签噪声"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam = max(lam, 1 - lam)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def train_one_round(recs, device, num_epochs, batch_size, round_tag,
                    save_oof_path=None, resume=None, mixup_alpha=0.2, seed=0):
    """做一轮 5 折训练，支持断点续训"""
    num_folds = 5
    weight = torch.tensor(compute_weights(recs), dtype=torch.float32).to(device)
    criterion = FocalLoss(gamma=2.0, weight=weight)

    oof_preds = dict(resume["oof_preds"]) if resume else {}
    fold_accs = list(resume["fold_accs"]) if resume else []
    start_fold = resume["fold"] if resume else 0

    for fold in range(start_fold, num_folds):
        print(f"\n===== [{round_tag}] Fold {fold} =====")
        train_loader, val_loader = get_dataloaders(recs, fold, batch_size)
        model = create_model(num_classes=num_classes).to(device)
        center_loss = CenterLoss(num_classes, model.num_features).to(device)
        ema = ModelEMA(model, decay=0.999)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(center_loss.parameters()),
            lr=1e-4, weight_decay=0.01,
        )
        # warm-up + 余弦退火
        warmup_epochs = 3

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_val_acc = 0.0
        this_start_epoch = 0
        best_fold_preds = {}
        # 恢复本折状态（仅当 checkpoint 正好停在这个 fold 的某个 epoch）
        if resume and resume["round"] == round_tag and resume["fold"] == fold and resume.get("epoch", -1) >= 0:
            model.load_state_dict(resume["model_state"])
            center_loss.load_state_dict(resume["center_state"])
            optimizer.load_state_dict(resume["optimizer_state"])
            scheduler.load_state_dict(resume["scheduler_state"])
            best_val_acc = resume["best_val_acc"]
            this_start_epoch = resume["epoch"] + 1
            ema = ModelEMA(model, decay=0.999)  # 重新初始化 EMA
            print(f"从 epoch {this_start_epoch} 继续")

        for epoch in range(this_start_epoch, num_epochs):
            # 训练
            model.train()
            running_loss = 0.0
            for images, labels, _ in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                if mixup_alpha > 0:
                    images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
                    logits, feat = model(images)
                    loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b) \
                        + 0.01 * center_loss(feat, y_a)
                else:
                    logits, feat = model(images)
                    loss = criterion(logits, labels) + 0.01 * center_loss(feat, labels)
                loss.backward()
                optimizer.step()
                ema.update(model)
                running_loss += loss.item()

            # 验证（用 EMA 模型）
            ema.ema.eval()
            correct = 0
            total = 0
            fold_preds = {}
            with torch.no_grad():
                for images, labels, filenames in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs, _ = ema.ema(images)
                    _, preds = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()
                    for fn, p in zip(filenames, preds):
                        fold_preds[fn] = p.item()

            val_acc = correct / total
            running_loss /= len(train_loader)
            print(f"epoch {epoch} | 训练损失 {running_loss:.3f} | 验证准确率 {val_acc:.2%}")
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_fold_preds = fold_preds.copy()
                torch.save(ema.ema.state_dict(), SAVE_DIR / f"best_model_fold{fold}_seed{seed}.pth")

            # 每个 epoch 结束保存 checkpoint（断点续训）
            save_ckpt(round_tag, fold, epoch, model, center_loss, optimizer, scheduler,
                      best_val_acc, fold_accs, oof_preds)

        oof_preds.update(best_fold_preds)
        fold_accs.append(best_val_acc)
        print(f"Fold {fold} 最佳验证准确率: {best_val_acc:.2%}")

    if save_oof_path is not None:
        with open(save_oof_path, "w") as f:
            json.dump(oof_preds, f)
        print(f"OOF 预测已保存到 {save_oof_path}")

    print(f"[{round_tag}] 各折最佳: {[f'{a:.2%}' for a in fold_accs]}, 平均 {sum(fold_accs)/len(fold_accs):.2%}")
    return oof_preds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 15
    batch_size = 32
    seeds = [0, 1]  # 多种子集成

    # 根据 checkpoint 确定从哪个种子继续（避免重训已完成的种子）
    start_seed = 0
    resume = None
    if CKPT_PATH.exists():
        ck = torch.load(CKPT_PATH, map_location="cpu")
        for i, s in enumerate(seeds):
            if ck.get("round") == f"R4_s{s}":
                start_seed = i
                resume = ck
                print(f"检测到 {ck['round']} checkpoint：fold={ck['fold']}, epoch={ck['epoch']}，继续")
                break

    for i in range(start_seed, len(seeds)):
        seed = seeds[i]
        torch.manual_seed(seed)
        np.random.seed(seed)
        round_tag = f"R4_s{seed}"
        cur_resume = resume if i == start_seed else None

        print("=" * 50)
        print(f"{round_tag}：Focal Loss + EMA + CutOut + Mixup + 多种子集成")
        train_one_round(records, device, num_epochs, batch_size,
                        round_tag=round_tag, save_oof_path=SAVE_DIR / f"oof_preds_s{seed}.json",
                        resume=cur_resume, mixup_alpha=0.2, seed=seed)

    print("\n训练完成，最终模型保存在临时目录 best_model_fold*_seed*.pth")


if __name__ == "__main__":
    main()
