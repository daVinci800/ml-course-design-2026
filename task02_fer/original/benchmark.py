# 测 AMP 混合精度对 batch 32/64 的加速和显存占用
import time
import torch
from model import create_model


def bench(bs, use_amp):
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    model = create_model(num_classes=6).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    x = torch.randn(bs, 3, 224, 224, device=device)
    y = torch.randint(0, 6, (bs,), device=device)

    for _ in range(3):
        opt.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                logits, feat = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits, feat = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(20):
        opt.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                logits, feat = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits, feat = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
    torch.cuda.synchronize()
    t1 = time.time()

    tag = "AMP" if use_amp else "fp32"
    ms = (t1 - t0) / 20 * 1000
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"batch={bs} {tag}: {ms:.1f} ms/次，峰值显存 {mem:.2f} GB，吞吐 {bs / (ms / 1000):.0f} 张/秒", flush=True)
    return ms


def main():
    print("开始", flush=True)
    for use_amp in [False, True]:
        for bs in [32, 64]:
            try:
                bench(bs, use_amp)
            except Exception as e:
                print(f"batch={bs} {'AMP' if use_amp else 'fp32'} 失败: {e}", flush=True)


if __name__ == "__main__":
    main()
