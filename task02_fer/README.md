# 人脸情感分类
original/ 保留此前 ConvNeXt-Tiny + CBAM 方案原始代码。原临时目录训练产物未留存，不能用原方案的计划参数证明原训练已完成。
本报告重训使用 train_report.py：固定像素重复分组的约 80/20 分层留出，固定种子 20261110，12 epochs，ConvNeXt-Tiny + CBAM，class-weighted focal + Mixup + Center Loss + EMA。
同像素异标签组排除；同像素同标签组整组分配。划分、每轮日志和逐图验证预测都在 evidence。
正式训练环境为 Python 3.12、PyTorch 2.7.1 CUDA 12.6；预训练源为 timm ConvNeXt-Tiny in12k_ft_in1k，不是外部人脸任务权重。
```
python train_report.py --data-root /path/to/fer_data --cache-root /path/to/cache_clahe --weights /path/to/model.safetensors --output runs/report --epochs 12 --batch-size 32 --workers 4 --execute
```
CLAHE 缓存需要先依据 original/preprocess_cache.py 生成，注意配置 original/dataset.py 的数据路径。预处理保持 clipLimit=2.0、tileGridSize=(8,8)，灰度增强后转 RGB。
本次脚本修正了原实现加权 Focal 的 pt 定义、Mixup 下 Center Loss 只使用第一标签的问题，并保留 EMA 和原始参数。last.pt 是可审计训练状态快照，添加 --resume 从末个完整 epoch 恢复。
本次实际在第 4 轮后暂停并恢复，12 轮已完成，最佳为第 9 轮：验证 accuracy 0.688565，Macro-F1 0.671051。主进程 RNG 和优化器状态已恢复，但多进程增强工作线程重建，不能声称逐位等同于无中断训练。
验证集用于每轮模型选择；报告结果是留出验证成绩，不能称为独立测试成绩。训练权重保存在本机，不包含在本仓库。
