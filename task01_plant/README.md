# 植物分类
真实完成的分层五折实验。ResNet50 与 EfficientNet-B3 概率等权融合的 500 样本 OOF accuracy 为 0.920，macro-F1 为 0.920197。
原始源码在 src，固定划分在 data/folds_v2.csv，原始配置、训练日志、OOF、混淆矩阵在 evidence。
数据和权重不放入 GitHub。将训练图置于 data/train/类别名，测试图置于 data/test。
依赖见 requirements.txt。运行方式：
```
python -m src.train_round2_fast --manifest data/folds_v2.csv --data-root data --folds 0 1 2 3 4 --output-dir runs/round2_fast_resnet50
python -m src.train_round3_effnet --help
python -m src.infer_round3_effnet_blend --help
```
正式训练需要将 pretrained 权重放到源码指定位置，或使用相应入口的 --weights 选项。缓存路径在原始配置中留存。不同入口参数以 --help 为准。
OOF 参与模型与推理策略选择，92% 不能替代独立测试集或 Kaggle 成绩。
