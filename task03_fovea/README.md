# 中央凹定位
80 张训练图、20 张测试图。沿用原始五折划分。主要可复查方案为低学习率 YOLO11s 框中心定位，未检出时使用对应折热图模型自动回退。
本地 OOF 坐标 RMSE 16.579950，历史反馈记录原较好 Kaggle 成绩 14.39947；前者与后者属于不同评估口径。
源码保留全部主要对照与最后全量重训方案。2026-10-27 三种子全量候选没有独立验证成绩；不把旧 OOF 指标移用到它。
```
python run.py prepare --data-root /path/to/detection --out prepared
python optimize_detector.py --help
python full_refit_v12.py --help
```
数据位置需按本机调整。依赖见 requirements.txt，YOLO11s 使用官方 COCO 初始化。检查配置和命令后才显式运行训练。
prepared 保存原始划分与标注元数据；evidence 保存原始评估和历史提交反馈。图像、缓存和模型权重不入仓库。
