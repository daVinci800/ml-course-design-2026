# 眼底血管分割
已真实完成五折训练和推理。原尺寸 OOF Dice 0.817872，官方缩放规则代理 OOF Dice 0.842116。指标使用不同标签处理和阈值，不宜直接比较。
残差 U-Net + 空间注意力 + 深监督，RGB 与绿色 CLAHE 四通道输入。配置在 configs/main.json，真实每轮日志在 evidence/foldN/history.jsonl。
```
python main.py prepare --zip /path/to/2026task-4.zip
python main.py train --fold 0 --device cuda --execute
python main.py calibrate --device cuda --tta --metric official-proxy
python main.py predict --device cuda --tta --output predictions/main --csv-foreground vessel
```
依次训练 5 折后才校准与推理。最终校准扫描范围为 0.01–0.99，历史说明中的 0.20–0.80 为早期默认，报告和最终核验以实际日志为准。
标签中 255 为血管；展示图可输出血管 0 背景 255，CSV 编码仍是血管前景。
数据、模型、逐图概率不入仓库。完整原尺寸图和 CSV 本机保留。
