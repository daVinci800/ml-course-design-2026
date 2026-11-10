from pathlib import Path
import csv,json,shutil,hashlib
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'outputs'
PKG=OUT/'ml_course_design_code'
PKG.mkdir(parents=True,exist_ok=True)
PLANT=Path(r'C:\Users\Lenovo\PycharmProjects\plant_classifier')
FER=Path(r'C:\Users\Lenovo\PycharmProjects\ml_course_design\fer_classification')
FOVEA=Path(r'C:\Users\Lenovo\Documents\Codex\2026-10-20\1-0-0-2-3-rules\outputs\fovea_solution')
VESSEL=Path(r'C:\Users\Lenovo\Documents\Codex\2026-10-31\2d-0-255-rules-2026-11\outputs\retina_vessel')
def copy_file(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def tree(src,dst,extensions=None,exclude=()):
    if not src.exists():return
    for p in src.rglob('*'):
        if p.is_file() and not any(x in p.relative_to(src).parts for x in exclude) and (extensions is None or p.suffix.lower() in extensions):copy_file(p,dst/p.relative_to(src))
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
def collect():
    p1=PKG/'task01_plant'
    for name in ['src','configs','tests']:tree(PLANT/name,p1/name,{'.py','.json','.yaml','.yml'},('__pycache__',))
    for name in ['requirements.txt','check_environment.py','.gitignore']:copy_file(PLANT/name,p1/name)
    for name in ['folds_v2.csv','submission_fortask1.csv']:copy_file(PLANT/'data'/name,p1/'data'/name)
    for name in ['resnet50_seed2026','round2_fast_resnet50','round2_multiscale_final','round3_effnet_b3','round3_effnet_resnet_blend']:
        tree(PLANT/'runs'/name,p1/'evidence'/name,{'.json','.csv'},())
    tree(Path(r'C:\Users\Lenovo\Desktop\2026task1_extracted\analysis'),p1/'evidence'/'dataset_analysis',{'.json','.md','.csv'})
    write(p1/'README.md','''# 植物分类
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
''')
    p2=PKG/'task02_fer'
    tree(FER,p2/'original',{'.py'},('__pycache__','cache_clahe'))
    copy_file(BASE/'work/fer_report_train.py',p2/'train_report.py')
    tree(BASE/'work/fer_retrain',p2/'evidence', {'.json','.jsonl','.csv'})
    write(p2/'requirements.txt','torch==2.7.1\ntorchvision==0.22.1\ntimm==1.0.22\nsafetensors\nhuggingface_hub==0.35.3\nnumpy\nPillow\nopencv-python\nscikit-learn\n')
    write(p2/'README.md','''# 人脸情感分类
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
''')
    p3=PKG/'task03_fovea'
    for p in FOVEA.glob('*.py'):copy_file(p,p3/p.name)
    for name in ['fovea','configs','tests']:tree(FOVEA/name,p3/name,{'.py','.json','.yaml','.yml'},('__pycache__',))
    for name in ['requirements.txt','.gitignore']:copy_file(FOVEA/name,p3/name)
    tree(FOVEA/'prepared',p3/'prepared',{'.json','.csv'})
    for name in ['detector_v5','detector_v6','full_refit_v12','point_v8','point_v9','rcnn_v10','rcnn_refine_v11']:
        tree(FOVEA/'reports'/name,p3/'evidence'/name,{'.json','.csv','.md'})
    for name in ['本轮优化最终结果.md','20261027_全量重训与优化结果.md','20261027_提交反馈记录.md','论文依据.md']:
        copy_file(FOVEA/name,p3/'evidence'/name)
    tree(FOVEA/'submissions',p3/'evidence'/'submissions',{'.csv','.json'})
    write(p3/'README.md','''# 中央凹定位
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
''')
    p4=PKG/'task04_vessel'
    for name in ['vessel','configs','tests','scripts']:tree(VESSEL/name,p4/name,{'.py','.json','.ps1'},('__pycache__',))
    for name in ['main.py','requirements.txt','.gitignore']:copy_file(VESSEL/name,p4/name)
    for name in ['data_audit.json','方案与论文调研.md','qa_environment.txt']:copy_file(VESSEL/'docs'/name,p4/'evidence'/name)
    for fold in range(5):tree(VESSEL/f'runs/main/fold{fold}',p4/f'evidence/fold{fold}',{'.json','.jsonl'})
    for name in ['训练与提交核验.json','折外验证逐图结果.csv','提交说明.md']:
        copy_file(Path(r'C:\Users\Lenovo\Desktop\2026task4_提交结果')/name,p4/'evidence'/name)
    tree(VESSEL/'artifacts',p4/'evidence'/'artifacts',{'.json','.csv'})
    write(p4/'README.md','''# 眼底血管分割
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
''')
    write(PKG/'.gitignore','''.venv/
__pycache__/
*.pyc
*.pt
*.pth
*.safetensors
*.npy
*.npz
data/train/
data/test/
data/segmentation/
cache_clahe/
''')
    write(PKG/'README.md','''# 机器学习课程设计四项实验
作者：姜广阔，学号 20245621。
四个目录依次对应植物分类、人脸情感分类、中央凹定位、眼底血管分割。
每个任务包含实际源码、配置、复查用的训练和验证记录。数据集、虚拟环境、预训练及训练权重不上传 GitHub。
task02 是 2026-11-10 为报告重新开展的固定划分实验，与 original 中历史多种子五折计划分开。
所有指标以原始 evidence 文件为准；验证或 OOF 指标不冒充独立测试或在线成绩。
运行方法见各任务 README。原始配置中的绝对路径记录当时环境，迁移后按实际数据位置调整。
''')
    report_tools=PKG/'report_tools'
    if (BASE/'work/github_result.json').exists():
        copy_file(BASE/'work/github_result.json',report_tools/'github_result.json')
        with (PKG/'README.md').open('a',encoding='utf-8') as f:
            f.write('\nGitHub 仓库：'+load(BASE/'work/github_result.json')['url']+'\n')
    for name in ['build_report.py','make_figures.py','collect_evidence.py','math_equations.cjs']:
        copy_file(BASE/'work'/name,report_tools/name)
    tree(BASE/'work/figures',report_tools/'figures',{'.png','.json'})
    manifest=[]
    for p in sorted(PKG.rglob('*')):
        if p.is_file() and '.git' not in p.relative_to(PKG).parts and p.name!='source_manifest_sha256.json':manifest.append(dict(path=p.relative_to(PKG).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    save=PKG/'source_manifest_sha256.json';save.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(manifest),bytes=sum(r['bytes'] for r in manifest)),ensure_ascii=False))

if __name__=='__main__':collect()
