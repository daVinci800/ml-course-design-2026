"""Build the Oct27 delivery, without inventing a validation or leaderboard score."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from final_round import ROOT,setup,read,write


def main():
    setup()
    from full_refit_v12 import JOB,REPORT,SEEDS
    from run_final_round import process_alive
    from fovea.train import manifest_hash
    import torch
    state=read(JOB/'status.json')
    assert state['phase']=='completed' and not process_alive(state['pid'])
    done=read(REPORT/'completion.json')
    manifest=read(ROOT/'prepared/manifest.json')
    new=Path(done['submission'])
    old=ROOT/'submissions/submission_baseline_kaggle14_20261021.csv'
    assert hashlib.sha256(old.read_bytes()).hexdigest()=='b7bdec6f12ce5fe389bfa22f80f1bf383c7791904d31eeb30ace80148b72bf3c'
    assert hashlib.sha256(new.read_bytes()).hexdigest()==done['sha256']
    assert done['new_oof_rmse'] is None and done['new_public_rmse'] is None
    assert done['training_count_per_model']==80
    rows=list(csv.DictReader(new.open(encoding='utf-8-sig',newline='')))
    template=list(csv.DictReader(open(manifest['template'],encoding='utf-8-sig',newline='')))
    assert len(rows)==40 and [r['ImageID'] for r in rows]==[r['ImageID'] for r in template]
    tests={r['id']:r for r in manifest['records'] if r['split']=='test'}
    for r in rows:
        ident,axis=r['ImageID'].split('_Fovea_')
        limit=tests[int(ident)]['width' if axis=='X' else 'height']-1
        assert np.isfinite(float(r['value'])) and 0<=float(r['value'])<=limit
    check=[]
    for seed in SEEDS:
        folder=JOB/f'seed_{seed}'
        trained=read(folder/'completed.json')
        assert trained['epochs']==160 and trained['training_count']==80
        history=list(csv.DictReader((folder/'train/results.csv').open(encoding='utf-8-sig',newline='')))
        assert len(history)==160
        epoch_key=next(k for k in history[0] if k.strip()=='epoch')
        assert [int(float(row[epoch_key])) for row in history]==list(range(1,161))
        checkpoint=torch.load(folder/'averaged.pt',map_location='cpu',weights_only=False)
        assert checkpoint['fovea_seed']==seed and checkpoint['fovea_training_count']==80
        assert checkpoint['fovea_training_split']=='all_train'
        assert checkpoint['fovea_manifest_hash']==manifest_hash(manifest)
        assert checkpoint['averaged_epochs']==(80,120,160)
        check.append(dict(seed=seed,training_count=80,verified_training_epochs=len(history),sha256=hashlib.sha256((folder/'averaged.pt').read_bytes()).hexdigest()))
        del checkpoint
    write(REPORT/'delivery_verification.json',dict(rows=40,template_exact=True,finite_and_in_bounds=True,
          baseline_unchanged=True,checkpoints=check,submission_sha256=done['sha256'],new_oof_rmse=None,new_public_rmse=None))
    experiments=[('point_v8','point','冻结检测器的坐标分支'),('point_v9','point','坐标分支＋特征微调'),
                 ('rcnn_v10','rcnn','独立两阶段检测器'),('rcnn_refine_v11','rcnn','两阶段检测＋旧定位约束')]
    lines=['# 2026年10月27日：优化排查与全量重训候选','',
        '**本次交付是一份待Kaggle实测的全量重训候选。没有新的独立OOF分数，没有新的Kaggle分数，尚未达到或验证RMSE≤5。**','',
        '已知实际比赛成绩：较好原版 **14.39947**；上次融合版 **15.94521**。用户授权本轮最多四次新提交；本报告生成时新提交为 **0次**。',
        '', '自动提交受当前工具能力限制：无法读取和操作桌面窗口，屏幕截图调用返回“句柄无效”；本机未配置Kaggle API凭据。没有声称已在Edge点击或已上传。',
        '', '## 今天的独立验证排查','',
        '| 方案 | 单模型五折RMSE | 固定融合五折RMSE | 改善折数 | 通过预设门槛 |',
        '|---|---:|---:|---:|---|','| 原版方法 | 16.57995 | — | — | 已有14.39947比赛成绩 |']
    for name,key,label in experiments:
        d=read(ROOT/'reports'/name/'decision.json')
        lines.append(f'| {label} | {d[key]["rmse"]:.5f} | {d["proposed"]["rmse"]:.5f} | {d["improved_folds"]}/5 | {"是" if d["recommend_new"] else "否"} |')
    lines+=['', '这些新结构路线都未通过事先设定的稳定性门槛，没有把它们当作提分版本提交。局部约束是在观察两阶段模型前两折后加入，对全部图使用同一0.1倍视野直径半径；其余三折未用于设定该规则。没有继续搜索半径。',
       '', '## 最终候选如何生成','',
       '最终沿用已获14.39947的YOLO11s方法，从本机缓存的官方COCO权重重新训练，三个固定种子分别为20261027、20261028、20261029。每个模型使用全部80张训练图，原来的五折模型每个约使用64张。20张测试图不进入训练。',
       '', '每个种子固定训练160轮，平均第80、120、160轮EMA权重，不根据训练集终轮mAP选择检查点。最终预测为三个全量模型各25%，保留原较好版25%。没有搜索融合系数。',
       '', f'全量重训、权重平均和20张测试图推理用时 **{done["wall_minutes"]:.1f}分钟**。',
       '', '**全量重训没有留出新的验证集。旧版16.57995是旧五折模型的分数，不能移用成这些全量模型的验证分数。训练日志中的终轮val/mAP实际在训练图上计算，只是拟合情况，不是泛化验证。**',
       '', '全量使用已有标签是常见的最终重训方式，但在这个数据集上是否优于原版必须靠一次真实提交检验。三个种子和后期权重平均用于降低波动，不构成提分或达到5的保证。',
       '', '## 文件与核验','',
       '- `01_全量重训候选_待实测.csv`：本次唯一新候选，40行，模板顺序完全一致，坐标有限且在图像边界内。',
       '- `02_已得14.39947_原版备份.csv`：原较好版本，内容和SHA256保留，不需要重复提交占次数。',
       f'- 新候选SHA256：`{done["sha256"]}`。',
       '- 67项代码测试通过；训练列表检查确认只包含80张训练图，不含测试图。',
       '- 训练和推理：`full_refit_v12.py`；模型、完整断点和日志：`runs/full_refit_v12/`；核验：`reports/full_refit_v12/`。',
       '- 代码包包含源码、配置、验证证据和本报告，不含原始图像或训练权重。权重仍在项目目录，不要删除。',
       '', '80张训练图均为可见中央凹，缺少真实不可见负例，不能宣称已验证不可见识别。没有手工标注测试图或逐图修改坐标。',
       '', '公开榜只覆盖10张测试图，本地OOF和公开榜不能直接比较，公开榜改善也不保证最终私榜改善。最多四次新提交应按真实评分记录，目标≤5尚未实现。','']
    report=ROOT/'20261027_全量重训与优化结果.md'
    report.write_text('\n'.join(lines),encoding='utf8')
    delivery=ROOT.parent/'full_refit_delivery_20261027'
    delivery.mkdir(exist_ok=True)
    pairs={new:'01_全量重训候选_待实测.csv',old:'02_已得14.39947_原版备份.csv',report:report.name}
    for source,name in pairs.items():
        shutil.copy2(source,delivery/name)
        assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256((delivery/name).read_bytes()).digest()
    text='''本次唯一新候选：01_全量重训候选_待实测.csv
请在比赛“提交预测”页面选择该CSV。不要上传代码ZIP或整个文件夹。
该候选采用原较好方法在全部80张训练图上重训，不是已验证提分的新架构。
没有新OOF分数；新的Kaggle分数未知；未达到或验证RMSE≤5。

02_已得14.39947_原版备份.csv 是原较好版本，内容原样保留，不需要重复提交占次数。
上次融合15.94521版不作为本次候选。

本轮实际新提交0次，用户授权上限4次。当前环境无法操作桌面Edge，且未配置Kaggle API。
请在网页提交后记录本次实际分数，再决定后续优化；也可在本机配置官方Kaggle API凭据供自动提交，不要把密钥发到聊天。
在比赛截止前按Kaggle页面规则保留并选择最终提交。公开榜改善不保证私榜改善。

详细实验失败对照、全量重训方法和核验见结果报告。
代码包不包含原图或训练权重。完整权重和断点保存在项目runs/full_refit_v12，原版权重也保留。
'''
    (delivery/'00_提交说明.txt').write_text(text,encoding='utf-8-sig')
    with zipfile.ZipFile(delivery/'代码与验证报告.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.glob('*.py'): z.write(p,p.name)
        for p in ROOT.glob('requirements*.txt'): z.write(p,p.name)
        if (ROOT/'README.md').exists(): z.write(ROOT/'README.md','README.md')
        for name in ('fovea','tests','configs'):
            for p in (ROOT/name).glob('*'):
                if p.is_file() and p.suffix in ('.py','.json'): z.write(p,p.relative_to(ROOT))
        for name in [x[0] for x in experiments]+['full_refit_v12']:
            for p in (ROOT/'reports'/name).glob('*.json'): z.write(p,p.relative_to(ROOT))
        z.write(report,report.name)
        z.write(REPORT/'tests.log','reports/full_refit_v12/tests.log')
        z.write(ROOT/'reports/submission_budget_20261027.json','reports/submission_budget_20261027.json')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in delivery.iterdir() if p.is_file() and p.name!='文件校验.json'}
    write(delivery/'文件校验.json',hashes)
    print(json.dumps(dict(delivery=str(delivery),sha256=done['sha256'],new_score_unknown=True,hashes=hashes),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
