"""Verify completed artifacts and assemble a clear submission delivery folder."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from final_round import ROOT, WORK, read, write, setup


def main():
    setup()
    from optimize_point_v8 import JOB, REPORT
    from fovea.train import manifest_hash
    import torch
    import numpy as np
    state=read(JOB/'status.json')
    if state['phase']!='completed': raise RuntimeError('Training/validation not complete')
    decision=read(REPORT/'decision.json')
    completion=read(REPORT/'completion.json')
    manifest=read(ROOT/'prepared/manifest.json')
    old_file=ROOT/'submissions/submission_baseline_kaggle14_20261021.csv'
    assert hashlib.sha256(old_file.read_bytes()).hexdigest()=='b7bdec6f12ce5fe389bfa22f80f1bf383c7791904d31eeb30ace80148b72bf3c'
    proof=[]
    for fold in range(5):
        assert read(JOB/f'fold_{fold}/completed.json')['epochs']==160
        old=torch.load(ROOT/f'runs/detector_v6/fold_{fold}/point_best.pt',map_location='cpu',weights_only=False)['model'].float().state_dict()
        checkpoint=torch.load(JOB/f'fold_{fold}/averaged.pt',map_location='cpu',weights_only=False)
        assert checkpoint['fovea_fold']==fold and checkpoint['fovea_manifest_hash']==manifest_hash(manifest)
        new=checkpoint['model'].float().state_dict()
        changed=[k for k,v in old.items() if k not in new or not torch.equal(v,new[k])]
        assert not changed, (fold,changed)
        proof.append(dict(fold=fold,old_detector_tensors_checked=len(old),old_detector_unchanged=True,
              averaged_sha256=hashlib.sha256((JOB/f'fold_{fold}/averaged.pt').read_bytes()).hexdigest()))
        del old,new,checkpoint
    write(REPORT/'frozen_weight_verification.json',dict(folds=proof,all_frozen_weights_exact=True))
    before,point,after=decision['baseline'],decision['point'],decision['proposed']
    percent=(before['rmse']-after['rmse'])/before['rmse']*100
    recommend=decision['recommend_new']
    lines=['# 2026-10-27 中心凹定位优化结果','',
        f'已知 Kaggle：旧版 **14.39947**；上次融合版 **15.94521**。本轮没有自动提交，新的比赛分数未知。','',
        f'本轮五折固定训练、权重平均、验证及条件推理共用时 **{completion["wall_minutes"]:.1f} 分钟**。','',
        '| 同一批训练图的 OOF 结果 | RMSE |','|---|---:|',
        f'| 旧版 | {before["rmse"]:.5f} |',f'| 直接坐标分支单独预测 | {point["rmse"]:.5f} |',
        f'| 预先固定的 75%旧版＋25%坐标分支 | {after["rmse"]:.5f} |','',
        f'相对旧版本地变化：**{percent:.2f}%**；改善折数：**{decision["improved_folds"]}/5**。','',
        '| 验证折 | 旧版 | 新候选 |','|---|---:|---:|']
    for f in range(5):
        lines.append(f'| {f} | {before["per_fold_rmse"][str(f)]:.5f} | {after["per_fold_rmse"][str(f)]:.5f} |')
    paired=decision['paired']
    lines += ['',f'按重复图分组的成对 Bootstrap：改善比例 **{paired["fraction_bootstrap_improved"]*100:.2f}%**；新减旧 RMSE 差的 95% 区间为 **{paired["rmse_difference_95"]}**。',
        '', '该比例是重采样诊断，不是“比赛提分概率”。旧检测器检查点此前使用过这套验证折选择，因此整个诊断仍非全新独立验证。',
        '', '预先登记的推荐条件：总体 RMSE 至少降低 3%；至少四折改善；成对 Bootstrap 至少 90% 改善；任何一折恶化不超过 10%。',
        '', '**本轮通过推荐门槛，生成候选提交文件，建议先提交验证真实成绩。**' if recommend else '**本轮未通过推荐门槛，保留已得 14.39947 的旧版；没有生成或推荐未经支持的新提交文件。**',
        '', '训练方式：每折只加载同折旧检测器，全部原检测器权重和 BatchNorm 统计冻结，只训练 286377 个新增关键点参数。目标来自中心凹 CSV 的精确坐标；检测框用于目标分配。每折固定 160 轮，平均 120、140、160 轮 EMA，不选最低验证轮数，不搜索融合系数。',
        '', '五折导出权重已逐张量检查，全部原检测器张量与旧版一致；训练重复图始终留在同一折。58 项代码测试通过。未使用外部眼底数据或手工测试标注。',
        '', '80 张训练图均为可见中央凹，没有真实不可见样本来验证输出 (0,0) 的策略。',
        '', '复现：`optimize_point_v8.py`；训练日志和权重：`runs/point_v8/`；验证和决策：`reports/point_v8/`；本文件生成：`point_v8_report.py`。代码包不包含数据和训练权重。','']
    report=ROOT/'20261027_直接坐标优化结果.md'
    report.write_text('\n'.join(lines),encoding='utf8')
    folder=ROOT.parent/'point_v8_delivery_20261027'
    folder.mkdir(exist_ok=True)
    files={old_file:'02_已得14.39947_原版备份.csv',report:report.name}
    if recommend:
        submission=Path(completion['submission'])
        assert hashlib.sha256(submission.read_bytes()).hexdigest()==completion['sha256']
        rows=list(csv.DictReader(submission.open(encoding='utf-8-sig',newline='')))
        template=list(csv.DictReader(Path(manifest['template']).open(encoding='utf-8-sig',newline='')))
        assert len(rows)==40 and [r['ImageID'] for r in rows]==[r['ImageID'] for r in template]
        test={r['id']:r for r in manifest['records'] if r['split']=='test'}
        for r in rows:
            ident,axis=r['ImageID'].split('_Fovea_')
            limit=test[int(ident)]['width' if axis=='X' else 'height']-1
            assert np.isfinite(float(r['value'])) and 0<=float(r['value'])<=limit
        files[submission]='01_本次候选提交.csv'
    for source,name in files.items():
        shutil.copy2(source,folder/name)
        assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256((folder/name).read_bytes()).digest()
    instruction=('本次建议先上传 01_本次候选提交.csv 验证比赛成绩。新分数未知，不能保证优于14.39947。\n' if recommend else '本次实验未通过推荐门槛，请继续保留已得14.39947的原提交。\n')
    instruction+='02_已得14.39947_原版备份.csv 是成绩较好的旧版，内容原样保留。\n上次15.94521版本不作为本次推荐。\n比赛每天最多5次提交，请保留较好成绩，并在截止前按Kaggle页面规则选择最终提交。\n详细证据见本文件夹的优化结果报告。代码包包含源代码和验证报告，不含模型权重和原始图片；模型仍在项目目录。\n没有自动提交Kaggle。\n'
    (folder/'00_提交说明.txt').write_text(instruction,encoding='utf-8-sig')
    with zipfile.ZipFile(folder/'代码与验证报告.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.glob('*.py'): z.write(p,p.name)
        for group in ('fovea','tests'):
            for p in (ROOT/group).glob('*.py'): z.write(p,p.relative_to(ROOT))
        for p in REPORT.glob('*.json'): z.write(p,p.relative_to(ROOT))
        z.write(report,report.name)
        z.write(ROOT/'20261027_直接坐标优化说明.md','20261027_直接坐标优化说明.md')
        z.write(REPORT/'tests.log','reports/point_v8/tests.log')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file() and p.name!='文件校验.json'}
    write(folder/'文件校验.json',hashes)
    print(json.dumps(dict(folder=str(folder),recommend_new=recommend,metrics=after['rmse'],improved_folds=decision['improved_folds'],hashes=hashes),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
