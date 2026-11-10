"""Audit, summarize all Oct27 experiments, and prepare a single clear delivery."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from final_round import ROOT, read, write, setup


def main():
    setup()
    from rcnn_refine_v11 import JOB,REPORT
    from optimize_rcnn_v10 import JOB as TRAIN_JOB
    from run_final_round import process_alive
    state=read(JOB/'status.json')
    assert state['phase']=='completed'
    assert not process_alive(state['pid'])
    for f in range(5): assert read(TRAIN_JOB/f'fold_{f}/completed.json')['epochs']==40
    decision=read(REPORT/'decision.json')
    completion=read(REPORT/'completion.json')
    experiments=[]
    for name,key,label in [('point_v8','point','冻结旧检测器，直接坐标分支'),('point_v9','point','微调特征融合和坐标分支'),('rcnn_v10','rcnn','独立两阶段检测器'),('rcnn_refine_v11','rcnn','两阶段检测＋旧定位约束')]:
        d=read(ROOT/'reports'/name/'decision.json')
        experiments.append((name,key,label,d))
    before,after=decision['baseline'],decision['proposed']
    lines=['# 2026年10月27日优化结果','',
       '已知 Kaggle：原较好版本 **14.39947**；上次融合版本 **15.94521**。目标是继续降低误差，用户授权最多四次新提交，目标 RMSE 不超过5。',
       '', '截至本报告生成，尚未进行新的 Kaggle 提交。当前工具无法读取和操作桌面窗口，截图调用也返回“句柄无效”；本机没有现成 Kaggle API 登录配置。未把训练或验证次数记为提交次数。',
       '', '| 本地五折方案 | 单模型 RMSE | 固定融合 RMSE | 改善折数 | 通过推荐门槛 |',
       '|---|---:|---:|---:|---|',f'| 原较好版 | {before["rmse"]:.5f} | — | — | 已有比赛成绩 |']
    for name,key,label,d in experiments:
        lines.append(f'| {label} | {d[key]["rmse"]:.5f} | {d["proposed"]["rmse"]:.5f} | {d["improved_folds"]}/5 | {"是" if d["recommend_new"] else "否"} |')
    lines+=['','本次每组只检验一个事先固定的融合比例，不进行大规模配方搜索。坐标分支第一组使用75%旧版，后两组使用50%旧版。',
      '',f'独立两阶段模型阶段耗时 **{read(ROOT/'reports/rcnn_v10/completion.json')["wall_minutes"]:.1f} 分钟**，五折各40轮，平均第30、35、40轮EMA。使用本机已缓存的官方COCO权重，没有外部眼底标签。训练框宽高保留原标注，中心使用CSV精确坐标。',
      '', '| 验证折 | 旧版 | 两阶段模型 | 固定融合 |','|---|---:|---:|---:|']
    for f in range(5):
        lines.append(f'| {f} | {before["per_fold_rmse"][str(f)]:.5f} | {decision["rcnn"]["per_fold_rmse"][str(f)]:.5f} | {after["per_fold_rmse"][str(f)]:.5f} |')
    p=decision['paired']
    lines+=['',f'两阶段固定融合的成对分组Bootstrap改善比例为 **{p["fraction_bootstrap_improved"]*100:.2f}%**，新减旧RMSE差的95%区间为 **{p["rmse_difference_95"]}**。',
      '', '推荐门槛在训练前固定：总体RMSE至少降低3%，至少四折改善，成对分组Bootstrap至少90%改善，任一折恶化不超过10%。',
      '', '**带旧定位约束的两阶段模型通过门槛，生成本次候选CSV；新的比赛分数未知。**' if decision['recommend_new'] else '**带旧定位约束的两阶段模型未通过门槛；当前没有证据充分的新推荐文件，原14.39947版本保留。**',
      '', '本地OOF与公开榜不能直接比较。公开榜只含10张测试图，旧检查点也曾使用同套验证折选择；Bootstrap是条件诊断，不是“比赛提分概率”。目前没有达到或验证RMSE≤5，不能承诺四次内一定达到。',
      '', '重复图始终保持同折。两阶段模型每折独立从官方COCO权重初始化，没有使用别折训练后的权重。最后才按固定融合比例处理测试图，没有手工修正测试坐标。80张训练图均可见，缺少不可见负例来验证输出(0,0)的能力。',
      '', '66项代码检查通过。两阶段模型结构参考Torchvision FasterRCNN ResNet50 FPN v2，其官方实现引用 [Benchmarking Detection Transfer Learning with Vision Transformers](https://arxiv.org/abs/2111.11429)。这里没有重新检索网络或下载新权重。',
      '', '训练代码：`optimize_rcnn_v10.py`；日志、完整断点和平均权重：`runs/rcnn_v10/`；全部验证证据：`reports/rcnn_v10/`。代码包不含原图和模型权重，完整权重保留在项目目录。','']
    later=decision['later_fold_diagnostic']
    lines += ['', '局部约束是在观察原始两阶段模型的前两折后加入的：对全部图使用相同的0.1倍视野直径搜索半径，选取范围内最高置信度框，否则回退同折旧模型。其余三折没有用于设定规则，之后没有搜索半径或修改阈值。',
        '', f'后续三折的单独对照：旧版RMSE {later["baseline"]["rmse"]:.5f}，固定局部精修融合RMSE {later["proposed"]["rmse"]:.5f}。此诊断仍受旧检查点选择影响，不是完全独立测试。', '']
    report=ROOT/'20261027_再次优化结果.md'
    report.write_text('\n'.join(lines),encoding='utf8')
    folder=ROOT.parent/'optimization_refine_delivery_20261027'
    folder.mkdir(exist_ok=True)
    old=ROOT/'submissions/submission_baseline_kaggle14_20261021.csv'
    assert hashlib.sha256(old.read_bytes()).hexdigest()=='b7bdec6f12ce5fe389bfa22f80f1bf383c7791904d31eeb30ace80148b72bf3c'
    pairs={old:'02_已得14.39947_原版备份.csv',report:report.name}
    if decision['recommend_new']:
        new=Path(completion['submission'])
        assert hashlib.sha256(new.read_bytes()).hexdigest()==completion['sha256']
        manifest=read(ROOT/'prepared/manifest.json')
        rows=list(csv.DictReader(new.open(encoding='utf-8-sig',newline='')))
        template=list(csv.DictReader(Path(manifest['template']).open(encoding='utf-8-sig',newline='')))
        assert len(rows)==40 and [r['ImageID'] for r in rows]==[r['ImageID'] for r in template]
        records={r['id']:r for r in manifest['records'] if r['split']=='test'}
        for r in rows:
            ident,axis=r['ImageID'].split('_Fovea_')
            limit=records[int(ident)]['width' if axis=='X' else 'height']-1
            assert np.isfinite(float(r['value'])) and 0<=float(r['value'])<=limit
        pairs[new]='01_本次候选提交.csv'
    for source,name in pairs.items():
        shutil.copy2(source,folder/name)
        assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256((folder/name).read_bytes()).digest()
    text=('建议先提交01_本次候选提交.csv，等待Kaggle实际评分。\n' if decision['recommend_new'] else '本轮未获得通过推荐门槛的新候选，继续保留14.39947版。\n')
    text+='02_已得14.39947_原版备份.csv是原较好版本，内容保持不变。\n本轮实际新提交0次，用户授权最多4次；未达到或验证RMSE≤5。\n当前无法操作桌面Edge，且未配置Kaggle API。需要用户在网页提交，或在本机配置官方API凭据后再自动提交。不要把密钥发到聊天。\n本地验证不等于比赛成绩，不保证新文件优于旧版。比赛截止前请按Kaggle页面规则保留并选择最终提交。\n代码包不包含原图或模型权重。详细验证及限制见结果报告。\n'
    (folder/'00_提交说明.txt').write_text(text,encoding='utf-8-sig')
    with zipfile.ZipFile(folder/'代码与验证报告.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.glob('*.py'): z.write(p,p.name)
        for name in ('fovea','tests'):
            for p in (ROOT/name).glob('*.py'): z.write(p,p.relative_to(ROOT))
        for name,_,_,_ in experiments:
            for p in (ROOT/'reports'/name).glob('*.json'): z.write(p,p.relative_to(ROOT))
        z.write(report,report.name)
        z.write(REPORT/'tests.log','reports/rcnn_v10/tests.log')
        z.write(ROOT/'reports/submission_budget_20261027.json','reports/submission_budget_20261027.json')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file() and p.name!='文件校验.json'}
    write(folder/'文件校验.json',hashes)
    print(json.dumps(dict(folder=str(folder),recommend_new=decision['recommend_new'],oof_rmse=after['rmse'],hashes=hashes),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
