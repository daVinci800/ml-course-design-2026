"""Generate the final local validation report and a source-only reproducibility archive."""
import json
from pathlib import Path
import zipfile


def build_report():
    from final_round import ROOT,REPORT,EXPERIMENTS,inputs,read
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    completion=read(REPORT/'completion.json')
    selection=read(REPORT/'selection.json')
    manifest,records,fallback,baseline=inputs()
    selected={int(k):v for k,v in read(REPORT/'selected_oof.json')['predictions'].items()}
    from fovea.evaluate import summarize
    metrics={'previous':summarize(records,baseline)}
    for name in EXPERIMENTS: metrics[name]=read(REPORT/f'{name}_deployed_oof.json')['metrics']
    dense_metrics={name:read(REPORT/f'{name}_dense_oof.json')['metrics'] for name in (*EXPERIMENTS,'previous')}
    metrics['selected']=selection['metrics']
    labels={'previous':'已得约14分的旧方案','fov':'去黑边，第一个种子','roi':'局部高分辨率',
            'full_highres':'保留全图，1280高分辨率','full_seed2':'原全图，第二个种子','selected':'本次最终选择'}
    before=metrics['previous']['rmse'];after=metrics['selected']['rmse']
    improvement=(before-after)/before*100
    lines=['# 最后一轮优化结果','',
       f'新方案本地五折 OOF RMSE：**{after:.5f}**；旧方案：**{before:.5f}**，变化 **{improvement:.2f}%**。',
       '', '用户报告旧文件的 Kaggle 成绩约为 **14**。新文件没有自动提交，真实新比赛分数未知；本地改善不能直接换算成排行榜分数。',
       '',f'本轮从方案登记到完成推理约 **{completion["wall_minutes"]:.1f} 分钟**。这个时间包含实验、验证、推理和暂停恢复测试，不包含以前的训练。',
       '', '## 同一套五折的原图像素坐标验证','',
       '| 方案 | RMSE | X RMSE | Y RMSE | 95%分位欧氏误差 |',
       '|---|---:|---:|---:|---:|']
    for k,m in metrics.items():
        lines.append(f'| {labels[k]} | {m["rmse"]:.5f} | {m["rmse_x"]:.3f} | {m["rmse_y"]:.3f} | {m["p95_euclidean"]:.3f} |')
    lines += ['', '固定候选框加权中心策略的独立对照：','',
              '| 方案 | 加权中心 RMSE |','|---|---:|']
    for k,m in dense_metrics.items():lines.append(f'| {labels[k]} | {m["rmse"]:.5f} |')
    lines += ['',f'最终配方：`{selection["selected"]}`。所有图片使用相同的权重：',
              '', '```json',json.dumps(selection['weights'],indent=2,ensure_ascii=False),'```',
              '', '| 验证折 | 旧方案 RMSE | 本次 RMSE |','|---|---:|---:|']
    for f in range(5):
        lines.append(f'| {f} | {metrics["previous"]["per_fold_rmse"][str(f)]:.5f} | {metrics["selected"]["per_fold_rmse"][str(f)]:.5f} |')
    lines += ['', '## 稳健性和边界','',
       f'只用其余四折选择融合配方，再评估被留出的一折，合并后的配方选择诊断 RMSE 为 **{selection["recipe_crossfit_metrics"]["rmse"]:.5f}**，高于旧方案的 **{before:.5f}**。这说明融合配方对这 80 张训练图的划分较敏感，本次最低 OOF 分数尚不足以证明在新数据上稳定提升。该诊断仍不能消除检查点选择的乐观偏差，不能视作新的独立测试集。',
       '', '80 张训练图均为可见中央凹，缺少不可见的真实负例。本轮没有宣称已解决或验证不可见识别，也没有凭空设置输出 (0,0) 的阈值。',
       '', '训练重复图保持在同一折，局部模型验证的裁剪中心来自对应折的旧检测器。选择只基于训练图 OOF，随后才推理 20 张测试图。没有手工修正测试坐标，也没有自动提交 Kaggle。',
       '', '## 文件和复现','',
       '- 新提交：`submissions/submission_final_round_v7.csv`，40 行，已核对模板顺序、有限数值和坐标边界。',
       '- 已得约14分的原文件：`submissions/submission_baseline_kaggle14_20261021.csv`，内容保留。',
       f'- 新文件 SHA256：`{completion["sha256"]}`。',
       '- 训练与推理：`final_round.py`；串行队列与预算：`run_final_round.py`。',
       '- 每个模型的完整断点、选中权重、训练日志：`runs/final_round_v7/`。',
       '- 保存权重的验证复跑：`reports/final_round_v7/checkpoint_replay_verification.json`。',
       '- 暂停续训已做实际测试，保存模型、EMA、优化器、调度器、scaler、随机状态；参见 `resume_verification.json`。',
       '- 包括坐标几何、候选框融合、自动回退、分组隔离以及 Windows 写入故障恢复和进程退出检测，测试共53项通过。',
       '', '![五折验证比较](reports/final_round_v7/validation_comparison.png)', '']
    (ROOT/'最后一轮优化结果.md').write_text('\n'.join(lines),encoding='utf8')
    old_ed=np.array([np.linalg.norm(np.asarray(baseline[r['id']]['xy'])-r['xy']) for r in records])
    new_ed=np.array([np.linalg.norm(np.asarray(selected[r['id']]['xy'])-r['xy']) for r in records])
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    for values,label in [(old_ed,'Previous'),(new_ed,'Selected')]:
        axes[0].plot(np.sort(values),np.arange(1,len(values)+1)/len(values),label=label,linewidth=2)
    axes[0].set(xlabel='Euclidean error (original pixels)',ylabel='Fraction of validation images',title='Out-of-fold error distribution')
    axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].scatter(old_ed,new_ed,c=[r['fold'] for r in records],cmap='tab10',s=30)
    limit=max(old_ed.max(),new_ed.max())*1.05
    axes[1].plot([0,limit],[0,limit],color='gray',linestyle='--')
    axes[1].set(xlabel='Previous error (pixels)',ylabel='Selected error (pixels)',title='Each training image; below line improves')
    for i in np.argsort(old_ed)[-4:]:axes[1].annotate(str(records[i]['id']),(old_ed[i],new_ed[i]),fontsize=8)
    x=np.arange(5)
    axes[2].bar(x-.18,[metrics['previous']['per_fold_rmse'][str(f)] for f in range(5)],width=.36,label='Previous')
    axes[2].bar(x+.18,[metrics['selected']['per_fold_rmse'][str(f)] for f in range(5)],width=.36,label='Selected')
    axes[2].set(xticks=x,xlabel='Validation fold',ylabel='Coordinate RMSE (original pixels)',title=f'OOF RMSE: {before:.2f} -> {after:.2f}')
    axes[2].legend()
    fig.tight_layout();fig.savefig(REPORT/'validation_comparison.png',dpi=170);plt.close(fig)
    archive=ROOT.parent/'fovea_final_round_v7_code.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('fovea','tests','configs'):
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts: z.write(p,p.relative_to(ROOT))
        for pattern in ('*.py','*.md','requirements*.txt'):
            for p in ROOT.glob(pattern): z.write(p,p.relative_to(ROOT))
        for p in REPORT.glob('*.json'): z.write(p,p.relative_to(ROOT))
        z.write(REPORT/'validation_comparison.png','reports/final_round_v7/validation_comparison.png')
    print(f'Report and source archive written: {archive}',flush=True)


if __name__=='__main__':build_report()
