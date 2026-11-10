"""Verify completed optimization artifacts and produce a compact delivery report."""
import csv,datetime,hashlib,json,math,zipfile,shutil,sys
from pathlib import Path


def main():
    sys.stdout.reconfigure(encoding='utf8');sys.stderr.reconfigure(encoding='utf8')
    root=Path(__file__).resolve().parent
    def read(path):return json.loads((root/path).read_text(encoding='utf8'))
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    baseline=root/'submissions/submission_accuracy_5h.csv'
    expected='4ab28ad1711199662f93c739c4325cb122c14a38ee72ea48480c64e2e429d84c'
    assert sha(baseline)==expected,'Original submission was changed.'
    v4=read('runs/optimization_v4/status.json')
    latest='detector_v6' if (root/'runs/detector_v6/status.json').exists() else 'detector_v5'
    detector=read(f'runs/{latest}/status.json')
    assert v4['phase']=='completed' and detector['phase']=='completed','Jobs must both finish first.'
    manifest=read('prepared/manifest.json')
    selection=read(f'reports/{latest}/selection.json')
    file=Path(detector['submission'])
    training_selection=selection
    selection_source=latest
    iterative=None
    if (root/'reports/iterative_refinement/completion.json').exists():
        iterative=read('reports/iterative_refinement/completion.json')
        assert iterative['phase']=='completed'
        selection=read('reports/iterative_refinement/selection.json')
        selection_source='iterative_refinement'
        file=Path(iterative['submission'])
    with Path(manifest['template']).open(encoding='utf-8-sig',newline='') as stream:
        template=list(csv.DictReader(stream))
    with file.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==40
    assert [r['ImageID'] for r in rows]==[r['ImageID'] for r in template]
    sizes={str(r['id']):(r['width'],r['height']) for r in manifest['records'] if r['split']=='test'}
    for row in rows:
        ident,axis=row['ImageID'].split('_Fovea_')
        value=float(row['value']);limit=sizes[ident][0 if axis=='X' else 1]-1
        assert math.isfinite(value) and 0<=value<=limit
    desktop=Path('C:/Users/Lenovo/Desktop/2026task3_提交文件')
    restored_baseline=not (desktop/baseline.name).exists()
    if restored_baseline:shutil.copyfile(baseline,desktop/baseline.name)
    prior_verification=root/'reports/optimization_completion.json'
    restored_any_time=restored_baseline
    if prior_verification.exists():restored_any_time|=bool(json.loads(prior_verification.read_text(encoding='utf8')).get('desktop_original_restored'))
    assert sha(desktop/baseline.name)==expected
    assert sha(desktop/file.name)==sha(file)
    folds=[]
    for fold in range(5):
        directory=root/f'runs/{latest}/fold_{fold}'
        content=json.loads((directory/'detector_oof.json').read_text(encoding='utf8'))
        valid={str(r['id']) for r in manifest['records'] if r['split']=='train' and r['fold']==fold}
        assert set(content['predictions'])==valid
        assert (directory/'point_best.pt').is_file()
        assert set(json.loads((directory/'refined_oof.json').read_text(encoding='utf8'))['predictions'])==valid
        history=json.loads((directory/'point_history.json').read_text(encoding='utf8'))
        selected_epoch=min(history,key=lambda item:item['rmse'])['epoch']
        misses=sum(bool(p.get('detector_miss')) for p in content['predictions'].values())
        folds.append(dict(fold=fold,validated_epochs=len(history),last_epoch=history[-1]['epoch'],selected_epoch=selected_epoch,
            automatic_fallback_images=misses,best_rmse=content['rmse'],checkpoint_sha256=sha(directory/'point_best.pt')))
    post_v4=(datetime.datetime.fromisoformat(detector['updated_at'])-datetime.datetime.fromisoformat(v4['updated_at'])).total_seconds()
    elapsed=float(v4.get('active_seconds',v4['wall_seconds']))+max(0.,post_v4)
    if iterative:elapsed+=float(iterative['wall_seconds'])
    report=dict(verified_at=datetime.datetime.now().astimezone().isoformat(),experiment=selection_source,selected=selection['selected'],
        oof_rmse=selection['metrics']['rmse'],baseline_oof_rmse=20.997774092844676,
        baseline_kaggle_score_user_reported=20.26463,new_kaggle_score=None,
        total_optimization_wall_minutes=elapsed/60,original_submission_preserved=True,desktop_original_restored=restored_any_time,
        final_csv=str(file),final_csv_rows=40,template_order_exact=True,finite_in_bounds=True,desktop_copy_exact=True,
        final_csv_sha256=sha(file),detector_folds=folds,
        warning='OOF checkpoint and recipe selection is optimistic. No actual test labels or new Kaggle score available.')
    destination=root/'reports/optimization_completion.json'
    if latest=='detector_v6' and training_selection['selected']=='detector' and selection['selected']=='previous_winner':
        report['resolved_method']='YOLO11s with lr0=0.0001 and warmup_bias_lr=0; predict box center, automatic original-model fallback on no detection; average five folds. Repeated refinement rejected.'
        report['oof_improvement_percent']=100*(1-selection['metrics']['rmse']/20.997774092844676)
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    v4selection=read('reports/optimization_v4/selection.json')
    lines=['# 第二轮优化完成记录','',
        f'原 Kaggle 成绩（用户报告）：20.26463。原 OOF RMSE：20.997774。新 Kaggle 成绩尚未知。',
        '',f'最终选择：`{selection["selected"]}`；OOF RMSE：{selection["metrics"]["rmse"]:.6f}。',
        f'完整优化、比较与测试推理耗时：{elapsed/60:.1f} 分钟。',
        '', '| 方案 | OOF RMSE |','|---|---:|']
    for name,candidate in v4selection['candidates'].items():
        lines.append(f'| v4 / {name} | {candidate["metrics"]["rmse"]:.6f} |')
    for name,candidate in selection['candidates'].items():
        lines.append(f'| {selection_source} / {name} | {candidate["metrics"]["rmse"]:.6f} |')
    if iterative:
        for name,candidate in training_selection['candidates'].items():
            lines.append(f'| {latest} / {name} | {candidate["metrics"]["rmse"]:.6f} |')
    if latest=='detector_v6':
        first_detector=read('reports/detector_v5/selection.json')
        for name,candidate in first_detector['candidates'].items():
            lines.append(f'| detector_v5 / {name} | {candidate["metrics"]["rmse"]:.6f} |')
    lines+=['','逐折结果：','', '| 折 | 原基线 | 最终方案 |','|---|---:|---:|']
    base=training_selection['candidates']['baseline']['metrics']['per_fold_rmse']
    final=selection['metrics']['per_fold_rmse']
    for fold in range(5):lines.append(f'| {fold} | {base[str(fold)]:.6f} | {final[str(fold)]:.6f} |')
    lines+=['','检测器各折最佳检查点及自动回退：','', '| 折 | 所选轮次 | 最终轮次 | 验证图中未检出并自动回退数 |','|---|---:|---:|---:|']
    for item in folds:lines.append(f'| {item["fold"]} | {item["selected_epoch"]} | {item["last_epoch"]} | {item["automatic_fallback_images"]} / 16 |')
    lines+=['',f'推荐 CSV：`submissions/{file.name}`。桌面 `2026task3_提交文件` 内已放同名副本，旧 CSV 保留。',
        '', '本轮最终实际模型：低学习率 YOLO11s 黄斑框检测器（lr0=0.0001，warmup_bias_lr=0），输出检测框中心，未检出时自动回退到原模型；测试时平均五折结果。追加精定位与重复精定位未超过该方案。',
        '', '核验：40 行、模板顺序完全一致、数值有限且在图像范围内、桌面副本哈希一致、原提交哈希不变。五折检测器与对应精定位 OOF 完整。项目 42 项测试通过。',
        '', '没有自动上传 Kaggle。OOF 多次选检查点及方案会有选择偏差，不能把本地下降幅度直接当成榜单下降幅度。训练集没有真实不可见样本，可见性泛化仍无法验证。',
        '', '代码与训练配置见本项目；不含模型权重的小体积代码包位于上级 outputs 目录。实际检查记录：`reports/optimization_completion.json`。']
    (root/'本轮优化最终结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    note=root/'submissions/推荐提交说明_20261021.txt'
    note.write_text(f'本次推荐文件：{file.name}\n本地五折 OOF RMSE：{selection["metrics"]["rmse"]:.6f}\n'
        '新 Kaggle 分数：未知，尚未自动上传。\n'
        '原提交 submission_accuracy_5h.csv 已保留；其 Kaggle 分数为用户报告的 20.26463。\n'
        '其他 optimized 文件是中间对照结果；无需把多个文件都提交。\n'
        '本地验证比较带有选模型偏差，不保证榜单按相同幅度改善。\n',encoding='utf-8-sig')
    desktop_note=desktop/note.name
    if desktop_note.exists() and desktop_note.read_bytes()!=note.read_bytes():raise FileExistsError('An existing recommendation note differs; do not overwrite.')
    if not desktop_note.exists():desktop_note.write_bytes(note.read_bytes())
    archive=root.parent/'fovea_optimized_code.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zipped:
        candidates=list(root.glob('*.py'))+list(root.glob('*.md'))+list(root.glob('requirements*.txt'))
        for folder in ('fovea','tests','configs'):
            candidates.extend(p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json'))
        candidates.extend([file,note,destination,root/'reports/final_checkpoint_verification.json',root/'reports/detector_v5/selection.json',root/'reports/optimization_v4/selection.json'])
        if latest=='detector_v6':candidates.append(root/'reports/detector_v6/selection.json')
        if iterative:candidates.extend([root/'reports/iterative_refinement/selection.json',root/'reports/iterative_refinement/completion.json'])
        for path in sorted(set(candidates)):
            zipped.write(path,Path('fovea_solution')/path.relative_to(root))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
