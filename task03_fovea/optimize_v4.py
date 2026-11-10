"""Two predeclared full-schedule experiments, strict OOF selection and inference."""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute-training',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Continue saved folds and optimizer state, excluding paused time from budget.')
    args=parser.parse_args()
    if not args.execute_training:
        parser.error('Explicit --execute-training is required.')
    root=Path(__file__).resolve().parent
    os.chdir(root)
    sys.stdout.reconfigure(encoding='utf8')
    sys.stderr.reconfigure(encoding='utf8')
    os.environ.setdefault('TORCH_HOME',str(root.parent.parent/'work'/'pretrained_cache'))
    import numpy as np
    import torch
    from fovea.prepare import read_manifest,write_json
    from fovea.train import train_folds,manifest_hash,TrainingPaused
    from fovea.evaluate import collect_oof,summarize
    from fovea.ensemble import candidate_recipes,combine_predictions
    from fovea.inference import predict_manifest,make_submission
    job=root/'runs/optimization_v4'
    report_dir=root/'reports/optimization_v4'
    job.mkdir(parents=True,exist_ok=True)
    report_dir.mkdir(parents=True,exist_ok=True)
    status_path=job/'status.json'
    previous=None
    if args.resume:
        previous=json.loads(status_path.read_text(encoding='utf8'))
        if previous['phase'] not in ('paused','failed'):
            raise RuntimeError('Resume requires an explicitly paused or failed job.')
    elif status_path.exists():
        raise FileExistsError('This optimization job already exists; inspect it instead of launching twice.')
    lock=(job/'active.lock').open('x',encoding='utf8')
    lock.write(str(os.getpid()));lock.flush()
    started=time.monotonic()
    prior_active=float(previous.get('active_seconds',previous.get('wall_seconds',0))) if previous else 0.
    state=dict(previous) if previous else dict(started_at=datetime.datetime.now().astimezone().isoformat(),
               baseline_kaggle_score_user_reported=20.26463,total_soft_budget_seconds=16200)
    state.update(pid=os.getpid(),resumed_at=datetime.datetime.now().astimezone().isoformat() if previous else None)
    pause_file=job/'pause.request'
    os.environ['FOVEA_PAUSE_FILE']=str(pause_file)
    if args.resume:pause_file.unlink(missing_ok=True)
    def active_seconds():return prior_active+time.monotonic()-started
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.datetime.now().astimezone().isoformat(),
                     wall_seconds=active_seconds(),active_seconds=active_seconds(),**extra)
        temporary=status_path.with_suffix('.tmp')
        write_json(temporary,state);temporary.replace(status_path)
        print(json.dumps(dict(phase=phase,**extra),ensure_ascii=False),flush=True)
    awake=False
    try:
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable.')
        torch.set_num_threads(4)
        if os.name=='nt': awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        manifest=read_manifest('prepared/manifest.json')
        registry=dict(baseline=root/'runs/accuracy_5h',resnet=root/'runs/resnet34_v4',convnext=root/'runs/convnext_v4')
        recipes=candidate_recipes()
        baseline_file=root/'submissions/submission_accuracy_5h.csv'
        if not args.resume:write_json(report_dir/'predeclared_plan.json',dict(recipes=recipes,manifest_hash=manifest_hash(manifest),
            baseline_kaggle_score_user_reported=20.26463,baseline_submission_sha256=hashlib.sha256(baseline_file.read_bytes()).hexdigest(),
            experiments=['configs/resnet34_v4.json','configs/convnext_v4.json'],
            selection='Minimum original-pixel grouped OOF RMSE among predeclared recipes; no leaderboard tuning.',
            source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'fovea').glob('*.py'))}))
        for index,name in enumerate(('resnet','convnext')):
            config_path=root/'configs'/('resnet34_v4.json' if name=='resnet' else 'convnext_v4.json')
            config=json.loads(config_path.read_text(encoding='utf8'))
            # One total training allowance, including cache/loading/validation.
            # Leave ten minutes for final reports and inference.
            remaining=15600-active_seconds()
            allowance=int(remaining/(2-index))
            if allowance<180: raise TimeoutError('Insufficient time for another complete experiment.')
            registry[name].mkdir(parents=True,exist_ok=True)
            saved_config=registry[name]/'requested_config.json'
            if args.resume and saved_config.exists():
                config=json.loads(saved_config.read_text(encoding='utf8'))
                allowance=config['time_budget_seconds']
            else:
                config['time_budget_seconds']=allowance
                write_json(saved_config,config)
            status('training',experiment=name,run_dir=str(registry[name]),experiment_budget_seconds=allowance)
            train_folds(manifest,config,list(range(manifest['n_folds'])),registry[name],'cuda',resume=args.resume)
            torch.cuda.empty_cache()
        status('evaluating_predeclared_ensembles')
        components={}
        for name,directory in registry.items():
            for stage in ('coarse','fine'):
                records,pred=collect_oof(manifest,directory,stage)
                components[name+'_'+stage]=pred
        baseline_stats=summarize(records,components['baseline_fine'])
        results={}
        for name,weights in recipes.items():
            pred=combine_predictions(components,weights)
            metrics=summarize(records,pred)
            metrics['folds_better_than_baseline']=sum(metrics['per_fold_rmse'][str(f)]<baseline_stats['per_fold_rmse'][str(f)] for f in range(5))
            results[name]=dict(weights=weights,metrics=metrics)
            write_json(report_dir/(name+'_oof.json'),dict(manifest_hash=manifest_hash(manifest),predictions=pred,metrics=metrics,weights=weights))
            print(f'OOF {name}: RMSE={metrics["rmse"]:.4f}, p95 ED={metrics["p95_euclidean"]:.3f}, improved folds={metrics["folds_better_than_baseline"]}/5',flush=True)
        selected=min(results,key=lambda name:(results[name]['metrics']['rmse'],results[name]['metrics']['p95_euclidean']))
        chosen=results[selected]
        selection=dict(selected=selected,**chosen,baseline_oof_rmse=baseline_stats['rmse'],candidates=results,
            improvement_px=baseline_stats['rmse']-chosen['metrics']['rmse'],baseline_kaggle_score_user_reported=20.26463,
            warning='OOF checkpoint/recipe selection is optimistic. Public score was not used for recipe selection; the new test score is unknown.')
        write_json(report_dir/'selection.json',selection)
        status('predicting_test',selected=selected,selected_oof_rmse=chosen['metrics']['rmse'],oof_improvement_px=selection['improvement_px'])
        test_components={}
        component_outputs=report_dir/'test_components'
        for key in chosen['weights']:
            name,stage=key.rsplit('_',1)
            directory=registry[name]
            pairs=[(directory/f'fold_{f}'/'coarse_best.pt',directory/f'fold_{f}'/'fine_best.pt' if stage=='fine' else None) for f in range(5)]
            test_components[key]=predict_manifest(manifest,pairs,component_outputs/(key+'.csv'),'cuda',tta=True,stage=stage,policy='integral_fixed')
        predictions=combine_predictions(test_components,chosen['weights'])
        test=[r for r in manifest['records'] if r['split']=='test']
        submission=root/'submissions/submission_optimized_v4.csv'
        make_submission(manifest['template'],predictions,submission,{r['id']:(r['width'],r['height']) for r in test})
        write_json(submission.with_suffix('.diagnostics.json'),dict(predictions=predictions,weights=chosen['weights'],selected=selected,
            manifest_hash=manifest_hash(manifest),oof_rmse=chosen['metrics']['rmse'],warning='No actual test ground truth is available.'))
        desktop_file=None
        if selection['improvement_px']>0:
            folder=Path('C:/Users/Lenovo/Desktop/2026task3_提交文件')
            if folder.is_dir():
                desktop_file=folder/submission.name
                if desktop_file.exists(): raise FileExistsError(f'Do not overwrite an existing submission: {desktop_file}')
                shutil.copyfile(submission,desktop_file)
                if hashlib.sha256(desktop_file.read_bytes()).digest()!=hashlib.sha256(submission.read_bytes()).digest():
                    raise RuntimeError('Desktop copy hash mismatch.')
        summary=f'''# 第二轮优化实际结果\n\n原 Kaggle 成绩（用户报告）：20.26463；原 OOF RMSE：{baseline_stats['rmse']:.6f}。\n\n两组新增训练已完成；所选策略：{selected}，新 OOF RMSE：{chosen['metrics']['rmse']:.6f}，相对原 OOF 改善：{selection['improvement_px']:.6f} 像素。\n\n新提交文件：`submissions/{submission.name}`。尚未上传 Kaggle，未知新的测试集成绩；验证选择本身带有乐观偏差。旧提交保留。\n\n详细指标、固定候选与各折表现见 `reports/optimization_v4/selection.json`。总运行耗时约 {(time.monotonic()-started)/60:.2f} 分钟。\n'''
        (root/'第二轮优化结果.md').write_text(summary,encoding='utf8')
        status('completed',submission=str(submission),desktop_submission=str(desktop_file) if desktop_file else None,
            selection_report=str(report_dir/'selection.json'),submission_sha256=hashlib.sha256(submission.read_bytes()).hexdigest())
    except TrainingPaused as exc:
        status('paused',paused_at=datetime.datetime.now().astimezone().isoformat(),pause_reason=str(exc))
    except BaseException as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        lock.close();(job/'active.lock').unlink(missing_ok=True)
        if awake: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__': main()
