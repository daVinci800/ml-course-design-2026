"""Explicitly authorized local training, held-out policy comparison and prediction.

Run with --execute-training. No Kaggle upload is performed. Windows workers
import this module safely because work is guarded by __main__.
"""
import argparse
import ctypes
import datetime
import json
import os
import sys
from pathlib import Path
import time
import traceback


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute-training',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--config',default='configs/accuracy_5h.json')
    parser.add_argument('--run-dir',default='runs/accuracy_5h')
    args=parser.parse_args()
    if not args.execute_training and not args.prepare_only:
        parser.error('Explicit --execute-training or --prepare-only is required.')
    root=Path(__file__).resolve().parent
    sys.stdout.reconfigure(encoding='utf8')
    sys.stderr.reconfigure(encoding='utf8')
    os.chdir(root)
    os.environ.setdefault('TORCH_HOME',str(root.parent.parent/'work'/'pretrained_cache'))
    os.environ.setdefault('PYTHONUNBUFFERED','1')
    import torch
    import torchvision
    from torchvision.models import ResNet34_Weights
    from fovea.prepare import read_manifest,write_json
    run_dir=Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True,exist_ok=True)
    status_path=run_dir/'status.json'
    started=time.monotonic()
    state=dict(pid=os.getpid(),started_at=datetime.datetime.now().astimezone().isoformat(),
               run_dir=str(run_dir),config=args.config,training_started=False)
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.datetime.now().astimezone().isoformat(),
                     wall_seconds=time.monotonic()-started,**extra)
        temporary=status_path.with_suffix('.tmp')
        write_json(temporary,state)
        temporary.replace(status_path)
        print(json.dumps(dict(phase=phase,**extra),ensure_ascii=False),flush=True)
    awake=False
    lock=None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; CPU fallback is deliberately disabled.')
        config=json.loads(Path(args.config).read_text(encoding='utf8'))
        torch.set_num_threads(config.get('cpu_threads',4))
        info=dict(torch=torch.__version__,torchvision=torchvision.__version__,cuda=torch.version.cuda,
                  device=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0))
        probe=torch.ones(32,32,device='cuda')
        assert float((probe@probe).mean())==32.
        torch.cuda.synchronize()
        del probe
        write_json(run_dir/'environment.json',info)
        status('preparing_weights',environment=info)
        # Download official weights here, before starting the training budget.
        weights=ResNet34_Weights.DEFAULT.get_state_dict(progress=False,check_hash=True)
        del weights
        if args.prepare_only:
            status('environment_ready')
            return
        lock_path=run_dir/'active.lock'
        lock=lock_path.open('x',encoding='utf8')
        lock.write(str(os.getpid())); lock.flush()
        if os.name=='nt':
            awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        manifest=read_manifest(root/'prepared/manifest.json')
        from fovea.train import train_folds
        status('training',training_started=True)
        train_folds(manifest,config,list(range(manifest['n_folds'])),run_dir,'cuda',args.resume)
        from fovea.evaluate import evaluate
        from fovea.ablate import ablate
        from fovea.inference import predict_manifest
        status('evaluating_coarse_and_fine')
        report_dir=root/'reports'/'accuracy_run'
        coarse=evaluate(manifest,[run_dir],report_dir/'coarse_metrics.json',stage='coarse')
        evaluate(manifest,[run_dir],report_dir/'fine_metrics.json',stage='fine')
        status('comparing_inference_policies')
        comparison=ablate(manifest,run_dir,report_dir/'policy_ablation','cuda')
        candidates=[dict(stage='coarse',policy='checkpoint',metrics=coarse)]
        candidates.extend(dict(stage='fine',policy=name,metrics=metrics) for name,metrics in comparison['policies'].items())
        selected=min(candidates,key=lambda c:(c['metrics']['rmse'],c['metrics']['p95_euclidean']))
        selection=dict(selected=selected,candidates=candidates,
            criterion='Minimum aggregate original-pixel held-out RMSE; p95 Euclidean error breaks exact ties.',
            warning='Checkpoint and policy selection make OOF optimistic; this is not a test leaderboard score.')
        write_json(report_dir/'selection.json',selection)
        status('predicting_test',selected_stage=selected['stage'],selected_policy=selected['policy'],
               selected_oof_rmse=selected['metrics']['rmse'])
        pairs=[(run_dir/f'fold_{fold}'/'coarse_best.pt',run_dir/f'fold_{fold}'/'fine_best.pt' if selected['stage']=='fine' else None)
               for fold in range(manifest['n_folds'])]
        submission=root/'submissions'/'submission_accuracy_5h.csv'
        predict_manifest(manifest,pairs,submission,'cuda',tta=True,stage=selected['stage'],policy=selected['policy'])
        status('completed',submission=str(submission),selection_report=str(report_dir/'selection.json'))
    except BaseException as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        if lock is not None:
            lock.close()
            (run_dir/'active.lock').unlink(missing_ok=True)
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    main()
