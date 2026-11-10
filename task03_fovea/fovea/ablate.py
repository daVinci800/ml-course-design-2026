"""Compare four predeclared policies using existing held-out checkpoints only."""
import hashlib
from pathlib import Path
import torch
from .data import stage_settings
from .evaluate import summarize
from .inference import load_checkpoint,predict_records
from .policies import POLICIES,policy_settings,refinement_windows
from .prepare import write_json


def ablate(manifest,run_dir,output,device='cuda'):
    from .train import manifest_hash
    device=torch.device(device)
    records=[r for r in manifest['records'] if r['split']=='train']
    manifest_id=manifest_hash(manifest)
    predictions={name:{} for name in POLICIES}
    source=[]
    for fold in range(manifest['n_folds']):
        valid=[r for r in records if r['fold']==fold]
        directory=Path(run_dir)/f'fold_{fold}'
        cp_path,fp_path=directory/'coarse_best.pt',directory/'fine_best.pt'
        coarse,cp=load_checkpoint(cp_path,device)
        torch.set_num_threads(cp['config'].get('cpu_threads',4))
        if cp['manifest_hash']!=manifest_id or cp['fold']!=fold or cp['stage']!='coarse':
            raise ValueError('Wrong coarse checkpoint manifest/fold/stage.')
        if cp.get('presence_enabled'):
            raise ValueError('Visibility classifier requires real-negative calibration.')
        coarse_settings=stage_settings(cp['config'],'coarse')
        # One identical coarse estimate for all candidates; no test images.
        cpred=predict_records(coarse,valid,'coarse',coarse_settings,device,tta=True)
        del coarse
        fine,fp=load_checkpoint(fp_path,device)
        coarse_hash=hashlib.sha256(cp_path.read_bytes()).hexdigest()
        if fp['manifest_hash']!=manifest_id or fp['fold']!=fold or fp['stage']!='fine' or fp['coarse_sha256']!=coarse_hash:
            raise ValueError('Fine checkpoint does not match its coarse fold.')
        if fp.get('presence_enabled'):
            raise ValueError('Visibility classifier requires real-negative calibration.')
        for roi in ('fixed','adaptive'):
            settings=policy_settings(stage_settings(fp['config'],'fine'),'integral_'+roi)
            centers,sides=refinement_windows(cpred,settings)
            decoded=predict_records(fine,valid,'fine',settings,device,centers=centers,crop_sides=sides,tta=True,decoder_methods=('integral','dark'))
            coverage=sum(max(abs(centers[r['id']][j]-r['xy'][j]) for j in range(2))<sides[r['id']]/2 for r in valid)/len(valid)
            for decoder,pred in decoded.items():
                policy=decoder+'_'+roi
                predictions[policy].update(pred)
                write_json(Path(output)/policy/f'fold_{fold}'/'fine_oof.json',dict(
                    stage='fine',fold=fold,manifest_hash=manifest_id,tta=True,policy=policy,
                    predictions=pred,crop_coverage=coverage,crop_sides=sides))
        source.append(dict(fold=fold,coarse_sha256=coarse_hash,fine_sha256=hashlib.sha256(fp_path.read_bytes()).hexdigest()))
        del fine
        if device.type=='cuda':
            torch.cuda.empty_cache()
        print(f'Inference-only ablation completed fold {fold}.',flush=True)
    report=dict(manifest_hash=manifest_id,checkpoints=source,training_started=False,optimizer_updates=0,
                policies={name:summarize(records,p) for name,p in predictions.items()},
                warnings=['These are exploratory comparisons on checkpoint-selection folds; policy selection adds optimism.',
                          'Do not call the selected score an unbiased test RMSE or tune to public/test labels.',
                          'Adaptive spread is a heuristic, not calibrated uncertainty; confident mistakes can remain.'])
    write_json(Path(output)/'comparison.json',report)
    for name,metrics in report['policies'].items():
        print(f'{name}: OOF RMSE={metrics["rmse"]:.3f}px, p95 ED={metrics["p95_euclidean"]:.3f}px')
    return report
