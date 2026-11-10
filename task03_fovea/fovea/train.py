import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import FoveaDataset,EpochSampler,worker_init,stage_settings
from .model import Localizer
from .loss import localization_loss
from .inference import predict_records,load_checkpoint
from .prepare import write_json
from .budget import RunBudget,fit_epoch_limit
from .cache import build_cache
from .policies import refinement_windows


class TrainingPaused(RuntimeError):
    """A requested pause after saving a complete epoch training state."""


def pause_requested(run_dir):
    global_request=os.environ.get('FOVEA_PAUSE_FILE')
    return (Path(run_dir)/'pause.request').exists() or bool(global_request and Path(global_request).exists())


def seed_everything(seed,deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=not deterministic
    torch.backends.cudnn.allow_tf32=not deterministic
    torch.backends.cuda.matmul.allow_tf32=not deterministic
    torch.use_deterministic_algorithms(deterministic,warn_only=True)


def manifest_hash(manifest):
    portable=dict(version=manifest['version'],seed=manifest['seed'],n_folds=manifest['n_folds'],
                  template_sha256=manifest['template_sha256'],
                  records=[{k:v for k,v in r.items() if k!='path'} for r in manifest['records']])
    return hashlib.sha256(json.dumps(portable,sort_keys=True).encode()).hexdigest()


def rmse(records,predictions):
    errors=np.array([np.array(predictions[r['id']]['xy'])-r['xy'] for r in records])
    return float(np.sqrt(np.mean(errors**2)))


def save_atomic(value,path):
    path=Path(path)
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary)
    os.replace(temporary,path)


def checkpoint_metadata(config,manifest,fold,stage,presence_enabled,coarse_path=None):
    return dict(format='fovea-v1',config=config,manifest_hash=manifest_hash(manifest),
                fold=fold,stage=stage,presence_enabled=presence_enabled,
                coarse_sha256=hashlib.sha256(Path(coarse_path).read_bytes()).hexdigest() if coarse_path else None)


def epoch_sigma(settings,epoch):
    progress=min(1.,max(0.,epoch/max(1,settings['epochs']-1)))
    return settings['sigma']*(1-progress)+settings.get('sigma_end',settings['sigma'])*progress


def early_stopping_ready(bad,settings,completed_epochs):
    return bad>=settings['patience'] and completed_epochs>=settings.get('min_epochs',0)


@torch.no_grad()
def update_ema(ema,model,decay):
    target=list(ema.parameters())
    torch._foreach_mul_(target,decay)
    torch._foreach_add_(target,list(model.parameters()),alpha=1-decay)
    for a,b in zip(ema.buffers(),model.buffers()):
        a.copy_(b)


def stage_complete(manifest,config,fold,stage,run_dir):
    directory=Path(run_dir)/f'fold_{fold}'
    path=directory/f'{stage}_done.json'
    if not path.exists():
        return False
    state=json.loads(path.read_text(encoding='utf-8'))
    if state['config']!=config or state['manifest_hash']!=manifest_hash(manifest):
        raise ValueError('Completed stage does not match the requested configuration/data.')
    best=directory/f'{stage}_best.pt'
    if not best.exists() or not (directory/f'{stage}_oof.json').exists():
        return False
    if hashlib.sha256(best.read_bytes()).hexdigest()!=state['best_sha256']:
        raise ValueError('Completed checkpoint was modified.')
    if stage=='fine' and hashlib.sha256((directory/'coarse_best.pt').read_bytes()).hexdigest()!=state['coarse_sha256']:
        raise ValueError('Coarse checkpoint changed after refinement finished.')
    return True


def train_stage(manifest,config,fold,stage,run_dir,device,resume=False,deadline=None,budget=None):
    records=[r for r in manifest['records'] if r['split']=='train']
    train=[r for r in records if r['fold']!=fold]
    valid=[r for r in records if r['fold']==fold]
    if not train or not valid:
        raise ValueError('Empty training or validation fold.')
    if {r['group'] for r in train}&{r['group'] for r in valid}:
        raise ValueError('Training/validation group leakage.')
    fold_dir=Path(run_dir)/f'fold_{fold}'
    fold_dir.mkdir(parents=True,exist_ok=True)
    best_path=fold_dir/f'{stage}_best.pt'
    last_path=fold_dir/f'{stage}_last.pt'
    history_path=fold_dir/f'{stage}_history.csv'
    if not resume and (best_path.exists() or last_path.exists()):
        raise FileExistsError(f'Existing run at {fold_dir}; use --resume or another --run-dir.')
    coarse_path=fold_dir/'coarse_best.pt' if stage=='fine' else None
    seed=config['seed']+fold*101+(10000 if stage=='fine' else 0)
    seed_everything(seed,config.get('deterministic',True))
    settings=stage_settings(config,stage)
    centers=None
    crop_sides=None
    coverage=None
    if stage=='fine':
        coarse,coarse_ckpt=load_checkpoint(coarse_path,device)
        if coarse_ckpt['manifest_hash']!=manifest_hash(manifest) or coarse_ckpt['fold']!=fold:
            raise ValueError('Coarse checkpoint was not trained on this manifest/fold.')
        if any(coarse_ckpt['config'][key]!=config[key] for key in ['backbone','pretrained','decoder_channels']):
            raise ValueError('Fine/coarse backbone, normalization and decoder width must match.')
        prediction=predict_records(coarse,valid,'coarse',stage_settings(coarse_ckpt['config'],'coarse'),device,tta=config['tta'])
        centers,crop_sides=refinement_windows(prediction,settings)
        coverage=np.mean([max(abs(np.array(centers[r['id']])-r['xy']))<crop_sides[r['id']]/2 for r in valid if r['visible']])
        print(f'fold {fold}: fine validation crop coverage = {coverage:.4f}',flush=True)
        if config.get('initialize_fine_from_coarse',True):
            model=coarse
        else:
            del coarse
            model=Localizer(config['backbone'],config['pretrained'],config['decoder_channels']).to(device)
    else:
        model=Localizer(config['backbone'],config['pretrained'],config['decoder_channels']).to(device)
    channels_last=config.get('channels_last',False) and device.type=='cuda'
    if channels_last:
        model=model.to(memory_format=torch.channels_last)
    presence_enabled=len({r['visible'] for r in train})==2 and config.get('train_presence_if_available',True)
    training=FoveaDataset(train,stage,settings,training=True,seed=seed)
    workers=config.get('workers',0)
    loader_options=dict(num_workers=workers,pin_memory=device.type=='cuda',drop_last=False)
    if workers:
        loader_options.update(persistent_workers=True,prefetch_factor=2,worker_init_fn=worker_init)
    loader=DataLoader(training,batch_size=config['batch_size'],sampler=EpochSampler(training,seed),**loader_options)
    optimizer=torch.optim.AdamW([
        {'params':model.encoder.parameters(),'lr':settings['lr']*config['encoder_lr_factor']},
        {'params':[p for name,p in model.named_parameters() if not name.startswith('encoder.')], 'lr':settings['lr']}
    ],weight_decay=config['weight_decay'],fused=bool(config.get('fused_optimizer',False) and device.type=='cuda'))
    epochs=settings['epochs']
    warmup=settings.get('warmup_epochs',5)
    def schedule(epoch):
        if epoch<warmup:
            return (epoch+1)/max(1,warmup)
        t=(epoch-warmup)/max(1,epochs-warmup)
        return 0.03+0.97*0.5*(1+math.cos(math.pi*t))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    amp=bool(config.get('amp',True) and device.type=='cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    ema=copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    start,best,bad,already_finished=0,float('inf'),0,False
    meta=checkpoint_metadata(config,manifest,fold,stage,presence_enabled,coarse_path)
    if resume and last_path.exists():
        state=torch.load(last_path,map_location=device,weights_only=False)
        for field in ['config','manifest_hash','fold','stage','coarse_sha256']:
            if state[field]!=meta[field]:
                raise ValueError(f'Resume configuration mismatch: {field}')
        model.load_state_dict(state['raw_model'])
        ema.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start,best,bad=state['epoch']+1,state['best'],state['bad']
        epochs=state.get('effective_epochs',epochs)
        if best_path.exists():
            saved_best=torch.load(best_path,map_location='cpu',weights_only=False)
            for field in ['config','manifest_hash','fold','stage','coarse_sha256']:
                if saved_best[field]!=meta[field]:
                    raise ValueError(f'Best checkpoint mismatch on resume: {field}')
            best=min(best,float(saved_best['validation_rmse']))
            del saved_best
        already_finished=state.get('finished',False)
        torch.set_rng_state(state['rng'].cpu())
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        if device.type=='cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    elif resume and best_path.exists():
        raise FileNotFoundError('Cannot resume without the matching last checkpoint.')
    write_json(fold_dir/'resolved_config.json',config)
    if already_finished or early_stopping_ready(bad,settings,start):
        start=epochs
    last_epoch=start-1
    reason='max_epochs'
    epoch_times=[]
    for epoch in range(start,epochs):
        if epoch>=epochs:
            break
        if deadline is not None and time.monotonic()>=deadline:
            reason='stage_time_budget'
            break
        last_epoch=epoch
        training.epoch=epoch
        model.encoder.requires_grad_(epoch>=settings.get('freeze_encoder_epochs',0))
        model.train()
        totals=[]
        stopped=False
        tick=time.monotonic()
        sigma=epoch_sigma(dict(settings,epochs=epochs),epoch)
        for batch in loader:
            if deadline is not None and time.monotonic()>=deadline:
                stopped=True
                break
            batch={k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            if channels_last:
                batch['image']=batch['image'].contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp):
                result=model(batch['image'])
                loss,_=localization_loss(result,batch,sigma=sigma,
                    coordinate_scale=settings['coordinate_scale'],box_weight=config['box_weight'],
                    presence_enabled=presence_enabled,return_parts=False)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: fold {fold}, {stage}, epoch {epoch}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,foreach=True)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema,model,config['ema_decay'])
            totals.append(loss.detach())
        if device.type=='cuda':
            torch.cuda.synchronize(device)
        update_seconds=time.monotonic()-tick
        scheduler.step()
        if stopped:
            reason='stage_time_budget'
        should_evaluate=(epoch==0 or (epoch+1)%settings.get('eval_every',1)==0 or epoch==epochs-1 or stopped)
        score=None
        if should_evaluate:
            prediction=predict_records(ema,valid,stage,settings,device,centers=centers,tta=config.get('validation_tta',False),crop_sides=crop_sides)
            score=rmse(valid,prediction)
            if score<best:
                best,bad=score,0
                save_atomic(dict(**meta,model=ema.state_dict(),epoch=epoch,validation_rmse=score),best_path)
            else:
                bad+=1
        early=early_stopping_ready(bad,settings,epoch+1)
        if early:
            reason='early_stopping'
        if epoch>=settings.get('freeze_encoder_epochs',0):
            epoch_times.append(time.monotonic()-tick)
        # Measure three unfrozen epochs; frozen warmup would underestimate backward cost.
        if deadline is not None and len(epoch_times)==3 and not stopped:
            duration=float(np.mean(epoch_times))
            affordable=fit_epoch_limit(epochs,epoch+1,duration,deadline-time.monotonic())
            if affordable<epochs:
                epochs=max(epoch+1,affordable)
                print(f'fold {fold} {stage}: observed throughput adjusts max epochs to {epochs}',flush=True)
        finished=stopped or early or epoch==epochs-1
        if finished and score is None:
            prediction=predict_records(ema,valid,stage,settings,device,centers=centers,tta=config.get('validation_tta',False),crop_sides=crop_sides)
            score=rmse(valid,prediction)
            if score<best:
                best,bad=score,0
                save_atomic(dict(**meta,model=ema.state_dict(),epoch=epoch,validation_rmse=score),best_path)
            else:
                bad+=1
            should_evaluate=True
        row=dict(epoch=epoch,steps=len(totals),train_loss=float(torch.stack(totals).mean()) if totals else '',
                 validation_rmse=score if score is not None else '',best_rmse=best,
                 mean_step_seconds=update_seconds/max(1,len(totals)),sigma=sigma)
        with history_path.open('a' if epoch else 'w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if epoch==0:
                writer.writeheader()
            writer.writerow(row)
        pausing=pause_requested(run_dir)
        if (epoch+1)%settings.get('save_last_every',1)==0 or finished or pausing:
            save_atomic(dict(**meta,model=ema.state_dict(),raw_model=model.state_dict(),
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                epoch=epoch,best=best,bad=bad,finished=finished,effective_epochs=epochs,
                rng=torch.get_rng_state(),python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None),last_path)
        if budget:
            budget.save(current_fold=fold,current_stage=stage,last_epoch=epoch)
        if should_evaluate:
            print(f'fold {fold} {stage} {epoch+1}/{epochs}: RMSE={score:.3f}px, best={best:.3f}, step={row["mean_step_seconds"]:.3f}s',flush=True)
        if pausing:
            raise TrainingPaused(f'Paused fold {fold} {stage} after epoch {epoch+1}; full state saved to {last_path}')
        if finished:
            break
    del loader,training,model,ema,optimizer
    if device.type=='cuda':
        torch.cuda.empty_cache()
    if not best_path.exists():
        raise TimeoutError('Stage budget expired before a model could be trained and validated.')
    selected,checkpoint=load_checkpoint(best_path,device)
    oof=predict_records(selected,valid,stage,settings,device,centers=centers,tta=config['tta'],crop_sides=crop_sides)
    write_json(fold_dir/f'{stage}_oof.json',dict(stage=stage,fold=fold,manifest_hash=manifest_hash(manifest),
               tta=config['tta'],predictions=oof,rmse=rmse(valid,oof)))
    write_json(fold_dir/f'{stage}_done.json',dict(config=config,manifest_hash=manifest_hash(manifest),
               best_sha256=hashlib.sha256(best_path.read_bytes()).hexdigest(),
               coarse_sha256=meta['coarse_sha256'],last_epoch=last_epoch,reason=reason,
               crop_coverage=float(coverage) if coverage is not None and np.isfinite(coverage) else None))
    print(f'fold {fold} {stage} TTA OOF RMSE: {rmse(valid,oof):.3f}px ({reason})',flush=True)


def train_folds(manifest,config,folds,run_dir,device='cuda',resume=False,stages=('coarse','fine')):
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install a CUDA-enabled PyTorch build, or explicitly use --device cpu.')
    config=copy.deepcopy(config)
    torch.set_num_threads(config.get('cpu_threads',4))
    tasks=[(fold,stage) for fold in folds for stage in stages]
    if any(not 0<=fold<manifest['n_folds'] for fold in folds) or len(set(folds))!=len(folds):
        raise ValueError('Invalid or duplicate folds.')
    if resume:
        tasks=[(fold,stage) for fold,stage in tasks if not stage_complete(manifest,config,fold,stage,run_dir)]
    if not tasks:
        print('All requested stages are already complete.',flush=True)
        return
    budget=RunBudget(Path(run_dir)/'budget.json',config['time_budget_seconds'],resume) if config.get('time_budget_seconds') else None
    if budget:
        budget.save(status='preparing')
    try:
        if pause_requested(run_dir):
            raise TrainingPaused('Pause requested before training; resume explicitly to continue.')
        if config.get('cache_dir'):
            info=build_cache(manifest['records'],config['cache_dir'])
            print('Pixel cache:',info,flush=True)
        write_json(Path(run_dir)/'runtime.json',dict(torch=torch.__version__,device=str(device),
            gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,config=config))
        for i,(fold,stage) in enumerate(tasks):
            if pause_requested(run_dir):
                raise TrainingPaused('Pause requested between stages.')
            weights=[config[s].get('budget_weight',1.) for _,s in tasks[i:]]
            deadline=budget.stage_deadline(weights,config.get('stage_reserve_seconds',60)) if budget else None
            train_stage(manifest,config,fold,stage,run_dir,device,resume,deadline,budget)
            if budget:
                budget.save(status='between_stages',completed_this_session=i+1)
    finally:
        if budget:
            budget.save(status='stopped_or_finished')
