import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import TrainDataset, load_rgb, load_label, preprocess, read_folds
from .inference import predict_probability, binary_dice
from .losses import segmentation_loss
from .model import build_model
from .runtime import seed_all, choose_device, training_fingerprint, save_json, atomic_torch_save


@torch.no_grad()
def update_ema(ema, model, decay):
    for ep,p in zip(ema.parameters(),model.parameters()):
        ep.mul_(decay).add_(p,alpha=1-decay)
    for eb,b in zip(ema.buffers(),model.buffers()):
        eb.copy_(b)


def validate(model, root, ids, config, device):
    predictions={}; scores=[]
    for sid in ids:
        x=preprocess(load_rgb(Path(root)/f'train/image/{sid}.jpg'),config)
        probability=predict_probability(model,x,device,tta=False,amp=config['amp'])
        target=load_label(Path(root)/f'train/label/{sid}.jpg')
        scores.append(binary_dice(probability>=config['selection_threshold'],target))
        predictions[sid]=probability
    return float(np.mean(scores)),predictions


def run_train(args, config):
    if not args.execute:
        raise SystemExit('Training is locked by default. No training was started. Add --execute only when you intend to train.')
    device=choose_device(args.device)
    folds=read_folds(args.splits)
    if folds['seed']!=config['seed'] or folds['fold_count']!=config['folds']:
        raise ValueError('Config and split seed/fold count differ; regenerate splits in a new experiment directory')
    if args.fold not in range(folds['fold_count']):
        raise ValueError('Invalid fold')
    split=folds['folds'][args.fold]
    if set(split['train_ids']) & set(split['val_ids']):
        raise ValueError('Train/validation leakage')
    fingerprint=training_fingerprint(args.data)
    if folds.get('training_fingerprint')!=fingerprint:
        raise ValueError('Data changed after split creation')
    seed_all(config['seed']+args.fold)
    cv2_threads=__import__('cv2'); cv2_threads.setNumThreads(0)
    run=Path(args.run_dir)/f'fold{args.fold}'; run.mkdir(parents=True,exist_ok=True)
    if (run/'last.pt').exists() and not args.resume:
        raise FileExistsError('This fold already has a checkpoint. Use --resume or a new run directory.')
    save_json(run/'config.json',config); save_json(run/'split.json',split)
    dataset=TrainDataset(args.data,split['train_ids'],config)
    loader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=False,num_workers=config['num_workers'],pin_memory=device.type=='cuda',persistent_workers=False)
    model=build_model(config).to(device)
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=config['amp'] and device.type=='cuda')
    start_epoch=0; best=-1.; bad_epochs=0
    if args.resume:
        last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        if last['config']!=config or last['split']!=split or last['training_fingerprint']!=fingerprint:
            raise ValueError('Resume config/data/split mismatch')
        model.load_state_dict(last['raw_model']); ema.load_state_dict(last['model'])
        optimizer.load_state_dict(last['optimizer']); scaler.load_state_dict(last['scaler'])
        start_epoch=last['completed_epochs']; best=last['best_dice']; bad_epochs=last['bad_epochs']
        torch.set_rng_state(last['torch_rng']); random.setstate(last['python_rng'])
        if device.type=='cuda' and last['cuda_rng'] is not None: torch.cuda.set_rng_state_all(last['cuda_rng'])
    print(f'TRAINING explicitly enabled: fold={args.fold}, device={device}, images={len(split["train_ids"])}/{len(split["val_ids"])}',flush=True)
    for epoch in range(start_epoch,config['epochs']):
        dataset.epoch=epoch; model.train(); total_loss=0
        for step,(x,y) in enumerate(loader):
            progress=epoch+step/len(loader)
            warmup=config['warmup_epochs']
            if progress < warmup:
                factor=(progress+1/len(loader))/max(1,warmup)
            else:
                phase=(progress-warmup)/max(1,config['epochs']-warmup)
                factor=0.01+0.99*0.5*(1+math.cos(math.pi*phase))
            for group in optimizer.param_groups: group['lr']=config['learning_rate']*factor
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=config['amp'] and device.type=='cuda'):
                outputs=model(x)
            loss,details=segmentation_loss(outputs,y,config,epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, step {step}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            previous_scale=scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= previous_scale:
                update_ema(ema,model,config['ema_decay'])
            total_loss+=float(loss.detach())
        score,predictions=validate(ema,args.data,split['val_ids'],config,device)
        improved=score>best
        if improved: best=score; bad_epochs=0
        else: bad_epochs+=1
        checkpoint={'model':ema.state_dict(),'raw_model':model.state_dict(),'config':config,'split':split,
                    'training_fingerprint':fingerprint,'completed_epochs':epoch+1,'best_dice':best,'val_dice':score,
                    'bad_epochs':bad_epochs,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                    'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        atomic_torch_save(checkpoint,run/'last.pt')
        if improved:
            compact={k:v for k,v in checkpoint.items() if k not in ['raw_model','optimizer','scaler','torch_rng','python_rng','cuda_rng']}
            atomic_torch_save(compact,run/'best.pt')
            (run/'oof_native').mkdir(exist_ok=True)
            for sid,prob in predictions.items(): np.save(run/f'oof_native/{sid}.npy',prob.astype(np.float32))
        row={'epoch':epoch+1,'train_loss':total_loss/len(loader),'validation_vessel_dice':score,'best_vessel_dice':best,'learning_rate':optimizer.param_groups[0]['lr'],**details}
        with open(run/'history.jsonl','a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if (Path(args.run_dir)/'PAUSE_REQUESTED').exists():
            print('Pause requested. Complete epoch checkpoint saved; resume with --resume.',flush=True)
            return
        if epoch+1>=config['min_epochs'] and bad_epochs>=config['patience']:
            print('Early stopping.',flush=True); break
