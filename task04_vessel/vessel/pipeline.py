import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .data import load_rgb, load_label, preprocess, read_folds
from .inference import predict_probability, binary_dice
from .rle import resize_probability, official_foreground, read_ids, masks_to_csv
from .runtime import checkpoint_load, training_fingerprint, save_json, choose_device


def calibrate(args):
    device=choose_device(args.device)
    folds=read_folds(args.splits)
    fingerprint=training_fingerprint(args.data)
    if folds['training_fingerprint']!=fingerprint: raise ValueError('Split/data mismatch')
    run=Path(args.run_dir)
    cache=run/'oof_tta' if args.tta else run/'oof_plain'
    cache.mkdir(parents=True,exist_ok=True)
    probabilities={}; targets={}; provenance=[]
    seen=set()
    # Each image is predicted ONLY by the model that did not train on it.
    for split in folds['folds']:
        path=run/f'fold{split["fold"]}/best.pt'
        model,ckpt=checkpoint_load(path,device)
        if ckpt['split']!=split or ckpt['training_fingerprint']!=fingerprint:
            raise ValueError(f'Checkpoint split/data mismatch: {path}')
        provenance.append({'fold':split['fold'],'checkpoint':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'completed_epochs':ckpt['completed_epochs']})
        for sid in split['val_ids']:
            if sid in seen or sid in split['train_ids']: raise ValueError('OOF leakage')
            seen.add(sid)
            prob=predict_probability(model,preprocess(load_rgb(Path(args.data)/f'train/image/{sid}.jpg'),ckpt['config']),device,tta=args.tta,amp=ckpt['config']['amp'])
            np.save(cache/f'{sid}.npy',prob)
            probabilities[sid]=prob
            targets[sid]=load_label(Path(args.data)/f'train/label/{sid}.jpg').astype(bool)
        del model
        if device.type=='cuda': torch.cuda.empty_cache()
    expected=set(p.stem for p in (Path(args.data)/'train/image').glob('*.jpg'))
    if seen!=expected: raise ValueError('OOF coverage incomplete')
    # Competition GT construction is not published: report two explicit proxies.
    prob512={sid:resize_probability(p) for sid,p in probabilities.items()}
    target_nearest={sid:np.asarray(Image.fromarray(t.astype(np.uint8)*255).resize((512,512),Image.Resampling.NEAREST))>0 for sid,t in targets.items()}
    target_official={sid:official_foreground(Path(args.data)/f'train/label/{sid}.jpg') for sid in targets}
    sweep=[]
    for threshold in np.round(np.arange(0.01,0.991,0.01),2):
        native=[binary_dice(probabilities[s]>=threshold,targets[s]) for s in targets]
        nn=[binary_dice(prob512[s]>=threshold,target_nearest[s]) for s in targets]
        off=[binary_dice(prob512[s]>=threshold,target_official[s]) for s in targets]
        score={'threshold':float(threshold),'native_vessel_dice':float(np.mean(native)),
               'submission_nearest_gt_vessel_dice':float(np.mean(nn)),
               'submission_official_gt_vessel_dice':float(np.mean(off)),
               'submission_official_gt_background_dice':float(np.mean([binary_dice(prob512[s]<threshold,~target_official[s]) for s in targets]))}
        sweep.append(score)
    metric={'native':'native_vessel_dice','official-proxy':'submission_official_gt_vessel_dice','nearest-proxy':'submission_nearest_gt_vessel_dice',
            'background-proxy':'submission_official_gt_background_dice'}[args.metric]
    best=max(sweep,key=lambda r:(r[metric],-abs(r['threshold']-0.5)))
    native_best=max(sweep,key=lambda r:(r['native_vessel_dice'],-abs(r['threshold']-0.5)))
    with open(run/'threshold_sweep.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(sweep[0])); w.writeheader(); w.writerows(sweep)
    report={'threshold':best['threshold'],'native_threshold':native_best['threshold'],
            'native_best_oof_scores':native_best,'threshold_search_range':[0.01,0.99],
            'selection_metric':metric,'tta':args.tta,'training_fingerprint':fingerprint,
            'best_oof_scores':best,'checkpoints':provenance,'num_oof_images':len(seen),
            'caveat':'OOF guides model/threshold selection; selection score is not an unbiased test estimate. Server GT resize/polarity are unverified.'}
    save_json(run/'calibration.json',report)
    print(report)


def predict(args):
    import json
    device=choose_device(args.device)
    run=Path(args.run_dir)
    calibration=json.loads((run/'calibration.json').read_text(encoding='utf-8'))
    if args.tta != calibration['tta']: raise ValueError('TTA must match OOF calibration')
    if training_fingerprint(args.data)!=calibration['training_fingerprint']:
        raise ValueError('Training data differ from calibration')
    ids=read_ids(Path(args.data)/'sample_submission.csv')
    files={p.stem:p for p in (Path(args.data)/'test/image').glob('*.jpg')}
    if set(files)!=set(ids): raise ValueError('Test IDs differ from sample submission')
    output=Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Prediction output must be new or empty; choose another --output')
    native=output/'masks_vessel0_native'; submission=output/'submission_masks_512'; probdir=output/'probabilities'
    for p in [native,submission,probdir]: p.mkdir(parents=True,exist_ok=True)
    sums={sid:np.zeros(load_rgb(files[sid]).shape[:2],dtype=np.float32) for sid in ids}
    checkpoints=calibration['checkpoints']
    for record in checkpoints:
        path=run/f'fold{record["fold"]}/best.pt'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Checkpoint changed since OOF calibration; rerun calibration')
        model,ckpt=checkpoint_load(path,device)
        if ckpt['training_fingerprint']!=calibration['training_fingerprint']: raise ValueError('Checkpoint/data mismatch')
        for sid in ids:
            prob=predict_probability(model,preprocess(load_rgb(files[sid]),ckpt['config']),device,tta=args.tta,amp=ckpt['config']['amp'])
            sums[sid] += prob/len(checkpoints)
        del model
        if device.type=='cuda': torch.cuda.empty_cache()
    threshold=calibration['threshold']
    native_threshold=calibration.get('native_threshold',threshold)
    fractions={}
    for sid,prob in sums.items():
        np.save(probdir/f'{sid}.npy',prob)
        Image.fromarray(np.where(prob>=native_threshold,0,255).astype(np.uint8)).save(native/f'{sid}.png')
        vessel512=resize_probability(prob)>=threshold
        # Explicit polarity. Never invert inside the official converter.
        mask=np.where(vessel512,255,0) if args.csv_foreground=='vessel' else np.where(vessel512,0,255)
        Image.fromarray(mask.astype(np.uint8)).save(submission/f'{sid}.png')
        fractions[sid]=float(vessel512.mean())
    masks_to_csv(submission,Path(args.data)/'sample_submission.csv',output/'submission.csv')
    manifest={'csv_foreground':args.csv_foreground,'threshold':threshold,'native_threshold':native_threshold,
              'tta':args.tta,'ensemble_models':len(checkpoints),
              'native_mask_values':'vessel=0; background=255','csv_encoded_values':'nonzero pixels of submission_masks_512',
              'vessel_fractions':fractions,'calibration':calibration,'warning':'Polarity follows explicit --csv-foreground. Check competition clarification before upload.'}
    save_json(output/'manifest.json',manifest)
    print(f'Created {output.resolve()} (local files only; no upload).')
