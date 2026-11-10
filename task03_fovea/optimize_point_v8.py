"""Fixed-schedule point branch, frozen old detector, five isolated folds."""
import argparse
import copy
import datetime as dt
import gc
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback

from final_round import ROOT, WORK, read, write, setup, inputs, blend, score

JOB = ROOT/'runs/point_v8'
REPORT = ROOT/'reports/point_v8'
EPOCHS = 160
AVERAGE_EPOCHS = (120, 140, 160)


def prepare(manifest):
    import shutil
    from fovea.detector_data import yolo_box
    destination = WORK/'point_v8_data'
    images, labels = destination/'images', destination/'labels'
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(exist_ok=True)
    records = [r for r in manifest['records'] if r['split']=='train']
    for r in records:
        target = images/f'{r["id"]:04d}.jpg'
        if not target.exists(): shutil.copyfile(r['path'], target)
        # Preserve the supplied detection box; train the keypoint on the precise CSV coordinate.
        values = [0, *yolo_box(r['box'],r['width'],r['height']),
                  r['xy'][0]/r['width'], r['xy'][1]/r['height'], 2]
        (labels/(target.stem+'.txt')).write_text(' '.join(f'{v:.10f}' for v in values)+'\n',encoding='utf8')
    paths = []
    for fold in range(5):
        tr = [r for r in records if r['fold']!=fold]
        va = [r for r in records if r['fold']==fold]
        if {r['group'] for r in tr}&{r['group'] for r in va}: raise ValueError('Group leakage')
        for name, group in [('train',tr),('val',va)]:
            (destination/f'{name}_{fold}.txt').write_text('\n'.join((images/f'{r["id"]:04d}.jpg').as_posix() for r in group)+'\n',encoding='utf8')
        path = destination/f'fold_{fold}.yaml'
        write(path, dict(path=destination.as_posix(),train=f'train_{fold}.txt',val=f'val_{fold}.txt',
                         names=['macula'],kpt_shape=[1,3],flip_idx=[0]))
        paths.append(path)
    return paths


def predict(model, records, fallback):
    from fovea.point_branch import extract_points
    results = model.predict(source=[r['path'] for r in records],imgsz=960,conf=.001,iou=.5,max_det=1,
                            device=0,batch=4,half=True,verbose=False,save=False,augment=False)
    return extract_points(results,records,fallback)


def paired_diagnostic(records, previous, proposed):
    import numpy as np
    old = np.array([np.asarray(previous[r['id']]['xy'])-r['xy'] for r in records])
    new = np.array([np.asarray(proposed[r['id']]['xy'])-r['xy'] for r in records])
    groups = sorted({r['group'] for r in records})
    indices = [np.array([i for i,r in enumerate(records) if r['group']==g]) for g in groups]
    rng = np.random.default_rng(20261027)
    difference = []
    for _ in range(5000):
        ix = np.concatenate([indices[i] for i in rng.integers(0,len(indices),len(indices))])
        difference.append(float(np.sqrt(np.mean(new[ix]**2))-np.sqrt(np.mean(old[ix]**2))))
    return dict(rmse_difference_95=np.quantile(difference,[.025,.975]).tolist(),
                fraction_bootstrap_improved=float(np.mean(np.array(difference)<0)),
                warning='Paired grouped diagnostic, conditional on existing selected old checkpoints; not an independent test.')


def finish(manifest, records, fallback, baseline):
    from ultralytics import YOLO
    from fovea.evaluate import summarize
    from fovea.train import manifest_hash
    from fovea.inference import make_submission
    import csv
    components = {'baseline':baseline,'point':{}}
    for fold in range(5):
        model = YOLO(str(JOB/f'fold_{fold}/averaged.pt'))
        if model.ckpt['fovea_fold']!=fold or model.ckpt['fovea_manifest_hash']!=manifest_hash(manifest):
            raise ValueError('OOF checkpoint mismatch')
        valid = [r for r in records if r['fold']==fold]
        components['point'].update(predict(model,valid,baseline))
        del model
    write(REPORT/'point_oof.json',dict(predictions=components['point'],metrics=summarize(records,components['point'])))
    # Single candidate fixed before training. No search over ensemble coefficients.
    weights = {'baseline':.75,'point':.25}
    proposed = blend(components,weights)
    before, after = summarize(records,baseline), summarize(records,proposed)
    diagnostic = paired_diagnostic(records,baseline,proposed)
    improved_folds = sum(after['per_fold_rmse'][str(f)]<before['per_fold_rmse'][str(f)] for f in range(5))
    acceptable = (after['rmse']<=before['rmse']*.97 and improved_folds>=4
                  and diagnostic['fraction_bootstrap_improved']>=.9
                  and all(after['per_fold_rmse'][str(f)]<=before['per_fold_rmse'][str(f)]*1.1 for f in range(5)))
    write(REPORT/'decision.json',dict(baseline=before,point=summarize(records,components['point']),
          proposed=after,weights=weights,paired=diagnostic,improved_folds=improved_folds,
          recommend_new=acceptable,old_public_score=14.39947,rejected_v7_public_score=15.94521,
          new_public_score=None,policy='75% old +25% point fixed; >=3% overall improvement, >=4 improved folds, >=90% paired bootstrap improvement, no fold >10% worse.'))
    if not acceptable:
        print(f'GATE REJECTED: old={before["rmse"]:.5f}, proposed={after["rmse"]:.5f}, improved_folds={improved_folds}',flush=True)
        return dict(recommend_new=False,oof_rmse=after['rmse'],reason='Predeclared stability gate failed; retain 14.39947 submission')
    test = [r for r in manifest['records'] if r['split']=='test']
    old_test = {r['id']:dict(xy=[0.,0.]) for r in test}
    baseline_csv = ROOT/'submissions/submission_baseline_kaggle14_20261021.csv'
    if hashlib.sha256(baseline_csv.read_bytes()).hexdigest()!='b7bdec6f12ce5fe389bfa22f80f1bf383c7791904d31eeb30ace80148b72bf3c':
        raise ValueError('Baseline CSV changed')
    for row in csv.DictReader(baseline_csv.open(encoding='utf-8-sig',newline='')):
        ident,axis = row['ImageID'].split('_Fovea_')
        old_test[int(ident)]['xy'][0 if axis=='X' else 1] = float(row['value'])
    members = {}
    for fold in range(5):
        model = YOLO(str(JOB/f'fold_{fold}/averaged.pt'))
        members[str(fold)] = predict(model,test,old_test)
        del model
    point = blend(members,{str(f):.2 for f in range(5)})
    result = blend(dict(baseline=old_test,point=point),weights)
    submission = ROOT/'submissions/submission_point_v8_20261027.csv'
    if submission.exists(): raise FileExistsError(submission)
    make_submission(manifest['template'],result,submission,{r['id']:(r['width'],r['height']) for r in test})
    write(REPORT/'test_predictions.json',dict(predictions=result,members=members,weights=weights))
    return dict(recommend_new=True,submission=str(submission),oof_rmse=after['rmse'],sha256=hashlib.sha256(submission.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    setup()
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.loss import v8PoseLoss
    from ultralytics.nn.tasks import yaml_model_load
    from fovea.point_branch import CoordinateLoss
    from fovea.train import manifest_hash
    JOB.mkdir(exist_ok=True)
    REPORT.mkdir(exist_ok=True)
    if (JOB/'status.json').exists() and not args.resume: raise FileExistsError('Job exists; use --resume')
    started = time.monotonic()
    prior = read(JOB/'status.json') if (JOB/'status.json').exists() else {}
    used = prior.get('active_seconds',0)
    def status(phase,**kw):
        write(JOB/'status.json',dict(phase=phase,pid=os.getpid(),active_seconds=used+time.monotonic()-started,
              updated_at=dt.datetime.now().astimezone().isoformat(),**kw))
    if prior.get('phase')=='completed': return
    manifest, records, fallback, baseline = inputs()
    if not (REPORT/'plan.json').exists():
        write(REPORT/'plan.json',dict(started_at=dt.datetime.now().astimezone().isoformat(),epochs=EPOCHS,
             average_epochs=AVERAGE_EPOCHS,imgsz=960,trainable='Only model.23.cv4 point branch; all original detector parameters and BN frozen',
             init='matching-fold v6 only; no cross-fold weight transfer',lr0=.001,
             validation='Only after fixed late-EMA averaging, no best epoch selection',
             proposal=dict(baseline=.75,point=.25),manifest_hash=manifest_hash(manifest),
             gate='>=3% RMSE improvement, >=4/5 folds improve, >=90% paired grouped bootstrap improves, no fold >10% worse',
             public_scores=dict(baseline=14.39947,rejected_v7=15.94521),external_data=False))
    awake = False
    try:
        if os.name=='nt':
            import ctypes
            awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        status('preparing')
        yamls = prepare(manifest)
        cfg = yaml_model_load('yolo11s-pose.yaml')
        cfg.update(nc=1,kpt_shape=[1,3])
        config_path = JOB/'yolo11s-point-pose.yaml'
        # YAML writer preserves scale='s'; the model constructor uses this field.
        import yaml
        config_path.write_text(yaml.safe_dump(cfg),encoding='utf8')
        for fold in range(5):
            fd=JOB/f'fold_{fold}'
            fd.mkdir(exist_ok=True)
            if (fd/'completed.json').exists(): continue
            old_path=ROOT/f'runs/detector_v6/fold_{fold}/point_best.pt'
            old=YOLO(str(old_path))
            if old.ckpt['fovea_fold']!=fold or old.ckpt['fovea_manifest_hash']!=manifest_hash(manifest): raise ValueError('Warmstart identity mismatch')
            torch.manual_seed(20261027+fold)
            model=YOLO(str(config_path),task='pose')
            model.load(str(old_path))
            del old
            checkpoints = {}
            saved_path=fd/'full_last.pt'
            saved=torch.load(saved_path,map_location='cpu',weights_only=False) if saved_path.exists() else None
            if saved is not None:
                if saved['fold']!=fold or saved['manifest_hash']!=manifest_hash(manifest): raise ValueError('Resume identity mismatch')
                checkpoints=saved['averaging']
            def begin(trainer):
                loss=v8PoseLoss(trainer.model)
                loss.keypoint_loss=CoordinateLoss()
                trainer.model.criterion=loss
                trainable=[n for n,p in trainer.model.named_parameters() if p.requires_grad]
                if not trainable or any(not n.startswith('model.23.cv4.') for n in trainable):
                    raise RuntimeError('Unexpected trainable detector parameters')
                write(fd/'trainable.json',dict(parameters=trainable,count=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)))
                if saved is not None:
                    trainer.model.load_state_dict(saved['raw_model'])
                    trainer.ema.ema.load_state_dict(saved['ema'])
                    trainer.ema.updates=saved['ema_updates']
                    trainer.optimizer.load_state_dict(saved['optimizer'])
                    trainer.scaler.load_state_dict(saved['scaler'])
                    trainer.scheduler.load_state_dict(saved['scheduler'])
                    trainer.start_epoch=saved['epoch']
                    import random
                    import numpy as np
                    random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
                    torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
                    print(f'RESUMED fold={fold} epoch={trainer.start_epoch+1}',flush=True)
            seen=set()
            def end(trainer):
                epoch=trainer.epoch+1
                if epoch in seen: return
                seen.add(epoch)
                if epoch in AVERAGE_EPOCHS:
                    checkpoints[epoch]={k:v.detach().cpu().clone() for k,v in trainer.ema.ema.state_dict().items()}
                pause=(JOB/'pause.request').exists()
                if epoch%10==0 or pause:
                    import random
                    import numpy as np
                    state=dict(epoch=epoch,fold=fold,manifest_hash=manifest_hash(manifest),
                         raw_model=trainer.model.state_dict(),ema=trainer.ema.ema.state_dict(),ema_updates=trainer.ema.updates,
                         optimizer=trainer.optimizer.state_dict(),scaler=trainer.scaler.state_dict(),scheduler=trainer.scheduler.state_dict(),averaging=checkpoints,
                         python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                    temp=fd/'full_last.tmp'
                    torch.save(state,temp)
                    temp.replace(saved_path)
                    print(f'POINT_BRANCH fold={fold} epoch={epoch}/{EPOCHS} seconds={used+time.monotonic()-started:.1f}',flush=True)
                status('training',fold=fold,epoch=epoch,epochs=EPOCHS)
                if pause: raise InterruptedError('Pause requested; full state saved')
            model.add_callback('on_train_start',begin)
            model.add_callback('on_fit_epoch_end',end)
            model.train(data=str(yamls[fold]),epochs=EPOCHS,imgsz=960,batch=4,nbs=4,device=0,workers=2,
                freeze=list(range(23))+['23.cv2','23.cv3'],optimizer='AdamW',lr0=.001,lrf=.05,
                momentum=.9,weight_decay=.0005,cos_lr=True,warmup_epochs=3,warmup_bias_lr=0.,
                pose=1.,kobj=.1,patience=0,seed=20261027+fold,deterministic=False,cache='ram',amp=True,
                mosaic=0.,mixup=0.,copy_paste=0.,degrees=10.,translate=.05,scale=.25,
                fliplr=.5,flipud=0.,hsv_h=.015,hsv_s=.3,hsv_v=.3,close_mosaic=0,
                val=False,save=False,plots=False,verbose=False,project=str(fd),name='train',exist_ok=True)
            if set(checkpoints)!=set(AVERAGE_EPOCHS): raise ValueError('Missing fixed averaging checkpoint')
            snapshot=copy.deepcopy(model.trainer.ema.ema).cpu().float()
            state=snapshot.state_dict()
            for k,v in state.items():
                values=[checkpoints[e][k] for e in AVERAGE_EPOCHS]
                state[k]=torch.stack(values).mean(0) if v.is_floating_point() else values[-1]
            snapshot.load_state_dict(state)
            snapshot.criterion=None
            snapshot=snapshot.half()
            torch.save(dict(model=snapshot,ema=None,epoch=EPOCHS-1,train_args=vars(model.trainer.args),
                version=ultralytics.__version__,fovea_fold=fold,fovea_manifest_hash=manifest_hash(manifest)),fd/'averaged.pt')
            write(fd/'completed.json',dict(epochs=EPOCHS,averaged_epochs=AVERAGE_EPOCHS,source_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest()))
            del model,saved,snapshot,checkpoints
            gc.collect();torch.cuda.empty_cache()
        status('validating')
        result=finish(manifest,records,fallback,baseline)
        result['wall_minutes']=(dt.datetime.now().astimezone()-dt.datetime.fromisoformat(read(REPORT/'plan.json')['started_at'])).total_seconds()/60
        write(REPORT/'completion.json',result)
        status('completed',completion=result)
    except InterruptedError as exc:
        status('paused',reason=str(exc))
    except BaseException as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        if awake: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__': main()
