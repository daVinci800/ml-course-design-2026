"""Final refit of the validated old recipe on all 80 training images, three seeds.

No new validation score is claimed for these models. Public score must be measured.
"""
import argparse
import copy
import datetime as dt
import gc
import hashlib
import os
import random
import time
import traceback

import numpy as np

from final_round import ROOT,WORK,setup,read,write,inputs,blend,predict

JOB=ROOT/'runs/full_refit_v12'
REPORT=ROOT/'reports/full_refit_v12'
SEEDS=(20261027,20261028,20261029)
EPOCHS=160
AVERAGE_EPOCHS=(80,120,160)


def prepare(manifest):
    from fovea.detector_data import prepare_detector_folds
    destination=WORK/'full_refit_v12_data'
    prepare_detector_folds(manifest,destination)
    records=[r for r in manifest['records'] if r['split']=='train']
    images=[(destination/'images'/f'{r["id"]:04d}.jpg').as_posix() for r in records]
    if len(images)!=80: raise ValueError('Unexpected training count')
    (destination/'all_train.txt').write_text('\n'.join(images)+'\n',encoding='utf8')
    path=destination/'all_train.yaml'
    # Ultralytics requires a val entry; its terminal mAP is training-set mAP and is ignored.
    write(path,dict(path=destination.as_posix(),train='all_train.txt',val='all_train.txt',names=['macula']))
    return path


def finish(manifest):
    import csv
    import torch
    from ultralytics import YOLO
    from fovea.inference import make_submission
    from fovea.train import manifest_hash
    test=[r for r in manifest['records'] if r['split']=='test']
    old={r['id']:dict(xy=[0.,0.]) for r in test}
    baseline=ROOT/'submissions/submission_baseline_kaggle14_20261021.csv'
    if hashlib.sha256(baseline.read_bytes()).hexdigest()!='b7bdec6f12ce5fe389bfa22f80f1bf383c7791904d31eeb30ace80148b72bf3c':
        raise ValueError('Protected baseline changed')
    for row in csv.DictReader(baseline.open(encoding='utf-8-sig',newline='')):
        ident,axis=row['ImageID'].split('_Fovea_')
        old[int(ident)]['xy'][0 if axis=='X' else 1]=float(row['value'])
    members={};identities=[]
    for seed in SEEDS:
        path=JOB/f'seed_{seed}/averaged.pt'
        model=YOLO(str(path))
        ckpt=model.ckpt
        assert ckpt['fovea_manifest_hash']==manifest_hash(manifest)
        assert ckpt['fovea_training_count']==80 and ckpt['fovea_seed']==seed
        members[str(seed)]=predict(model,test,old,size=960)
        identities.append(dict(seed=seed,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),training_count=80,average_epochs=AVERAGE_EPOCHS))
        del model
        gc.collect();torch.cuda.empty_cache()
    all_data=blend(members,{str(seed):1/3 for seed in SEEDS})
    weights=dict(baseline=.25,all_data=.75)
    final=blend(dict(baseline=old,all_data=all_data),weights)
    output=ROOT/'submissions/submission_full_refit_v12_20261027.csv'
    if output.exists(): raise FileExistsError(output)
    make_submission(manifest['template'],final,output,{r['id']:(r['width'],r['height']) for r in test})
    rows=list(csv.DictReader(output.open(encoding='utf-8-sig',newline='')))
    template=list(csv.DictReader(open(manifest['template'],encoding='utf-8-sig',newline='')))
    assert len(rows)==40 and [r['ImageID'] for r in rows]==[r['ImageID'] for r in template]
    for r in rows: assert np.isfinite(float(r['value']))
    write(REPORT/'test_predictions.json',dict(predictions=final,all_data=all_data,members=members,weights=weights,checkpoints=identities))
    return dict(submission=str(output),rows=40,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        checkpoints=identities,weights=weights,training_count_per_model=80,baseline_public_rmse=14.39947,
        new_oof_rmse=None,new_public_rmse=None,
        status='Candidate from standard all-data refit; unmeasured public generalization. No claim of RMSE5 or improvement.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    setup()
    import torch
    import ultralytics
    from ultralytics import YOLO
    from fovea.train import manifest_hash
    from run_final_round import process_alive
    JOB.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    prior=read(JOB/'status.json') if (JOB/'status.json').exists() else {}
    if prior and not args.resume: raise FileExistsError('Job exists')
    if prior.get('phase')=='completed': return
    if prior and process_alive(prior.get('pid')): raise RuntimeError('Existing refit process is alive')
    if args.resume: (JOB/'pause.request').unlink(missing_ok=True)
    used=prior.get('active_seconds',0)
    started=time.monotonic()
    def status(phase,**kw):
        write(JOB/'status.json',dict(pid=os.getpid(),phase=phase,active_seconds=used+time.monotonic()-started,
              updated_at=dt.datetime.now().astimezone().isoformat(),**kw))
    manifest,_,_,_=inputs()
    if not (REPORT/'plan.json').exists():
        write(REPORT/'plan.json',dict(started_at=dt.datetime.now().astimezone().isoformat(),seeds=SEEDS,epochs=EPOCHS,
           average_epochs=AVERAGE_EPOCHS,imgsz=960,batch=4,lr0=.0001,warmup_bias_lr=0.,training_count=80,
           architecture='Same YOLO11s COCO architecture as public14.39947 recipe',
           validation='No held-out rows. Do not report terminal training mAP as validation. No test ground truth.',
           weights=dict(baseline=.25,all_data=.75),manifest_hash=manifest_hash(manifest),
           rationale='The three new architectural experiments failed stability gates. Refit the validated old architecture on all supplied labels.',
           source='Cached official COCO yolo11s; no external retinal labels',public_score_unknown=True))
    awake=False
    try:
        if os.name=='nt':
            import ctypes
            awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        status('preparing')
        data=prepare(manifest)
        for index,seed in enumerate(SEEDS):
            fd=JOB/f'seed_{seed}';fd.mkdir(exist_ok=True)
            if (fd/'completed.json').exists(): continue
            model=YOLO(str(WORK/'pretrained_cache/yolo11s.pt'))
            saved_path=fd/'full_last.pt'
            saved=torch.load(saved_path,map_location='cpu',weights_only=False) if saved_path.exists() else None
            average={} if saved is None else saved['averaging']
            count=0 if saved is None else saved['average_count']
            if saved is not None:
                assert saved['seed']==seed and saved['manifest_hash']==manifest_hash(manifest)
            seen=set()
            def restore(trainer):
                if saved is None: return
                trainer.model.load_state_dict(saved['model']);trainer.ema.ema.load_state_dict(saved['ema'])
                trainer.ema.updates=saved['ema_updates'];trainer.optimizer.load_state_dict(saved['optimizer'])
                trainer.scaler.load_state_dict(saved['scaler']);trainer.scheduler.load_state_dict(saved['scheduler'])
                trainer.start_epoch=saved['epoch']
                random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
                torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
                print(f'RESUMED seed={seed} epoch={trainer.start_epoch+1}',flush=True)
            def end(trainer):
                nonlocal count
                epoch=trainer.epoch+1
                if epoch in seen: return
                seen.add(epoch)
                if epoch in AVERAGE_EPOCHS:
                    for k,v in trainer.ema.ema.state_dict().items():
                        value=v.detach().cpu()
                        if k not in average: average[k]=value.clone()
                        elif value.is_floating_point(): average[k].add_(value)
                        else: average[k]=value.clone()
                    count+=1
                pause=(JOB/'pause.request').exists()
                if epoch%10==0 or pause:
                    snapshot=dict(seed=seed,epoch=epoch,manifest_hash=manifest_hash(manifest),model=trainer.model.state_dict(),
                        ema=trainer.ema.ema.state_dict(),ema_updates=trainer.ema.updates,optimizer=trainer.optimizer.state_dict(),
                        scaler=trainer.scaler.state_dict(),scheduler=trainer.scheduler.state_dict(),averaging=average,average_count=count,
                        python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                    temporary=fd/'full_last.tmp';torch.save(snapshot,temporary);temporary.replace(saved_path)
                    print(f'REFIT model={index+1}/3 seed={seed} epoch={epoch}/{EPOCHS} seconds={used+time.monotonic()-started:.1f}',flush=True)
                status('training',model=index+1,models=3,seed=seed,epoch=epoch,epochs=EPOCHS)
                if pause: raise InterruptedError('Full state saved at epoch boundary')
            model.add_callback('on_train_start',restore)
            model.add_callback('on_fit_epoch_end',end)
            model.train(data=str(data),epochs=EPOCHS,imgsz=960,batch=4,nbs=4,device=0,workers=2,
                optimizer='AdamW',lr0=.0001,lrf=.01,momentum=.9,weight_decay=.0005,cos_lr=True,
                warmup_epochs=3,warmup_bias_lr=0.,patience=0,seed=seed,deterministic=False,cache='ram',amp=True,
                mosaic=0.,mixup=0.,copy_paste=0.,degrees=10.,translate=.05,scale=.25,fliplr=.5,flipud=0.,
                hsv_h=.015,hsv_s=.3,hsv_v=.3,close_mosaic=0,val=False,save=False,plots=False,verbose=False,
                project=str(fd),name='train',exist_ok=True)
            assert count==len(AVERAGE_EPOCHS)
            for k,v in average.items():
                if v.is_floating_point(): average[k]=v/count
            snapshot=copy.deepcopy(model.trainer.ema.ema).float().cpu()
            snapshot.load_state_dict(average)
            torch.save(dict(model=snapshot.half(),ema=None,epoch=EPOCHS-1,train_args=vars(model.trainer.args),
                version=ultralytics.__version__,fovea_manifest_hash=manifest_hash(manifest),fovea_seed=seed,
                fovea_training_count=80,fovea_training_split='all_train',averaged_epochs=AVERAGE_EPOCHS),fd/'averaged.pt')
            write(fd/'completed.json',dict(seed=seed,epochs=EPOCHS,training_count=80,average_epochs=AVERAGE_EPOCHS))
            del model,saved,snapshot,average
            gc.collect();torch.cuda.empty_cache()
        status('predicting_test')
        result=finish(manifest)
        result['wall_minutes']=(dt.datetime.now().astimezone()-dt.datetime.fromisoformat(read(REPORT/'plan.json')['started_at'])).total_seconds()/60
        write(REPORT/'completion.json',result)
        status('completed',completion=result)
    except InterruptedError as exc:
        status('paused',reason=str(exc))
    except BaseException as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if awake: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__': main()
