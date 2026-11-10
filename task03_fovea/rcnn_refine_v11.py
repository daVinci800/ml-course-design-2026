"""Inference-only coarse-guided two-stage refinement; no test-image-specific rules."""
import datetime as dt
import gc
import hashlib
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from final_round import ROOT,setup,read,write,inputs,blend

JOB=ROOT/'runs/rcnn_refine_v11'
REPORT=ROOT/'reports/rcnn_refine_v11'
RADIUS_FRACTION=.1


def choose_local(boxes,scores,center,radius):
    boxes=np.asarray(boxes,dtype=float).reshape(-1,4)
    scores=np.asarray(scores,dtype=float)
    if len(boxes)!=len(scores): raise ValueError('Candidate length mismatch')
    points=(boxes[:,:2]+boxes[:,2:])/2
    valid=(np.isfinite(points).all(1)&np.isfinite(scores)&
           (np.linalg.norm(points-np.asarray(center),axis=1)<=radius))
    if not valid.any(): return np.asarray(center,dtype=float),True
    k=np.where(valid)[0][np.argmax(scores[valid])]
    return points[k],False


@torch.inference_mode()
def predict_local(model,records,cache,centers):
    from optimize_rcnn_v10 import RetinaDataset,collate
    from torch.utils.data import DataLoader
    model.eval()
    model.roi_heads.detections_per_img=100
    loader=DataLoader(RetinaDataset(records,cache),batch_size=2,shuffle=False,collate_fn=collate,num_workers=0)
    byid={r['id']:r for r in records}
    output={}
    for images,_,idents in loader:
        with torch.autocast('cuda',dtype=torch.float16): results=model([im.cuda() for im in images])
        for ident,result in zip(idents,results):
            r=byid[ident]
            _,sx,sy=cache[ident]
            boxes=result['boxes'].float().cpu().numpy()/[sx,sy,sx,sy]
            scores=result['scores'].float().cpu().numpy()
            xy,missing=choose_local(boxes,scores,centers[ident]['xy'],RADIUS_FRACTION*r['fov_side'])
            xy=np.clip(xy,[0,0],[r['width']-1,r['height']-1])
            output[ident]=dict(xy=xy.tolist(),detector_miss=missing)
    return output


def main():
    setup()
    from optimize_rcnn_v10 import JOB as TRAIN_JOB,build_model,load_cache
    from optimize_point_v8 import paired_diagnostic
    from fovea.evaluate import summarize
    from fovea.inference import make_submission
    from fovea.train import manifest_hash
    from run_final_round import process_alive
    import csv
    JOB.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    if (JOB/'status.json').exists(): raise FileExistsError('Refinement job already exists')
    started=time.monotonic()
    def status(phase,**kw):
        write(JOB/'status.json',dict(pid=os.getpid(),phase=phase,active_seconds=time.monotonic()-started,
              updated_at=dt.datetime.now().astimezone().isoformat(),**kw))
    manifest,records,fallback,baseline=inputs()
    plan=dict(declared_at=dt.datetime.now().astimezone().isoformat(),radius_fraction=RADIUS_FRACTION,
         candidate_count=100,choose='Highest confidence inside fixed radius around matching-fold old model; otherwise old point',
         weights=dict(baseline=.5,rcnn=.5),gate='>=3% overall improvement, >=4 improved folds, >=90% paired grouped bootstrap improvement, no fold >10% worse',
         reason='After observing only raw RCNN folds0/1, fold1 contains a 639px jump; reject anatomically distant false proposals globally.',
         first_raw_folds_seen=[0,1],remaining_folds_not_used_to_set_rule=[2,3,4],no_test_predictions_observed=True,
         limitation='Adaptive validation policy, not independent nested validation. No threshold search after this amendment.')
    write(REPORT/'plan.json',plan)
    try:
        status('waiting_for_training')
        while True:
            state=read(TRAIN_JOB/'status.json')
            if state['phase']=='completed': break
            if state['phase'] in ('failed','paused'):
                raise RuntimeError(f'Training {state["phase"]}; preserve outputs and inspect before continuation')
            if not process_alive(state['pid']): raise RuntimeError('Training process exited unexpectedly')
            time.sleep(10)
        status('validating')
        cache=load_cache(records)
        component={}
        for fold in range(5):
            path=TRAIN_JOB/f'fold_{fold}/averaged.pt'
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            assert checkpoint['fold']==fold and checkpoint['manifest_hash']==manifest_hash(manifest)
            model=build_model(False)
            model.load_state_dict(checkpoint['model'])
            model.cuda().eval()
            valid=[r for r in records if r['fold']==fold]
            component.update(predict_local(model,valid,cache,baseline))
            print(f'LOCAL fold={fold} done',flush=True)
            del model,checkpoint
            gc.collect();torch.cuda.empty_cache()
        weights={'baseline':.5,'rcnn':.5}
        proposal=blend(dict(baseline=baseline,rcnn=component),weights)
        before,alone,after=summarize(records,baseline),summarize(records,component),summarize(records,proposal)
        diagnostic=paired_diagnostic(records,baseline,proposal)
        improved=sum(after['per_fold_rmse'][str(f)]<before['per_fold_rmse'][str(f)] for f in range(5))
        acceptable=(after['rmse']<=before['rmse']*.97 and improved>=4 and diagnostic['fraction_bootstrap_improved']>=.9
            and all(after['per_fold_rmse'][str(f)]<=before['per_fold_rmse'][str(f)]*1.1 for f in range(5)))
        unseen=[r for r in records if r['fold'] in (2,3,4)]
        write(REPORT/'decision.json',dict(baseline=before,rcnn=alone,proposed=after,weights=weights,paired=diagnostic,
             improved_folds=improved,recommend_new=acceptable,old_public_score=14.39947,new_public_score=None,
             later_fold_diagnostic=dict(baseline=summarize(unseen,baseline),proposed=summarize(unseen,proposal)),
             warning='Rule added after raw folds0/1 were observed. Fixed on subsequent folds but still uses selected old checkpoints.'))
        write(REPORT/'rcnn_oof.json',dict(predictions=component,metrics=alone))
        write(REPORT/'proposed_oof.json',dict(predictions=proposal,metrics=after))
        result=dict(recommend_new=acceptable,oof_rmse=after['rmse'],improved_folds=improved)
        if acceptable:
            status('predicting_test')
            test=[r for r in manifest['records'] if r['split']=='test']
            testcache=load_cache(test)
            old={r['id']:dict(xy=[0.,0.]) for r in test}
            path=ROOT/'submissions/submission_baseline_kaggle14_20261021.csv'
            assert hashlib.sha256(path.read_bytes()).hexdigest()=='b7bdec6f12ce5fe389bfa22f80f1bf383c7791904d31eeb30ace80148b72bf3c'
            for row in csv.DictReader(path.open(encoding='utf-8-sig',newline='')):
                ident,axis=row['ImageID'].split('_Fovea_')
                old[int(ident)]['xy'][0 if axis=='X' else 1]=float(row['value'])
            members={}
            identities=[]
            from ultralytics import YOLO
            from final_round import predict as predict_old
            for fold in range(5):
                old_path=ROOT/f'runs/detector_v6/fold_{fold}/point_best.pt'
                detector=YOLO(str(old_path))
                assert detector.ckpt['fovea_fold']==fold and detector.ckpt['fovea_manifest_hash']==manifest_hash(manifest)
                centers=predict_old(detector,test,old)
                del detector
                gc.collect();torch.cuda.empty_cache()
                path=TRAIN_JOB/f'fold_{fold}/averaged.pt'
                checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                assert checkpoint['fold']==fold and checkpoint['manifest_hash']==manifest_hash(manifest)
                model=build_model(False);model.load_state_dict(checkpoint['model']);model.cuda().eval()
                members[str(fold)]=predict_local(model,test,testcache,centers)
                identities.append(dict(fold=fold,rcnn_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),coarse_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest()))
                del model,checkpoint
                gc.collect();torch.cuda.empty_cache()
            rcnn=blend(members,{str(f):.2 for f in range(5)})
            predictions=blend(dict(baseline=old,rcnn=rcnn),weights)
            output=ROOT/'submissions/submission_rcnn_refine_v11_20261027.csv'
            if output.exists(): raise FileExistsError(output)
            make_submission(manifest['template'],predictions,output,{r['id']:(r['width'],r['height']) for r in test})
            write(REPORT/'test_predictions.json',dict(predictions=predictions,members=members,weights=weights,checkpoints=identities))
            result.update(submission=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        result['wall_minutes']=(time.monotonic()-started)/60
        write(REPORT/'completion.json',result)
        status('completed',completion=result)
    except BaseException as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__': main()
