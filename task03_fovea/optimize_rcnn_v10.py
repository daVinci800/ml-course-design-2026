"""Independent two-stage detector; fixed epochs, precise coordinate-centred targets."""
import argparse
import copy
import datetime as dt
import gc
import hashlib
import math
import os
import random
import time
import traceback

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from final_round import ROOT, WORK, setup, read, write, inputs, blend
from fovea.point_branch import freeze_batchnorm_stats

JOB=ROOT/'runs/rcnn_v10'
REPORT=ROOT/'reports/rcnn_v10'
EPOCHS=40
AVERAGE_EPOCHS=(30,35,40)
WEIGHTS=WORK/'pretrained_cache/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'


def coordinate_box(record):
    wh=np.asarray(record['box'][2:])-record['box'][:2]
    xy=np.asarray(record['xy'])
    box=np.r_[xy-wh/2,xy+wh/2]
    if not (0<=box[0]<box[2]<=record['width'] and 0<=box[1]<box[3]<=record['height']):
        raise ValueError('Coordinate-centred box outside image')
    return box


def transform_box(box,matrix):
    corners=np.array([[box[0],box[1]],[box[2],box[1]],[box[0],box[3]],[box[2],box[3]]])
    transformed=corners@matrix[:,:2].T+matrix[:,2]
    return np.r_[transformed.min(0),transformed.max(0)]


class RetinaDataset(Dataset):
    def __init__(self,records,cache,training=False,seed=0):
        self.records,self.cache,self.training,self.seed=records,cache,training,seed
        self.epoch=0

    def __len__(self): return len(self.records)

    def __getitem__(self,index):
        import cv2
        r=self.records[index]
        image,sx,sy=self.cache[r['id']]
        h,w=image.shape[:2]
        if self.training:
            box=coordinate_box(r)*[sx,sy,sx,sy]
            rng=np.random.default_rng(self.seed+self.epoch*1009+r['id']*17)
            matrix=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),float(rng.uniform(-7,7)),float(rng.uniform(.9,1.1)))
            matrix[:,2]+=rng.uniform(-.03,.03,2)*[w,h]
            if rng.random()<.5:
                matrix=np.array([[-1,0,w-1],[0,1,0]],dtype=float)@np.vstack([matrix,[0,0,1]])
            augbox=transform_box(box,matrix)
            if 0<=augbox[0]<augbox[2]<w and 0<=augbox[1]<augbox[3]<h:
                image=cv2.warpAffine(image,matrix,(w,h),flags=cv2.INTER_LINEAR)
                box=augbox
            contrast=float(rng.uniform(.85,1.15))
            brightness=float(rng.uniform(-12,12))
            image=np.clip(image.astype(np.float32)*contrast+brightness,0,255).astype(np.uint8)
            target=dict(boxes=torch.tensor(box,dtype=torch.float32)[None],labels=torch.ones(1,dtype=torch.int64))
        else: target={}
        tensor=torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float()/255
        return tensor,target,r['id']


def collate(batch): return list(zip(*batch))


def build_model(pretrained):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,trainable_backbone_layers=5,
          min_size=800,max_size=1280,box_detections_per_img=1,box_score_thresh=.001,box_nms_thresh=.5,
          box_batch_size_per_image=256,rpn_pre_nms_top_n_train=1000,rpn_post_nms_top_n_train=500,
          rpn_pre_nms_top_n_test=500,rpn_post_nms_top_n_test=200)
    if pretrained: model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True))
    predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,2)
    torch.nn.init.normal_(predictor.cls_score.weight,std=.01)
    torch.nn.init.zeros_(predictor.cls_score.bias)
    torch.nn.init.normal_(predictor.bbox_pred.weight,std=.001)
    torch.nn.init.zeros_(predictor.bbox_pred.bias)
    model.roi_heads.box_predictor=predictor
    for name,p in model.backbone.body.named_parameters():
        p.requires_grad=name.startswith(('layer3.','layer4.'))
    freeze_batchnorm_stats(model)
    return model


@torch.inference_mode()
def predict(model,records,cache,fallback):
    model.eval()
    predictions={}
    loader=DataLoader(RetinaDataset(records,cache),batch_size=2,shuffle=False,collate_fn=collate,num_workers=0)
    byid={r['id']:r for r in records}
    for images,_,idents in loader:
        with torch.autocast('cuda',dtype=torch.float16): result=model([im.cuda() for im in images])
        for ident,output in zip(idents,result):
            record=byid[ident]
            image,sx,sy=cache[ident]
            boxes=output['boxes'].float().cpu().numpy()
            scores=output['scores'].float().cpu().numpy()
            valid=np.isfinite(boxes).all(1)&np.isfinite(scores)
            if valid.any():
                k=np.where(valid)[0][np.argmax(scores[valid])]
                xy=(boxes[k,:2]+boxes[k,2:])/2/[sx,sy]
                missing=False
            else:
                xy=fallback[ident]['xy'];missing=True
            xy=np.clip(xy,[0,0],[record['width']-1,record['height']-1])
            predictions[ident]=dict(xy=xy.tolist(),detector_miss=missing)
    return predictions


def load_cache(records):
    import cv2
    cv2.setNumThreads(2)
    result={}
    for r in records:
        image=cv2.imread(r['path'])
        if image is None: raise ValueError('Cannot decode image')
        h,w=image.shape[:2]
        scale=min(800/min(h,w),1280/max(h,w))
        nw,nh=round(w*scale),round(h*scale)
        image=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA)
        result[r['id']]=(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),nw/w,nh/h)
    return result


def finish(manifest,records,cache,baseline):
    from fovea.evaluate import summarize
    from fovea.inference import make_submission
    from fovea.train import manifest_hash
    from optimize_point_v8 import paired_diagnostic
    import csv
    oof={}
    for fold in range(5):
        saved=read(REPORT/f'fold_{fold}_oof.json')
        assert saved['fold']==fold and saved['manifest_hash']==manifest_hash(manifest)
        valid_ids={r['id'] for r in records if r['fold']==fold}
        assert {int(k) for k in saved['predictions']}==valid_ids
        oof.update({int(k):v for k,v in saved['predictions'].items()})
    weights={'baseline':.5,'rcnn':.5}
    proposal=blend(dict(baseline=baseline,rcnn=oof),weights)
    before,alone,after=summarize(records,baseline),summarize(records,oof),summarize(records,proposal)
    diagnostic=paired_diagnostic(records,baseline,proposal)
    folds=sum(after['per_fold_rmse'][str(f)]<before['per_fold_rmse'][str(f)] for f in range(5))
    accept=(after['rmse']<=before['rmse']*.97 and folds>=4 and diagnostic['fraction_bootstrap_improved']>=.9
            and all(after['per_fold_rmse'][str(f)]<=before['per_fold_rmse'][str(f)]*1.1 for f in range(5)))
    write(REPORT/'decision.json',dict(baseline=before,rcnn=alone,proposed=after,weights=weights,paired=diagnostic,
          improved_folds=folds,recommend_new=accept,old_public_score=14.39947,new_public_score=None))
    write(REPORT/'rcnn_oof.json',dict(predictions=oof,metrics=alone))
    write(REPORT/'proposed_oof.json',dict(predictions=proposal,metrics=after))
    result=dict(recommend_new=accept,oof_rmse=after['rmse'],improved_folds=folds)
    if not accept: return result
    test=[r for r in manifest['records'] if r['split']=='test']
    testcache=load_cache(test)
    old={r['id']:dict(xy=[0.,0.]) for r in test}
    path=ROOT/'submissions/submission_baseline_kaggle14_20261021.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='b7bdec6f12ce5fe389bfa22f80f1bf383c7791904d31eeb30ace80148b72bf3c'
    for row in csv.DictReader(path.open(encoding='utf-8-sig',newline='')):
        ident,axis=row['ImageID'].split('_Fovea_')
        old[int(ident)]['xy'][0 if axis=='X' else 1]=float(row['value'])
    members={}
    for fold in range(5):
        checkpoint=torch.load(JOB/f'fold_{fold}/averaged.pt',map_location='cpu',weights_only=False)
        assert checkpoint['fold']==fold and checkpoint['manifest_hash']==manifest_hash(manifest)
        model=build_model(False)
        model.load_state_dict(checkpoint['model'])
        model.cuda().eval()
        members[str(fold)]=predict(model,test,testcache,old)
        del model,checkpoint
        gc.collect();torch.cuda.empty_cache()
    rcnn=blend(members,{str(f):.2 for f in range(5)})
    predictions=blend(dict(baseline=old,rcnn=rcnn),weights)
    output=ROOT/'submissions/submission_rcnn_v10_20261027.csv'
    if output.exists(): raise FileExistsError(output)
    make_submission(manifest['template'],predictions,output,{r['id']:(r['width'],r['height']) for r in test})
    write(REPORT/'test_predictions.json',dict(predictions=predictions,members=members,weights=weights))
    result.update(submission=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    setup()
    from fovea.train import manifest_hash
    JOB.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    status_file=JOB/'status.json'
    prior=read(status_file) if status_file.exists() else {}
    if prior and not args.resume: raise FileExistsError('Job exists')
    if prior.get('phase')=='completed': return
    if args.resume: (JOB/'pause.request').unlink(missing_ok=True)
    started=time.monotonic();used=prior.get('active_seconds',0)
    def status(phase,**kw):
        write(status_file,dict(pid=os.getpid(),phase=phase,active_seconds=used+time.monotonic()-started,
              updated_at=dt.datetime.now().astimezone().isoformat(),**kw))
    manifest,records,fallback,baseline=inputs()
    if not (REPORT/'plan.json').exists():
        write(REPORT/'plan.json',dict(started_at=dt.datetime.now().astimezone().isoformat(),epochs=EPOCHS,
          average_epochs=AVERAGE_EPOCHS,model='torchvision FasterRCNN ResNet50 FPN v2',
          pretrained_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),manifest_hash=manifest_hash(manifest),
          trainable='layer3,layer4,FPN,RPN,ROI heads; BN statistics frozen',lr_backbone=.00002,lr_heads=.0002,
          batch=2,accumulate=2,min_size=800,max_size=1280,target='Box widths/heights from annotations, centres from precise CSV',
          proposal=dict(baseline=.5,rcnn=.5),gate='>=3% overall RMSE improvement, >=4/5 improved folds, >=90% paired grouped bootstrap improvement, no fold >10% worse',
          validation='One fixed late EMA average; no best checkpoint selection',external_retinal_data=False))
    awake=False
    try:
        if os.name=='nt':
            import ctypes
            awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        status('preparing')
        cache=load_cache(records)
        for fold in range(5):
            fd=JOB/f'fold_{fold}';fd.mkdir(exist_ok=True)
            if (fd/'completed.json').exists(): continue
            tr=[r for r in records if r['fold']!=fold];va=[r for r in records if r['fold']==fold]
            if {r['group'] for r in tr}&{r['group'] for r in va}: raise ValueError('Group leakage')
            seed=20261027+fold
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark=True
            dataset=RetinaDataset(tr,cache,training=True,seed=seed)
            generator=torch.Generator().manual_seed(seed)
            loader=DataLoader(dataset,batch_size=2,shuffle=True,generator=generator,collate_fn=collate,num_workers=0)
            model=build_model(True).cuda()
            ema=copy.deepcopy(model).eval()
            for p in ema.parameters(): p.requires_grad_(False)
            groups=[]
            for backbone,lr in [(True,.00002),(False,.0002)]:
                groups.append(dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('backbone.body.')==backbone],lr=lr,initial_lr=lr))
            optimizer=torch.optim.AdamW(groups,weight_decay=.0001)
            scaler=torch.amp.GradScaler('cuda')
            saved_path=JOB/'full_last.pt'
            saved=torch.load(saved_path,map_location='cpu',weights_only=False) if args.resume and saved_path.exists() else None
            first=0;averaging={};average_count=0
            if saved is not None and saved['fold']==fold:
                if saved['manifest_hash']!=manifest_hash(manifest): raise ValueError('Resume mismatch')
                model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema'])
                optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
                first=saved['epoch'];averaging=saved['averaging'];average_count=saved['average_count']
                random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
                torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
                generator.set_state(saved['loader_rng'])
                print(f'RESUME fold={fold} epoch={first+1}',flush=True)
            del saved
            for epoch in range(first,EPOCHS):
                dataset.epoch=epoch
                model.train();freeze_batchnorm_stats(model)
                factor=(.1+.9*(epoch+1)/2) if epoch<2 else .1+.9*(1+math.cos(math.pi*(epoch-2)/(EPOCHS-2)))/2
                for g in optimizer.param_groups: g['lr']=g['initial_lr']*factor
                optimizer.zero_grad(set_to_none=True)
                total=0
                for step,(images,targets,_) in enumerate(loader):
                    images=[im.cuda(non_blocking=True) for im in images]
                    targets=[{k:v.cuda(non_blocking=True) for k,v in t.items()} for t in targets]
                    with torch.autocast('cuda',dtype=torch.float16): losses=model(images,targets);loss=sum(losses.values())
                    if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
                    scaler.scale(loss/2).backward()
                    total+=float(loss.detach())
                    if (step+1)%2==0 or step+1==len(loader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                        scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                        with torch.no_grad():
                            torch._foreach_mul_(list(ema.parameters()),.99)
                            torch._foreach_add_(list(ema.parameters()),list(model.parameters()),alpha=.01)
                            for a,b in zip(ema.buffers(),model.buffers()): a.copy_(b)
                    status('training',fold=fold,epoch=epoch+1,epochs=EPOCHS,step=step+1,steps=len(loader))
                number=epoch+1
                if number in AVERAGE_EPOCHS:
                    for k,v in ema.state_dict().items():
                        cpu=v.detach().cpu()
                        if k not in averaging: averaging[k]=cpu.clone()
                        elif cpu.is_floating_point(): averaging[k].add_(cpu)
                        else: averaging[k]=cpu.clone()
                    average_count+=1
                pause=(JOB/'pause.request').exists()
                if number%5==0 or pause:
                    checkpoint=dict(fold=fold,epoch=number,manifest_hash=manifest_hash(manifest),
                        model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                        averaging=averaging,average_count=average_count,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=generator.get_state())
                    temp=JOB/'full_last.tmp';torch.save(checkpoint,temp);temp.replace(saved_path)
                    del checkpoint
                    print(f'RCNN fold={fold} epoch={number}/{EPOCHS} loss={total/len(loader):.5f} seconds={used+time.monotonic()-started:.1f}',flush=True)
                if pause:
                    status('paused',fold=fold,epoch=number,reason='Full state saved');return
                if used+time.monotonic()-started>10800:
                    (JOB/'pause.request').write_text('Three hour training-stage limit',encoding='utf8')
            assert average_count==len(AVERAGE_EPOCHS)
            for k,v in averaging.items():
                if v.is_floating_point(): averaging[k]=v/average_count
            ema.load_state_dict(averaging)
            torch.save(dict(model={k:v.cpu() for k,v in ema.state_dict().items()},fold=fold,
                 manifest_hash=manifest_hash(manifest),epochs=EPOCHS,averaged_epochs=AVERAGE_EPOCHS),fd/'averaged.pt')
            predictions=predict(ema,va,cache,baseline)
            write(REPORT/f'fold_{fold}_oof.json',dict(fold=fold,manifest_hash=manifest_hash(manifest),predictions=predictions))
            write(fd/'completed.json',dict(epochs=EPOCHS,averaged_epochs=AVERAGE_EPOCHS))
            del model,ema,optimizer,scaler,averaging
            gc.collect();torch.cuda.empty_cache()
        status('validating')
        result=finish(manifest,records,cache,baseline)
        result['wall_minutes']=(dt.datetime.now().astimezone()-dt.datetime.fromisoformat(read(REPORT/'plan.json')['started_at'])).total_seconds()/60
        write(REPORT/'completion.json',result)
        status('completed',completion=result)
    except BaseException as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if awake: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__': main()
