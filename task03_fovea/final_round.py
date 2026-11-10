"""Last optimization round: honest fold inference, higher resolution, resumable training.

No test labels or leaderboard-driven per-image rules are used. Run actions separately.
"""
import argparse
import copy
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent.parent / 'work'
JOB = ROOT / 'runs/final_round_v7'
REPORT = ROOT / 'reports/final_round_v7'
EXPERIMENTS = ('fov','roi','full_seed2','full_highres')


def variant_space(name):
    return 'roi' if name=='roi' else 'full' if name.startswith('full_') else 'fov'


def variant_size(name):
    return 1280 if name=='full_highres' else 768 if name=='roi' else 960


def setup():
    os.chdir(ROOT)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf8')
    os.environ['YOLO_CONFIG_DIR'] = str(WORK / 'ultralytics_config')
    os.environ['MPLCONFIGDIR'] = str(WORK / 'matplotlib_config')
    os.environ['TORCH_HOME'] = str(WORK / 'pretrained_cache')
    os.environ['ULTRALYTICS_AUTOINSTALL'] = 'false'
    JOB.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    import torch
    from ultralytics import settings
    torch.set_num_threads(4)
    settings.update(dict(sync=False, weights_dir=str(WORK/'pretrained_cache'),
                         datasets_dir=str(WORK/'detector_data'), runs_dir=str(JOB)))


def read(path):
    # A Windows reader/antivirus can temporarily prevent rename. The writer's
    # in-place fallback can also briefly expose an incomplete JSON document.
    for attempt in range(20):
        try:
            return json.loads(Path(path).read_text(encoding='utf8'))
        except (PermissionError, json.JSONDecodeError):
            if attempt == 19: raise
            time.sleep(.05)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import uuid
    tmp = path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    content = json.dumps(value, indent=2, ensure_ascii=False)
    tmp.write_text(content, encoding='utf8')
    try:
        for attempt in range(10):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt < 9: time.sleep(.05*(attempt+1))
        # Some Windows ACL/file-sharing combinations allow updating a file but
        # deny replacing its directory entry. Keep training alive in that case.
        with path.open('w', encoding='utf8') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        tmp.unlink(missing_ok=True)


def inputs():
    from fovea.prepare import read_manifest
    from fovea.evaluate import collect_oof
    from fovea.train import manifest_hash
    manifest = read_manifest(ROOT/'prepared/manifest.json')
    records, fallback = collect_oof(manifest, ROOT/'runs/accuracy_5h', 'fine')
    previous = read(ROOT/'reports/detector_v6/detector_oof.json')
    if previous['manifest_hash'] != manifest_hash(manifest):
        raise ValueError('Previous manifest mismatch')
    baseline = {int(k): v for k, v in previous['predictions'].items()}
    return manifest, records, fallback, baseline


def blend(components, weights):
    from fovea.ensemble import combine_predictions
    return combine_predictions(components, weights)


def score(records, pred):
    import numpy as np
    return float(np.sqrt(np.mean([(np.asarray(pred[r['id']]['xy'])-r['xy'])**2 for r in records])))


def crop(image, center, side):
    """Integer crop with padding; returned origin exactly inverts coordinates."""
    import cv2
    side = int(round(side))
    x, y = (int(round(float(v)-side/2)) for v in center)
    h, w = image.shape[:2]
    out = cv2.copyMakeBorder(image, max(0,-y), max(0,y+side-h),
                             max(0,-x), max(0,x+side-w), cv2.BORDER_CONSTANT)
    return out[max(y,0):max(y,0)+side, max(x,0):max(x,0)+side].copy(), (x,y)


def predict(model, records, fallback, size=960, flip=False, space='full', centers=None,aggregation='top'):
    import cv2
    import numpy as np
    from fovea.detector_data import detector_points
    effective_fallback=centers if space=='roi' else fallback
    if effective_fallback is None: raise ValueError('ROI inference requires automatic coarse centers')
    if space == 'full' and not flip:
        sources = [r['path'] for r in records]
        offsets = [(0,0)]*len(records)
        shapes = [(r['height'],r['width']) for r in records]
    else:
        sources, offsets, shapes = [], [], []
        for r in records:
            im = cv2.imread(r['path'])
            if im is None:
                raise ValueError('Cannot decode image')
            origin = (0,0)
            if space == 'fov':
                im, origin = crop(im, r['fov_center'], r['fov_side']*1.04)
            elif space == 'roi':
                im, origin = crop(im, centers[r['id']]['xy'], 1152)
            shapes.append(im.shape[:2]); offsets.append(origin)
            sources.append(np.ascontiguousarray(im[:,::-1]) if flip else im)
    if aggregation not in ('top','dense'): raise ValueError('Unknown box aggregation')
    result = model.predict(source=sources, imgsz=size, conf=.001, iou=.95 if aggregation=='dense' else .5,
                           max_det=30 if aggregation=='dense' else 1,
                           device=0, batch=4, half=True, verbose=False, save=False, augment=False)
    local, back = [], {}
    for r, (h,w), (ox,oy) in zip(records, shapes, offsets):
        local.append(dict(r,width=w,height=h))
        px,py = np.asarray(effective_fallback[r['id']]['xy'])-[ox,oy]
        back[r['id']] = dict(xy=[w-px if flip else px,py])
    points = detector_points(result,local,back)
    for r,(h,w),(ox,oy),res in zip(records,shapes,offsets,result):
        p = points[r['id']]
        if aggregation=='dense' and not p['detector_miss']:
            from fovea.box_fusion import dense_center
            fused=dense_center(res.boxes.xyxy.detach().float().cpu().numpy(),res.boxes.conf.detach().float().cpu().numpy())
            if fused is not None: p['xy']=fused.tolist()
        # A missed local detector falls back in original space, without clipping the fallback to its crop.
        if p['detector_miss']:
            p['xy'] = list(effective_fallback[r['id']]['xy'])
        else:
            x,y = p['xy']
            p['xy'] = np.clip([w-x+ox if flip else x+ox,y+oy], [0,0],
                               [r['width']-1,r['height']-1]).tolist()
        p['confidence'] = float(res.boxes.conf.max().item()) if len(res.boxes) else 0.
    return points


def probe():
    import numpy as np
    import torch
    from ultralytics import YOLO
    from fovea.evaluate import summarize
    from fovea.train import manifest_hash
    manifest, records, fallback, baseline = inputs()
    views = {'control':(960,False), 'flip':(960,True), 'small':(768,False),
             'large':(1152,False), 'large_flip':(1152,True), 'xl':(1280,False)}
    all_points = {k:{} for k in views}
    started = time.monotonic()
    for fold in range(5):
        model = YOLO(str(ROOT/f'runs/detector_v6/fold_{fold}/point_best.pt'))
        ckpt = model.ckpt
        if ckpt['fovea_fold'] != fold or ckpt['fovea_manifest_hash'] != manifest_hash(manifest):
            raise ValueError('Checkpoint identity mismatch')
        valid = [r for r in records if r['fold']==fold]
        for name,(size,flip) in views.items():
            p = predict(model,valid,fallback,size,flip)
            all_points[name].update(p)
            print(f'PROBE fold={fold} view={name} rmse={score(valid,p):.4f}',flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    recipes = {'control':{'control':1}, 'flip2':{'control':1,'flip':1},
               'large2':{'large':1,'large_flip':1},
               'four':{'control':1,'flip':1,'large':1,'large_flip':1},
               'scale3':{'control':1,'small':1,'large':1},
               'xl':{'xl':1}, 'large':{'large':1}}
    candidates = {k:dict(weights=w,metrics=summarize(records,blend(all_points,w))) for k,w in recipes.items()}
    winner = min(candidates,key=lambda k:candidates[k]['metrics']['rmse'])
    diffs = [np.linalg.norm(np.asarray(all_points['control'][r['id']]['xy'])-baseline[r['id']]['xy']) for r in records]
    errors = sorted([dict(id=r['id'],fold=r['fold'],error=(np.asarray(baseline[r['id']]['xy'])-r['xy']).tolist(),
                         ed=float(np.linalg.norm(np.asarray(baseline[r['id']]['xy'])-r['xy']))) for r in records],
                    key=lambda x:-x['ed'])
    for name,p in all_points.items():
        write(REPORT/f'probe_{name}_oof.json',dict(manifest_hash=manifest_hash(manifest),predictions=p))
    write(REPORT/'probe.json',dict(selected=winner,candidates=candidates,views=views,
          control_max_difference=max(diffs),baseline_metrics=summarize(records,baseline),
          baseline_outliers=errors,wall_seconds=time.monotonic()-started))
    print(json.dumps(dict(selected=winner,scores={k:v['metrics']['rmse'] for k,v in candidates.items()},
                          control_max_difference=max(diffs)),indent=2),flush=True)


def prepare_data(manifest, variant):
    """Fold lists isolate groups; ROI crops use only training labels, never held-out predictions."""
    import cv2
    import numpy as np
    from fovea.detector_data import yolo_box
    if variant.startswith('full_'):
        from fovea.detector_data import prepare_detector_folds
        return prepare_detector_folds(manifest,WORK/'detector_data')
    dest = WORK/f'final_v7_data/{variant}'
    images, labels = dest/'images', dest/'labels'
    images.mkdir(parents=True,exist_ok=True); labels.mkdir(exist_ok=True)
    paths = {}
    for r in manifest['records']:
        if r['split']!='train': continue
        im = cv2.imread(r['path'])
        specs = [(r['fov_center'],r['fov_side']*1.04)]
        if variant == 'roi':
            # Four independent translations prevent simply predicting the crop midpoint.
            rng = np.random.default_rng(20261027+r['id'])
            specs = [(np.asarray(r['xy'])+rng.uniform(-230,230,2),1152) for _ in range(4)]
        paths[r['id']] = []
        for i,(center,side) in enumerate(specs):
            cropped,(x,y) = crop(im,center,side)
            h,w = cropped.shape[:2]
            box = np.asarray(r['box'])-np.array([x,y,x,y])
            normalized = yolo_box(box,w,h)
            name = f'{r["id"]:04d}_{i}'
            target = images/(name+'.png')
            if not target.exists():
                if not cv2.imwrite(str(target),cropped): raise IOError('Writing crop failed')
            (labels/(name+'.txt')).write_text('0 '+' '.join(f'{v:.10f}' for v in normalized)+'\n',encoding='utf8')
            paths[r['id']].append(target.as_posix())
    yamls=[]
    train = [r for r in manifest['records'] if r['split']=='train']
    for fold in range(5):
        tr,va = [r for r in train if r['fold']!=fold],[r for r in train if r['fold']==fold]
        if {r['group'] for r in tr}&{r['group'] for r in va}: raise ValueError('Group leakage')
        for split,rows in [('train',tr),('val',va)]:
            (dest/f'{split}_{fold}.txt').write_text('\n'.join(p for r in rows for p in paths[r['id']])+'\n',encoding='utf8')
        yaml=dest/f'fold_{fold}.yaml'
        write(yaml,dict(path=dest.as_posix(),train=f'train_{fold}.txt',val=f'val_{fold}.txt',names=['macula']))
        yamls.append(yaml)
    return yamls


class Paused(Exception):
    pass


def train(variant, resume=False):
    import ctypes
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from fovea.train import manifest_hash
    from fovea.evaluate import summarize
    manifest, records, fallback, baseline = inputs()
    directory = JOB/variant
    directory.mkdir(exist_ok=True)
    status_path = directory/'status.json'
    previous=read(status_path) if status_path.exists() else {}
    if previous and not resume: raise FileExistsError('Use --resume to continue this variant')
    if previous.get('phase')=='completed': return
    lock_path=directory/'active.lock'
    if lock_path.exists(): raise RuntimeError('Active lock exists; inspect PID before removing it')
    lock=lock_path.open('x');lock.write(str(os.getpid()));lock.flush()
    started=time.monotonic();used=previous.get('active_seconds',0.)
    # Each variant has a bounded allowance. The coordinator records the complete round duration.
    allowance = 7200 if variant=='full_highres' else 6000 if variant=='roi' else 3900
    def status(phase,**extra):
        state=dict(phase=phase,pid=os.getpid(),variant=variant,
                   active_seconds=used+time.monotonic()-started,updated_at=dt.datetime.now().astimezone().isoformat(),**extra)
        write(status_path,state)
    def atomic_torch(obj,path):
        tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
    epochs=110 if variant=='roi' else 160 if variant.startswith('full_') else 180
    size=variant_size(variant)
    batch=2 if variant=='full_highres' else 4
    seed=20261024 if variant=='full_highres' else 20261103 if variant=='full_seed2' else 20261026
    status('preparing')
    awake=False
    try:
        if os.name=='nt': awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        yamls=prepare_data(manifest,variant)
        write(REPORT/f'{variant}_plan.json',dict(manifest_hash=manifest_hash(manifest),epochs=epochs,imgsz=size,
              seed=seed,lr0=.0001,warmup_bias_lr=0,batch=batch,effective_batch=4,roi_side=1152,
              checkpoint_interval=5,pause_file=str(JOB/'pause.request'),
              resume_note='Full FP32 raw model, EMA, optimizer, scaler, scheduler and RNG at epoch boundary. Worker prefetch may differ on resume.',
              pretrained='COCO yolo11s; no retinal external labels',
              validation='same held-out folds, original-coordinate RMSE; ROI validation centers from matching-fold v6 model'))
        for fold in range(5):
            fd=directory/f'fold_{fold}';fd.mkdir(exist_ok=True)
            if (fd/'completed.json').exists(): continue
            if (JOB/'pause.request').exists(): raise Paused('Pause requested before fold')
            valid=[r for r in records if r['fold']==fold]
            state_path=fd/'full_last.pt'
            saved=torch.load(state_path,map_location='cpu',weights_only=False) if state_path.exists() else None
            history=read(fd/'history.json') if (fd/'history.json').exists() else []
            best=min((h['rmse'] for h in history),default=float('inf'))
            if saved is not None and saved['epoch']+1>=epochs:
                if saved['fold']!=fold or saved['manifest_hash']!=manifest_hash(manifest):
                    raise ValueError('Resume identity mismatch at completed epoch boundary')
                if not (fd/'point_best.pt').exists() or not (fd/'detector_oof.json').exists():
                    raise RuntimeError('Completed boundary is missing a validated best checkpoint')
                write(fd/'completed.json',dict(best_rmse=best,epochs=epochs,resumed_completed_boundary=True))
                del saved
                continue
            model=YOLO(str(WORK/'pretrained_cache/yolo11s.pt'))
            # Fresh COCO initialization for every fold avoids cross-fold warm-start leakage.
            if saved is not None:
                if saved['fold']!=fold or saved['manifest_hash']!=manifest_hash(manifest): raise ValueError('Resume identity mismatch')
            seen_epochs=set()
            def restore(trainer):
                if saved is None: return
                trainer.model.load_state_dict(saved['raw_model'])
                trainer.ema.ema.load_state_dict(saved['ema'])
                trainer.ema.updates=saved['ema_updates']
                trainer.optimizer.load_state_dict(saved['optimizer'])
                trainer.scaler.load_state_dict(saved['scaler'])
                trainer.scheduler.load_state_dict(saved['scheduler'])
                trainer.start_epoch=saved['epoch']+1
                random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
                torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
                print(f'RESUMED {variant} fold {fold} epoch {trainer.start_epoch}',flush=True)
            def epoch_end(trainer):
                nonlocal best
                epoch=trainer.epoch+1
                if epoch in seen_epochs: return
                seen_epochs.add(epoch)
                pause=(JOB/'pause.request').exists()
                timeout=used+time.monotonic()-started>allowance
                validate=epoch==1 or epoch%5==0 or epoch==epochs or pause or timeout
                if validate:
                    wrapper=YOLO('yolo11s.yaml')
                    wrapper.model=copy.deepcopy(trainer.ema.ema).float().eval()
                    pred=predict(wrapper,valid,fallback,size=size,space=variant_space(variant),centers=baseline)
                    rmse=score(valid,pred)
                    history.append(dict(epoch=epoch,rmse=rmse,misses=sum(p['detector_miss'] for p in pred.values())))
                    if rmse<best:
                        best=rmse
                        snapshot=dict(model=copy.deepcopy(trainer.ema.ema).half().cpu(),ema=None,epoch=epoch-1,
                            train_args=vars(trainer.args),version=ultralytics.__version__,date=dt.datetime.now().isoformat(),
                            fovea_manifest_hash=manifest_hash(manifest),fovea_fold=fold,fovea_rmse=rmse,fovea_variant=variant)
                        atomic_torch(snapshot,fd/'point_best.pt');del snapshot
                        write(fd/'detector_oof.json',dict(stage='detector',fold=fold,manifest_hash=manifest_hash(manifest),predictions=pred,rmse=rmse))
                    write(fd/'history.json',history)
                    del wrapper
                    print(f'POINT {variant} fold {fold} epoch {epoch}: rmse={rmse:.4f} best={best:.4f}',flush=True)
                # Checkpoint raw FP32 state, not Ultralytics' lossy EMA-only resume state.
                if validate:
                    snapshot=dict(epoch=epoch-1,fold=fold,manifest_hash=manifest_hash(manifest),
                        raw_model={k:v.detach().cpu() for k,v in trainer.model.state_dict().items()},
                        ema={k:v.detach().cpu() for k,v in trainer.ema.ema.state_dict().items()},ema_updates=trainer.ema.updates,
                        optimizer=trainer.optimizer.state_dict(),scaler=trainer.scaler.state_dict(),scheduler=trainer.scheduler.state_dict(),
                        python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                    atomic_torch(snapshot,state_path);del snapshot
                status('training',fold=fold,epoch=epoch,epochs=epochs,best_rmse=best)
                if pause: raise Paused(f'Saved full state at fold {fold}, epoch {epoch}')
                if timeout:
                    # Do not create a partially trained later fold. Preserve full state for explicit resume.
                    raise Paused('Variant time allowance reached; state saved')
            model.add_callback('on_train_start',restore)
            model.add_callback('on_fit_epoch_end',epoch_end)
            model.train(data=str(yamls[fold]),epochs=epochs,imgsz=size,batch=batch,nbs=4,device=0,workers=2,
                optimizer='AdamW',lr0=.0001,lrf=.01,momentum=.9,weight_decay=.0005,cos_lr=True,
                warmup_epochs=3,warmup_bias_lr=0.,patience=0,seed=seed+fold,deterministic=False,
                cache='ram',amp=True,mosaic=0.,mixup=0.,copy_paste=0.,degrees=10. if variant.startswith('full_') else 7.,
                translate=.05 if variant.startswith('full_') else .08,scale=.25 if variant.startswith('full_') else .2,
                fliplr=.5,flipud=0.,hsv_h=.015,hsv_s=.3 if variant.startswith('full_') else .25,
                hsv_v=.3 if variant.startswith('full_') else .25,close_mosaic=0,
                val=False,save=False,plots=False,verbose=False,project=str(fd),name='train',exist_ok=True)
            write(fd/'completed.json',dict(best_rmse=best,epochs=epochs))
            del model,saved;gc.collect();torch.cuda.empty_cache()
        predictions={}
        for fold in range(5):
            p=read(directory/f'fold_{fold}/detector_oof.json')
            if p['fold']!=fold or p['manifest_hash']!=manifest_hash(manifest): raise ValueError('OOF identity mismatch')
            for k,v in p['predictions'].items():
                ident=int(k)
                if ident in predictions or not any(r['id']==ident and r['fold']==fold for r in records): raise ValueError('OOF leakage')
                predictions[ident]=v
        if set(predictions)!={r['id'] for r in records}: raise ValueError('Incomplete OOF')
        metrics=summarize(records,predictions)
        write(REPORT/f'{variant}_oof.json',dict(manifest_hash=manifest_hash(manifest),predictions=predictions,metrics=metrics))
        status('completed',rmse=metrics['rmse'])
    except Paused as exc:
        status('paused',reason=str(exc));print(str(exc),flush=True)
    except BaseException as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        lock.close();lock_path.unlink(missing_ok=True)
        if awake: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def finish():
    """Select on training OOF, then apply the exact recipe automatically to all test images."""
    import csv
    import itertools
    import numpy as np
    import torch
    from ultralytics import YOLO
    from fovea.evaluate import summarize,collect_oof
    from fovea.train import manifest_hash
    from fovea.inference import make_submission
    manifest, records, fallback, baseline = inputs()
    components={'previous':baseline}
    dense_components={}
    variants=[]
    replay_centers={int(k):v for k,v in read(REPORT/'probe_control_oof.json')['predictions'].items()}
    verification=[]
    for name in EXPERIMENTS:
        state_path=JOB/name/'status.json'
        if not state_path.exists() or read(state_path)['phase']!='completed':
            raise RuntimeError(f'Variant {name} must complete before final selection')
        data=read(REPORT/f'{name}_oof.json')
        if data['manifest_hash']!=manifest_hash(manifest): raise ValueError('Manifest mismatch')
        # Select using the saved deployment checkpoints, including FP16 serialization and ROI preprocessing.
        deployed={};dense_deployed={}
        for fold in range(5):
            valid=[r for r in records if r['fold']==fold]
            path=JOB/name/f'fold_{fold}/point_best.pt'
            model=YOLO(str(path))
            if model.ckpt['fovea_fold']!=fold or model.ckpt['fovea_manifest_hash']!=manifest_hash(manifest):
                raise ValueError('OOF replay identity mismatch')
            points=predict(model,valid,fallback,size=variant_size(name),
                           space=variant_space(name),centers=replay_centers)
            deployed.update(points)
            dense_deployed.update(predict(model,valid,fallback,size=variant_size(name),
                           space=variant_space(name),centers=replay_centers,aggregation='dense'))
            verification.append(dict(variant=name,fold=fold,rmse=score(valid,points),
                                      checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            del model;gc.collect();torch.cuda.empty_cache()
        components[name]=deployed
        dense_components[name+'_dense']=dense_deployed
        write(REPORT/f'{name}_deployed_oof.json',dict(manifest_hash=manifest_hash(manifest),predictions=deployed,
              metrics=summarize(records,deployed),training_selected_rmse=data['metrics']['rmse']))
        write(REPORT/f'{name}_dense_oof.json',dict(manifest_hash=manifest_hash(manifest),predictions=dense_deployed,
                                                metrics=summarize(records,dense_deployed)))
        variants.append(name)
    dense_previous={}
    for fold in range(5):
        valid=[r for r in records if r['fold']==fold]
        model=YOLO(str(ROOT/f'runs/detector_v6/fold_{fold}/point_best.pt'))
        if model.ckpt['fovea_fold']!=fold or model.ckpt['fovea_manifest_hash']!=manifest_hash(manifest):
            raise ValueError('Previous dense checkpoint identity mismatch')
        dense_previous.update(predict(model,valid,fallback,aggregation='dense'))
        del model;gc.collect();torch.cuda.empty_cache()
    dense_components['previous_dense']=dense_previous
    write(REPORT/'previous_dense_oof.json',dict(manifest_hash=manifest_hash(manifest),predictions=dense_previous,
                                              metrics=summarize(records,dense_previous)))
    components.update(dense_components)
    for name,directory in [('resnet_v4','resnet34_v4'),('convnext_v4','convnext_v4')]:
        _,components[name]=collect_oof(manifest,ROOT/'runs'/directory,'fine')
    # Coefficients and subset enumeration were fixed before these results were observed.
    recipes={'previous':{'previous':1}}
    for count in range(1,len(variants)+1):
        for names in itertools.combinations(variants,count):
            label='+'.join(names)
            recipes[label]={k:1/count for k in names}
            for previous_weight in (.25,.5,.75):
                recipes[f'previous{previous_weight}_{label}']={
                    'previous':previous_weight,**{k:(1-previous_weight)/count for k in names}}
    for name,weights in list(recipes.items()):
        recipes['dense_'+name]={k+'_dense':w for k,w in weights.items()}
    # Reuse existing independent fine-model errors with one small globally fixed weight.
    for name,weights in list(recipes.items()):
        recipes['resnet10_'+name]={**{k:.9*w for k,w in weights.items()},'resnet_v4':.1}
    for name,directory in [('resnet_v4','resnet34_v4'),('convnext_v4','convnext_v4')]:
        for amount in (.1,.25,.5):
            recipes[f'reuse_{name}_{amount}']={'previous':1-amount,name:amount}
    results={name:dict(weights=w,metrics=summarize(records,blend(components,w))) for name,w in recipes.items()}
    selected=min(results,key=lambda name:results[name]['metrics']['rmse'])
    chosen=results[selected]
    # Leave-one-fold-out recipe selection measures sensitivity to the small validation set.
    crossfit={};fold_choices={}
    for fold in range(5):
        tr=[r for r in records if r['fold']!=fold]
        winner=min(recipes,key=lambda k:score(tr,blend(components,recipes[k])))
        pred=blend(components,recipes[winner]);fold_choices[fold]=winner
        crossfit.update({r['id']:pred[r['id']] for r in records if r['fold']==fold})
    selected_oof=blend(components,chosen['weights'])
    write(REPORT/'selection.json',dict(selected=selected,**chosen,candidates=results,
          recipe_crossfit_metrics=summarize(records,crossfit),recipe_crossfit_choices=fold_choices,
          warning='OOF model/checkpoint selection is optimistic; crossfit recipe choice does not remove checkpoint selection bias. New Kaggle score is unknown.'))
    write(REPORT/'checkpoint_replay_verification.json',dict(manifest_hash=manifest_hash(manifest),folds=verification))
    write(REPORT/'selected_oof.json',dict(manifest_hash=manifest_hash(manifest),predictions=selected_oof,metrics=chosen['metrics']))
    print(f'SELECTED {selected}: OOF RMSE {chosen["metrics"]["rmse"]:.6f}',flush=True)
    test=[r for r in manifest['records'] if r['split']=='test']
    def from_csv(path):
        p={r['id']:dict(xy=[0.,0.]) for r in test}
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                ident,axis=row['ImageID'].split('_Fovea_')
                p[int(ident)]['xy'][0 if axis=='X' else 1]=float(row['value'])
        return p
    baseline_csv=ROOT/'submissions/submission_baseline_kaggle14_20261021.csv'
    if hashlib.sha256(baseline_csv.read_bytes()).hexdigest()!=read(REPORT/'round_plan.json')['baseline_sha256']:
        raise ValueError('Protected baseline changed')
    test_components={'previous':from_csv(baseline_csv)}
    old_fallback=from_csv(ROOT/'submissions/submission_accuracy_5h.csv')
    identities=[]
    for name,directory in [('resnet_v4','resnet34_v4'),('convnext_v4','convnext_v4')]:
        if name not in chosen['weights']: continue
        from fovea.inference import predict_manifest
        pairs=[(ROOT/'runs'/directory/f'fold_{f}/coarse_best.pt',ROOT/'runs'/directory/f'fold_{f}/fine_best.pt') for f in range(5)]
        test_components[name]=predict_manifest(manifest,pairs,REPORT/(name+'_test.csv'),
                                              'cuda',tta=True,stage='fine',policy='integral_fixed')
        for fold,pair in enumerate(pairs):
            for stage,path in zip(('coarse','fine'),pair):
                identities.append(dict(variant=name,fold=fold,role=stage,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if 'previous_dense' in chosen['weights']:
        members={}
        for fold in range(5):
            previous_path=ROOT/f'runs/detector_v6/fold_{fold}/point_best.pt'
            detector=YOLO(str(previous_path))
            if detector.ckpt['fovea_fold']!=fold or detector.ckpt['fovea_manifest_hash']!=manifest_hash(manifest):
                raise ValueError('Previous test checkpoint identity mismatch')
            identities.append(dict(variant='detector_v6',fold=fold,role='previous_dense',
                epoch=detector.ckpt['epoch']+1,sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest()))
            members[str(fold)]=predict(detector,test,old_fallback,aggregation='dense')
            del detector;gc.collect();torch.cuda.empty_cache()
        test_components['previous_dense']=blend(members,{str(f):.2 for f in range(5)})
    for name in variants:
        dense=name+'_dense' in chosen['weights']
        if name not in chosen['weights'] and not dense: continue
        component=name+'_dense' if dense else name
        members={}
        for fold in range(5):
            centers=None
            if name=='roi':
                previous_path=ROOT/f'runs/detector_v6/fold_{fold}/point_best.pt'
                previous=YOLO(str(previous_path))
                if previous.ckpt['fovea_fold']!=fold or previous.ckpt['fovea_manifest_hash']!=manifest_hash(manifest):
                    raise ValueError('ROI-center test checkpoint identity mismatch')
                identities.append(dict(variant='detector_v6',fold=fold,role='roi_center',
                    epoch=previous.ckpt['epoch']+1,sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest()))
                centers=predict(previous,test,old_fallback)
                del previous;gc.collect();torch.cuda.empty_cache()
            path=JOB/name/f'fold_{fold}/point_best.pt'
            detector=YOLO(str(path))
            if detector.ckpt['fovea_fold']!=fold or detector.ckpt['fovea_manifest_hash']!=manifest_hash(manifest):
                raise ValueError('Final checkpoint identity mismatch')
            identities.append(dict(variant=name,fold=fold,epoch=detector.ckpt['epoch']+1,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            members[str(fold)]=predict(detector,test,old_fallback,size=variant_size(name),
                                      space=variant_space(name),centers=centers,aggregation='dense' if dense else 'top')
            del detector;gc.collect();torch.cuda.empty_cache()
        test_components[component]=blend(members,{str(f):.2 for f in range(5)})
        write(REPORT/f'{component}_test.json',dict(manifest_hash=manifest_hash(manifest),predictions=test_components[component]))
    final=blend(test_components,chosen['weights'])
    output=ROOT/'submissions/submission_final_round_v7.csv'
    if output.exists(): raise FileExistsError('Final CSV already exists; refusing overwrite')
    make_submission(manifest['template'],final,output,{r['id']:(r['width'],r['height']) for r in test})
    write(output.with_suffix('.diagnostics.json'),dict(predictions=final,selected=selected,weights=chosen['weights'],
          manifest_hash=manifest_hash(manifest),oof_rmse=chosen['metrics']['rmse'],checkpoints=identities,
          visibility_policy='All visible, consistent with all 80 supplied labels; no validated negative detector.'))
    rows=list(csv.DictReader(output.open(encoding='utf8',newline='')))
    template=list(csv.DictReader(Path(manifest['template']).open(encoding='utf-8-sig',newline='')))
    assert len(rows)==40 and [r['ImageID'] for r in rows]==[r['ImageID'] for r in template]
    assert all(np.isfinite(float(r['value'])) for r in rows)
    plan=read(REPORT/'round_plan.json')
    elapsed=(dt.datetime.now().astimezone()-dt.datetime.fromisoformat(plan['started_at'])).total_seconds()/60
    write(REPORT/'completion.json',dict(completed_at=dt.datetime.now().astimezone().isoformat(),selected=selected,
          oof_rmse=chosen['metrics']['rmse'],previous_oof_rmse=score(records,baseline),user_reported_previous_kaggle=14,
          new_kaggle_score=None,wall_minutes=elapsed,submission=str(output),rows=40,template_order_exact=True,
          sha256=hashlib.sha256(output.read_bytes()).hexdigest(),baseline_unchanged=True,checkpoints=identities))
    print(f'Wrote {output}; {elapsed:.1f} minutes since round plan',flush=True)
    from final_round_report import build_report
    build_report()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['probe','train','finish'])
    parser.add_argument('--variant',choices=EXPERIMENTS,default='fov')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    setup()
    if args.action=='probe': probe()
    elif args.action=='train': train(args.variant,args.resume)
    else: finish()


if __name__=='__main__': main()
