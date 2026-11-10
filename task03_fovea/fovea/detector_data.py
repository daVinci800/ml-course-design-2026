"""Use provided training boxes; no external retinal labels or test annotations."""
import json
from pathlib import Path
import shutil
import numpy as np


def yolo_box(box,width,height):
    x1,y1,x2,y2=map(float,box)
    if not (0<=x1<x2<=width and 0<=y1<y2<=height):
        raise ValueError('Invalid source box.')
    return [(x1+x2)/(2*width),(y1+y2)/(2*height),(x2-x1)/width,(y2-y1)/height]


def prepare_detector_folds(manifest,destination):
    root=Path(destination).resolve()
    images,labels=root/'images',root/'labels'
    images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
    records=[r for r in manifest['records'] if r['split']=='train']
    paths={}
    for r in records:
        if not r['visible'] or not r.get('box'):
            raise ValueError('This detector experiment requires visible, boxed training records.')
        name=f'{r["id"]:04d}'
        target=images/(name+'.jpg')
        if not target.exists():shutil.copyfile(r['path'],target)
        label=yolo_box(r['box'],r['width'],r['height'])
        (labels/(name+'.txt')).write_text('0 '+' '.join(f'{v:.10f}' for v in label)+'\n',encoding='utf8')
        paths[r['id']]=target.as_posix()
    yamls=[]
    for fold in range(manifest['n_folds']):
        training=[r for r in records if r['fold']!=fold]
        valid=[r for r in records if r['fold']==fold]
        if {r['group'] for r in training}&{r['group'] for r in valid}:
            raise ValueError('Detector fold leakage.')
        (root/f'train_{fold}.txt').write_text('\n'.join(paths[r['id']] for r in training)+'\n',encoding='utf8')
        (root/f'val_{fold}.txt').write_text('\n'.join(paths[r['id']] for r in valid)+'\n',encoding='utf8')
        # JSON is valid YAML and avoids Windows path-escaping ambiguity.
        yaml=root/f'fold_{fold}.yaml'
        yaml.write_text(json.dumps(dict(path=root.as_posix(),train=f'train_{fold}.txt',val=f'val_{fold}.txt',names=['macula']),indent=2),encoding='utf8')
        yamls.append(yaml)
    return yamls


def detector_points(results,records,fallback):
    if len(results)!=len(records):raise ValueError('Detector output length mismatch.')
    predictions={}
    for result,r in zip(results,records):
        ident=r['id']
        box=result.boxes
        xy=None
        if box is not None and len(box):
            coords=box.xyxy.detach().float().cpu().numpy()
            scores=box.conf.detach().float().cpu().numpy()
            valid=np.isfinite(coords).all(1)&np.isfinite(scores)
            if valid.any():
                k=np.where(valid)[0][np.argmax(scores[valid])]
                xy=(coords[k,:2]+coords[k,2:])/2
        missing=xy is None
        if missing:
            if ident not in fallback:raise ValueError('Missing automatic detector fallback.')
            xy=np.asarray(fallback[ident]['xy'],dtype=float)
        xy=np.clip(xy,[0,0],[r['width']-1,r['height']-1])
        predictions[ident]=dict(xy=xy.tolist(),detector_miss=missing)
    return predictions
