import csv
import json
from pathlib import Path
import numpy as np
from .prepare import write_json


def collect_oof(manifest,run_dir,stage='fine'):
    records=[r for r in manifest['records'] if r['split']=='train']
    expected={r['id']:r for r in records}
    from .train import manifest_hash
    predictions={}
    for fold in range(manifest['n_folds']):
        path=Path(run_dir)/f'fold_{fold}'/f'{stage}_oof.json'
        content=json.loads(path.read_text(encoding='utf-8'))
        if content['manifest_hash']!=manifest_hash(manifest):
            raise ValueError('Manifest mismatch in OOF file.')
        if content['stage']!=stage or content['fold']!=fold:
            raise ValueError('Wrong OOF stage/fold.')
        for key,value in content['predictions'].items():
            ident=int(key)
            if ident in predictions or ident not in expected or expected[ident]['fold']!=fold:
                raise ValueError(f'OOF leakage, duplicate, or unknown ID {ident}')
            if not np.isfinite(value['xy']).all():
                raise ValueError('Nonfinite OOF prediction.')
            predictions[ident]=value
    if set(predictions)!=set(expected):
        raise ValueError('Incomplete OOF predictions.')
    return records,predictions


def summarize(records,predictions,seed=20261020):
    error=np.array([np.array(predictions[r['id']]['xy'])-r['xy'] for r in records])
    ed=np.sqrt((error**2).sum(1))
    squared=(error**2).mean(1)
    groups=sorted({r['group'] for r in records})
    group_indices=[[j for j,r in enumerate(records) if r['group']==g] for g in groups]
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(2000):
        indices=np.concatenate([group_indices[g] for g in rng.integers(0,len(groups),len(groups))])
        samples.append(float(np.sqrt(squared[indices].mean())))
    per_fold={str(f):float(np.sqrt(np.mean([squared[j] for j,r in enumerate(records) if r['fold']==f]))) for f in sorted({r['fold'] for r in records})}
    per_camera={str(w):float(np.sqrt(np.mean([squared[j] for j,r in enumerate(records) if r['width']==w]))) for w in sorted({r['width'] for r in records})}
    # The split experiment demonstrates how volatile a 10-image private board can be.
    half_scores=[float(np.sqrt(squared[rng.choice(len(records),min(10,len(records)),replace=False)].mean())) for _ in range(2000)]
    return dict(rmse=float(np.sqrt(squared.mean())),rmse_x=float(np.sqrt((error[:,0]**2).mean())),
                rmse_y=float(np.sqrt((error[:,1]**2).mean())),mean_euclidean=float(ed.mean()),
                median_euclidean=float(np.median(ed)),p95_euclidean=float(np.quantile(ed,0.95)),
                max_euclidean=float(ed.max()),sdr_20=float(np.mean(ed<=20)),sdr_40=float(np.mean(ed<=40)),
                per_fold_rmse=per_fold,per_camera_rmse=per_camera,
                approximate_group_bootstrap_95=np.quantile(samples,[0.025,0.975]).tolist(),
                simulated_10_image_rmse_5_95=np.quantile(half_scores,[0.05,0.95]).tolist(),
                warnings=['OOF checkpoint selection uses these folds; confidence intervals do not remove model-selection bias.',
                          'Ten-image simulation is a sensitivity diagnostic, not a confidence interval for the actual private leaderboard.',
                          'No real negative examples: visibility sensitivity/specificity cannot be estimated.'])


def evaluate(manifest,run_dirs,output,stage='fine',weights=None):
    collections=[collect_oof(manifest,d,stage) for d in run_dirs]
    if weights is None:
        weights=np.ones(len(collections))/len(collections)
    weights=np.asarray(weights,dtype=float)
    if len(weights)!=len(collections) or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
        raise ValueError('Invalid ensemble weights.')
    weights=weights/weights.sum()
    records=collections[0][0]
    predictions={r['id']:dict(xy=np.sum([w*np.array(p[r['id']]['xy']) for w,(_,p) in zip(weights,collections)],axis=0).tolist()) for r in records}
    report=summarize(records,predictions)
    report.update(stage=stage,runs=[str(d) for d in run_dirs],weights=weights.tolist())
    output=Path(output)
    write_json(output,report)
    with output.with_suffix('.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','fold','group','gt_x','gt_y','pred_x','pred_y','euclidean_error'])
        writer.writeheader()
        for r in records:
            xy=predictions[r['id']]['xy']
            writer.writerow(dict(id=r['id'],fold=r['fold'],group=r['group'],gt_x=r['xy'][0],gt_y=r['xy'][1],
                                 pred_x=xy[0],pred_y=xy[1],euclidean_error=float(np.linalg.norm(np.array(xy)-r['xy']))))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report
