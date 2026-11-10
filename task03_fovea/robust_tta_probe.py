"""Inference-only test of fixed symmetric TTA; uses held-out training images."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    os.chdir(root);sys.stdout.reconfigure(encoding='utf8')
    import numpy as np
    import torch
    from fovea.prepare import read_manifest,write_json
    from fovea.train import manifest_hash
    from fovea.data import stage_settings,to_tensor
    from fovea.cache import render_record
    from fovea.geometry import roi_matrix,decode_norm
    from fovea.inference import load_checkpoint
    from fovea.evaluate import summarize
    torch.set_num_threads(4)
    device=torch.device(args.device)
    manifest=read_manifest('prepared/manifest.json')
    records=[r for r in manifest['records'] if r['split']=='train']
    destination=root/'reports/robust_tta'
    methods=('mean4_control','symmetric_mean6','robust_median6')
    write_json(destination/'predeclared_plan.json',dict(methods=methods,device=str(device),
        view_scales=[1.,1.,.9,.9,1.1,1.1],view_flips=[False,True,False,True,False,True],
        note='No retraining or test GT. CPU OOF uses FP32; device is retained for later matching inference.'))
    all_predictions={m:{} for m in methods}
    start=time.monotonic()
    @torch.inference_mode()
    def views(model,valid,settings,centers=None):
        n=settings['size'];batch=[]
        outputs={r['id']:[] for r in valid}
        def flush():
            if not batch:return
            tensor=torch.stack([t for _,_,t in batch]).to(device)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                result=model(tensor)
            points=result['xy'].float().cpu().numpy()
            for (ident,matrix,_),point in zip(batch,points):outputs[ident].append(decode_norm(point,matrix,n))
            batch.clear()
        for r in valid:
            center=centers[r['id']] if centers else r['fov_center']
            side=settings['crop_side'] if centers else r['fov_side']
            for scale,flip in [(1.,False),(1.,True),(.9,False),(.9,True),(1.1,False),(1.1,True)]:
                a=roi_matrix(center,side*scale,n,flip=flip)
                image=render_record(r,a,n,settings.get('cache_dir'),settings.get('coarse_cache',False))
                batch.append((r['id'],a,to_tensor(image)))
                if len(batch)>=2:flush()
        flush()
        return {i:np.array(points) for i,points in outputs.items()}
    def aggregate(points,method):
        return np.median(points,axis=0) if method=='robust_median6' else points[[0,1,2,4]].mean(0) if method=='mean4_control' else points.mean(0)
    for fold in range(5):
        valid=[r for r in records if r['fold']==fold]
        directory=root/'runs/accuracy_5h'/f'fold_{fold}'
        coarse_path,fine_path=directory/'coarse_best.pt',directory/'fine_best.pt'
        coarse,cp=load_checkpoint(coarse_path,device)
        assert cp['fold']==fold and cp['stage']=='coarse' and cp['manifest_hash']==manifest_hash(manifest)
        cv=views(coarse,valid,stage_settings(cp['config'],'coarse'))
        del coarse
        fine,fp=load_checkpoint(fine_path,device)
        assert fp['fold']==fold and fp['stage']=='fine' and fp['manifest_hash']==cp['manifest_hash']
        assert fp['coarse_sha256']==hashlib.sha256(coarse_path.read_bytes()).hexdigest()
        for method in methods:
            centers={i:aggregate(points,method) for i,points in cv.items()}
            fv=views(fine,valid,stage_settings(fp['config'],'fine'),centers)
            pred={i:dict(xy=aggregate(points,method).tolist()) for i,points in fv.items()}
            all_predictions[method].update(pred)
            write_json(destination/method/f'fold_{fold}'/'fine_oof.json',dict(
                stage='fine',fold=fold,manifest_hash=manifest_hash(manifest),tta=True,predictions=pred))
        del fine
        print(f'Completed CPU robust TTA fold {fold}; elapsed {(time.monotonic()-start)/60:.1f} min',flush=True)
    report={method:summarize(records,pred) for method,pred in all_predictions.items()}
    write_json(destination/'comparison.json',dict(methods=report,seconds=time.monotonic()-start,
        warnings=['Same checkpoint-selection folds; these exploratory comparisons add selection bias.','No test score is known.']))
    print(json.dumps({method:metrics['rmse'] for method,metrics in report.items()},indent=2),flush=True)


if __name__=='__main__':main()
