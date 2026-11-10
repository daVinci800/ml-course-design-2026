import csv
import json
from pathlib import Path
import numpy as np
import torch
from .data import load_image,to_tensor,stage_settings
from .cache import render_record
from .geometry import roi_matrix,warp,decode_norm
from .model import Localizer
from .prepare import write_json
from .decoding import decode_heatmap
from .policies import policy_settings,refinement_windows


def load_checkpoint(path,device):
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint.get('format')!='fovea-v1':
        raise ValueError(f'Unsupported checkpoint {path}')
    config = checkpoint['config']
    # Architecture/norm structure must match training, without triggering a download.
    model = Localizer(config['backbone'],pretrained=False,channels=config.get('decoder_channels',96))
    if config.get('pretrained',False) and config['backbone'].startswith('resnet'):
        from torchvision import models
        encoder = getattr(models,config['backbone'])(weights=None)
        encoder.fc=torch.nn.Identity()
        model.encoder=encoder
        model.freeze_bn=True
    model.load_state_dict(checkpoint['model'])
    model=model.to(device)
    if config.get('channels_last',False) and torch.device(device).type=='cuda':
        model=model.to(memory_format=torch.channels_last)
    return model.eval(),checkpoint


@torch.inference_mode()
def predict_records(model,records,stage,stage_config,device,centers=None,tta=True,crop_sides=None,decoder_methods=None):
    methods=tuple(decoder_methods) if decoder_methods is not None else (stage_config.get('decoder','integral'),)
    if not methods or len(set(methods))!=len(methods) or any(m not in ('integral','dark') for m in methods):
        raise ValueError('Decoder methods must be nonempty, unique integral/dark values.')
    samples={method:{r['id']:dict(xy=[],presence=[]) for r in records} for method in methods}
    batch=[]
    n=stage_config['size']
    batch_size=stage_config.get('inference_batch_size',1)
    cuda=torch.device(device).type=='cuda'
    def flush():
        if not batch:
            return
        tensor=torch.stack([item[2] for item in batch]).to(device)
        if cuda:
            tensor=tensor.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=torch.device(device).type,enabled=cuda):
            output=model(tensor)
        presence=output['presence'].float().sigmoid().cpu().tolist()
        for decoder in methods:
            norm=(output['xy'] if decoder=='integral' else decode_heatmap(
                output['logits'],decoder,stage_config.get('dark_blur_sigma',1.0))).float().cpu().numpy()
            for j,(ident,matrix,_) in enumerate(batch):
                samples[decoder][ident]['xy'].append(decode_norm(norm[j],matrix,n))
                samples[decoder][ident]['presence'].append(presence[j])
        batch.clear()
    for r in records:
        if stage=='coarse':
            center,side=r['fov_center'],r['fov_side']
        else:
            center,side=centers[r['id']],crop_sides[r['id']] if crop_sides else stage_config['crop_side']
        views=[(1.0,False),(1.0,True),(0.90,False),(1.10,False)] if tta else [(1.0,False)]
        for scale,flip in views:
            matrix=roi_matrix(center,side*scale,n,flip=flip)
            image=render_record(r,matrix,n,stage_config.get('cache_dir'),stage_config.get('coarse_cache',False))
            batch.append((r['id'],matrix,to_tensor(image)))
            if len(batch)>=batch_size:
                flush()
    flush()
    predictions={method:{} for method in methods}
    for method in methods:
        for ident,values in samples[method].items():
            xy,presence=values['xy'],values['presence']
            predictions[method][ident]=dict(xy=np.mean(xy,axis=0).tolist(),
                                      tta_spread=float(np.sqrt(np.mean(np.sum((np.array(xy)-np.mean(xy,axis=0))**2,axis=1)))),
                                      presence=float(np.mean(presence)))
    return predictions if decoder_methods is not None else predictions[methods[0]]


def make_submission(template,predictions,path,dimensions):
    rows=list(csv.DictReader(Path(template).open(encoding='utf-8-sig',newline='')))
    keys=[r['ImageID'] for r in rows]
    expected={f'{ident}_Fovea_{axis}' for ident in predictions for axis in ('X','Y')}
    if len(keys)!=len(expected) or set(keys)!=expected or len(set(keys))!=len(keys):
        raise ValueError('Prediction IDs do not exactly match the submission template.')
    for ident,p in predictions.items():
        xy=np.asarray(p['xy'] if isinstance(p,dict) else p,dtype=float)
        if xy.shape!=(2,) or not np.isfinite(xy).all():
            raise ValueError(f'Nonfinite/invalid prediction: {ident}')
        w,h=dimensions[ident]
        if not (0<=xy[0]<=w-1 and 0<=xy[1]<=h-1):
            raise ValueError(f'Out of bounds prediction: {ident}: {xy}')
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['ImageID','value'])
        writer.writeheader()
        for row in rows:
            ident,axis=row['ImageID'].split('_Fovea_')
            p=predictions[int(ident)]
            xy=p['xy'] if isinstance(p,dict) else p
            writer.writerow(dict(ImageID=row['ImageID'],value=f'{float(xy[0 if axis=="X" else 1]):.6f}'))


def predict_manifest(manifest,pairs,output,device='cuda',tta=True,stage='fine',policy='checkpoint'):
    records=[r for r in manifest['records'] if r['split']=='test']
    if not pairs:
        raise ValueError('At least one coarse/fine checkpoint pair is required.')
    members=[]
    from .train import manifest_hash
    for coarse_path,fine_path in pairs:
        coarse,checkpoint=load_checkpoint(coarse_path,device)
        if checkpoint['stage']!='coarse':
            raise ValueError('First checkpoint must be a coarse model.')
        if checkpoint['manifest_hash']!=manifest_hash(manifest):
            raise ValueError('Checkpoint does not match the current images, labels, or folds.')
        if checkpoint.get('presence_enabled'):
            raise ValueError('This checkpoint contains a presence classifier; real-negative calibration is required before deployment.')
        cp=predict_records(coarse,records,'coarse',stage_settings(checkpoint['config'],'coarse'),device,tta=tta)
        del coarse
        if stage=='coarse':
            members.append({i:dict(xy=p['xy'],coarse_xy=p['xy'],tta_spread=p['tta_spread']) for i,p in cp.items()})
            continue
        fine,refine=load_checkpoint(fine_path,device)
        if refine['stage']!='fine' or refine['fold']!=checkpoint['fold']:
            raise ValueError('Fine/coarse stage or fold mismatch.')
        if refine['manifest_hash']!=checkpoint['manifest_hash']:
            raise ValueError('Coarse and fine models used different manifests.')
        # Enforce exactly the coarse checkpoint used to evaluate this fine model.
        import hashlib
        if refine.get('coarse_sha256')!=hashlib.sha256(Path(coarse_path).read_bytes()).hexdigest():
            raise ValueError('Fine model belongs to a different coarse checkpoint.')
        if refine.get('presence_enabled') or checkpoint.get('presence_enabled'):
            raise ValueError('Presence classification requires independently validated calibration before deployment; no threshold is supplied.')
        settings=policy_settings(stage_settings(refine['config'],'fine'),policy)
        if settings.get('adaptive_crop') and not tta:
            raise ValueError('Adaptive windows require coarse TTA.')
        centers,sides=refinement_windows(cp,settings)
        fp=predict_records(fine,records,'fine',settings,device,
                           centers=centers,crop_sides=sides,tta=tta)
        members.append({i:dict(xy=p['xy'],coarse_xy=cp[i]['xy'],tta_spread=p['tta_spread']) for i,p in fp.items()})
        del fine
    result={}
    for r in records:
        ident=r['id']
        points=np.array([p[ident]['xy'] for p in members])
        point=points.mean(0)
        # Bound small numerical extrapolations to valid pixel centers.
        point=np.clip(point,[0,0],[r['width']-1,r['height']-1])
        spread=float(np.sqrt(np.mean(np.sum((points-points.mean(0))**2,axis=1))))
        shift=float(np.mean([np.linalg.norm(np.array(p[ident]['xy'])-p[ident]['coarse_xy']) for p in members]))
        tta_spread=float(np.mean([p[ident]['tta_spread'] for p in members]))
        result[ident]=dict(xy=point.tolist(),ensemble_spread=spread,tta_spread=tta_spread,
                          coarse_to_fine_shift=shift,flag=spread>40 or tta_spread>30 or shift>160,
                          visibility_policy='all-visible: no real negative training examples')
    output=Path(output)
    make_submission(manifest['template'],result,output,{r['id']:(r['width'],r['height']) for r in records})
    write_json(output.with_suffix('.diagnostics.json'),dict(predictions=result,
               checkpoints=[[str(a),str(b) if b else None] for a,b in pairs],tta=tta,stage=stage,policy=policy,
               warning='Flags are for model diagnostics only. Competition rules prohibit hand-labeling test coordinates or visibility.'))
    print(f'Wrote {2*len(records)} rows to {output}')
    return result
