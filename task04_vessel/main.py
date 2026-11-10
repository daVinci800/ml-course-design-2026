"""Explicit command-line entry point; no command trains implicitly."""
import argparse
from pathlib import Path


def parser():
    p=argparse.ArgumentParser(description='Retinal vessel segmentation; training requires train --execute')
    sub=p.add_subparsers(dest='command',required=True)
    common=argparse.ArgumentParser(add_help=False)
    common.add_argument('--data',default='data/segmentation')
    common.add_argument('--config',default='configs/main.json')
    common.add_argument('--splits',default='artifacts/splits.json')
    common.add_argument('--run-dir',default='runs/main')
    common.add_argument('--device',default='auto')
    prep=sub.add_parser('prepare',parents=[common],help='Check data and write whole-image splits; no training')
    prep.add_argument('--zip',help='Optional ZIP to extract into --data parent; existing data are not overwritten')
    sub.add_parser('check',parents=[common],help='No-grad forward/loss/data check; never optimize or save weights')
    train=sub.add_parser('train',parents=[common])
    train.add_argument('--fold',type=int,required=True)
    train.add_argument('--execute',action='store_true',help='Explicitly unlock training')
    train.add_argument('--resume',action='store_true')
    cal=sub.add_parser('calibrate',parents=[common])
    cal.add_argument('--tta',action='store_true')
    cal.add_argument('--metric',choices=['native','official-proxy','nearest-proxy','background-proxy'],default='official-proxy')
    pred=sub.add_parser('predict',parents=[common])
    pred.add_argument('--tta',action='store_true')
    pred.add_argument('--output',required=True)
    pred.add_argument('--csv-foreground',choices=['vessel','background'],required=True,help='Required because prompt and supplied labels disagree')
    conv=sub.add_parser('submit',help='Convert already generated binary PNGs; equivalent to official RLE')
    conv.add_argument('--masks',required=True); conv.add_argument('--sample',required=True); conv.add_argument('--output',required=True)
    return p


def prepare(args,config):
    import hashlib,json,zipfile
    from PIL import Image
    import numpy as np
    from vessel.data import make_folds
    from vessel.runtime import training_fingerprint,save_json
    from vessel.rle import read_ids
    root=Path(args.data)
    if args.zip:
        if root.exists(): raise FileExistsError('Data directory already exists; extraction will not overwrite it')
        parent=root.resolve()
        with zipfile.ZipFile(args.zip) as z:
            for member in z.infolist():
                from pathlib import PurePosixPath
                parts=PurePosixPath(member.filename).parts
                if not parts or parts[0]!='segmentation':
                    raise ValueError('Expected archive top-level directory: segmentation')
                dest=parent.joinpath(*parts[1:]).resolve()
                if not dest.is_relative_to(parent): raise ValueError('Unsafe archive path')
                if not member.is_dir(): dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(z.read(member))
    train=list((root/'train/image').glob('*.jpg')); labels=list((root/'train/label').glob('*.jpg')); test=list((root/'test/image').glob('*.jpg'))
    ids=read_ids(root/'sample_submission.csv')
    if len(train)!=20 or len(labels)!=20 or len(test)!=20: raise ValueError('Expected 20 train images, 20 labels, 20 test images')
    if {p.stem for p in train}!={p.stem for p in labels}: raise ValueError('Unpaired labels')
    if set(ids)!={p.stem for p in test}: raise ValueError('Test/sample IDs differ')
    records=[]
    for p in sorted(train,key=lambda p:int(p.stem)):
        label=root/'train/label'/p.name
        with Image.open(p) as image, Image.open(label) as target:
            if image.size!=target.size: raise ValueError(f'Size mismatch: {p.name}')
            a=np.asarray(target.convert('L')); frac=float((a>=128).mean())
            if not 0.01<frac<0.3: raise ValueError(f'Unexpected label polarity/area at {p.name}: {frac}')
            records.append({'id':p.stem,'size':list(image.size),'image_actual_format':image.format,'label_actual_format':target.format,
                            'vessel_fraction':frac,'label_values':np.unique(a).tolist()})
    trhash={hashlib.sha256(p.read_bytes()).hexdigest() for p in train}
    if any(hashlib.sha256(p.read_bytes()).hexdigest() in trhash for p in test): raise ValueError('Train/test duplicate images')
    splits=make_folds(root,config['folds'],config['seed']); splits['training_fingerprint']=training_fingerprint(root)
    splitpath=Path(args.splits)
    if splitpath.exists() and json.loads(splitpath.read_text(encoding='utf-8'))!=splits:
        raise FileExistsError('Existing split manifest differs; choose another --splits path')
    save_json(splitpath,splits)
    save_json(splitpath.with_name('data_check.json'),{'train':records,'test_images':len(test),'training_started':False,'labels':'white=vascular foreground','prompt_conflict':True})
    print(f'Data checked. Splits saved to {splitpath.resolve()}. Training NOT started.')


def check(args,config):
    import json,torch,numpy as np
    from vessel.data import load_rgb,load_label,preprocess,augment,TrainDataset,read_folds
    from vessel.runtime import choose_device,save_json
    from vessel.model import build_model
    from vessel.losses import segmentation_loss
    from vessel.inference import predict_probability
    from vessel.rle import encode_rle,decode_rle
    device=choose_device(args.device); torch.manual_seed(config['seed'])
    model=build_model(config).to(device).eval()
    x=preprocess(load_rgb(Path(args.data)/'train/image/1.jpg'),config)
    y=torch.from_numpy(load_label(Path(args.data)/'train/label/1.jpg')[None,None]).to(device)
    before={k:v.clone() for k,v in model.state_dict().items()}
    with torch.inference_mode():
        outputs=model(x[None].to(device))
        loss,details=segmentation_loss(outputs,y,config,config['cldice_start_epoch']+config['cldice_ramp_epochs'])
    assert outputs['logits'].shape==y.shape and torch.isfinite(loss)
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    mask=np.zeros((512,512),bool); mask[0,:3]=True; mask[-1,-1]=True
    assert np.array_equal(decode_rle(encode_rle(mask)),mask)
    splits=read_folds(args.splits); dataset=TrainDataset(args.data,splits['folds'][0]['train_ids'],config)
    samples=[dataset[i] for i in range(4)]
    assert len({tuple(a.shape) for a,b in samples})==1
    assert all(set(torch.unique(b).tolist()).issubset({0.,1.}) for a,b in samples)
    report={'status':'passed','device':str(device),'torch':torch.__version__,'parameters':sum(p.numel() for p in model.parameters()),
            'input_shape':list(x.shape),'output_shape':list(outputs['logits'].shape),'finite_loss':float(loss),'loss_details':details,
            'training_started':False,'optimizer_steps':0,'backward_calls':0,'weights_unchanged':True,
            'note':'Loss is from random initialization and has no predictive-performance meaning.'}
    save_json(Path(args.splits).parent/'smoke_check.json',report); print(json.dumps(report,indent=2))


def main():
    args=parser().parse_args()
    if args.command=='submit':
        from vessel.rle import masks_to_csv
        masks_to_csv(args.masks,args.sample,args.output); print(f'CSV created: {args.output}'); return
    # Reject training before importing model/data code or touching checkpoint directories.
    if args.command=='train' and not args.execute:
        raise SystemExit('Training is locked: no work performed. Add --execute only when ready to train.')
    from vessel.runtime import load_config
    config=load_config(args.config)
    if args.command=='prepare': prepare(args,config)
    elif args.command=='check': check(args,config)
    elif args.command=='train':
        from vessel.train import run_train
        run_train(args,config)
    elif args.command=='calibrate':
        from vessel.pipeline import calibrate
        calibrate(args)
    elif args.command=='predict':
        from vessel.pipeline import predict
        predict(args)


if __name__=='__main__':
    main()
