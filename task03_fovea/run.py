"""CLI. Training requires BOTH the train command and --execute-training."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='Fovea localization: audit, grouped CV, coarse/fine heatmaps, submission validation.')
    commands=parser.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare',help='Audit images/labels and write fixed grouped folds; no training.')
    prepare.add_argument('--data-root',required=True)
    prepare.add_argument('--out',default='prepared')
    prepare.add_argument('--seed',type=int,default=20261020)
    prepare.add_argument('--n-folds',type=int,default=5)
    prepare.add_argument('--groups-csv',help='Optional known patient groups, columns id,group; never inferred from test labels.')
    cache=commands.add_parser('cache',help='Decode images once into lossless mmap caches; no model or training.')
    cache.add_argument('--manifest',default='prepared/manifest.json')
    cache.add_argument('--out',default='../../work/fovea_cache_v2')
    train=commands.add_parser('train',help='Without --execute-training this only prints the proposed run.')
    train.add_argument('--manifest',default='prepared/manifest.json')
    train.add_argument('--config',default='configs/fast_5h.json')
    train.add_argument('--run-dir',default='runs/fast_5h')
    train.add_argument('--folds',type=int,nargs='+',default=[0,1,2,3,4])
    train.add_argument('--stages',choices=['coarse','fine'],nargs='+',default=['coarse','fine'])
    train.add_argument('--device',default='cuda')
    train.add_argument('--resume',action='store_true')
    train.add_argument('--execute-training',action='store_true')
    evaluate=commands.add_parser('evaluate',help='Combine only strictly held-out predictions, report raw-pixel RMSE.')
    evaluate.add_argument('--manifest',default='prepared/manifest.json')
    evaluate.add_argument('--run-dirs',nargs='+',required=True)
    evaluate.add_argument('--out',default='reports/oof_metrics.json')
    evaluate.add_argument('--stage',choices=['coarse','fine'],default='fine')
    evaluate.add_argument('--weights',type=float,nargs='+')
    ablation=commands.add_parser('ablate',help='Inference only: compare four held-out decoding/crop policies; requires trained folds.')
    ablation.add_argument('--manifest',default='prepared/manifest.json')
    ablation.add_argument('--run-dir',required=True)
    ablation.add_argument('--out',default='reports/policy_ablation')
    ablation.add_argument('--device',default='cuda')
    predict=commands.add_parser('predict',help='Use existing matching checkpoints to make a template-ordered submission.')
    predict.add_argument('--manifest',default='prepared/manifest.json')
    predict.add_argument('--run-dirs',nargs='+',required=True)
    predict.add_argument('--out',default='submissions/submission.csv')
    predict.add_argument('--device',default='cuda')
    predict.add_argument('--no-tta',action='store_true')
    predict.add_argument('--stage',choices=['coarse','fine'],default='fine',help='Use the cascade only if OOF supports refinement.')
    predict.add_argument('--policy',choices=['checkpoint','integral_fixed','dark_fixed','integral_adaptive','dark_adaptive'],default='checkpoint',help='Fine-stage policy; compare with ablate before changing this.')
    args=parser.parse_args()
    if args.command=='prepare':
        from fovea.prepare import prepare
        prepare(args.data_root,args.out,args.seed,args.n_folds,args.groups_csv)
        return
    from fovea.prepare import read_manifest
    manifest=read_manifest(args.manifest)
    if args.command=='cache':
        from fovea.cache import build_cache
        print(json.dumps(build_cache(manifest['records'],args.out),indent=2))
        return
    if args.command=='train':
        config=json.loads(Path(args.config).read_text(encoding='utf-8'))
        if not args.execute_training:
            from fovea.budget import planned_steps
            print(json.dumps(dict(status='DRY RUN: no model initialization, weight download, or training',
                folds=args.folds,stages=args.stages,run_dir=args.run_dir,
                plan=planned_steps(manifest,config,args.folds,args.stages),config=config),ensure_ascii=False,indent=2))
            return
        from fovea.train import train_folds
        train_folds(manifest,config,args.folds,args.run_dir,args.device,args.resume,args.stages)
    elif args.command=='evaluate':
        from fovea.evaluate import evaluate
        evaluate(manifest,args.run_dirs,args.out,args.stage,args.weights)
    elif args.command=='ablate':
        from fovea.ablate import ablate
        ablate(manifest,args.run_dir,args.out,args.device)
    elif args.command=='predict':
        from fovea.inference import predict_manifest
        pairs=[(Path(d)/f'fold_{fold}'/'coarse_best.pt',Path(d)/f'fold_{fold}'/'fine_best.pt' if args.stage=='fine' else None)
               for d in args.run_dirs for fold in range(manifest['n_folds'])]
        missing=[str(p) for pair in pairs for p in pair if p is not None and not p.exists()]
        if missing:
            raise FileNotFoundError('Train all folds first; missing checkpoints:\n'+'\n'.join(missing))
        if args.no_tta and 'adaptive' in args.policy:
            raise ValueError('Adaptive windows require coarse TTA; omit --no-tta.')
        predict_manifest(manifest,pairs,args.out,args.device,tta=not args.no_tta,stage=args.stage,policy=args.policy)


if __name__=='__main__':
    main()
