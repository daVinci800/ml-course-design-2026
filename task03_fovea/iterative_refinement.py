"""Inference-only held-out comparison; never uses GT as a crop center."""
import argparse,json,hashlib,shutil,time
from pathlib import Path
import numpy as np
import torch
from fovea.prepare import read_manifest,write_json
from fovea.train import manifest_hash
from fovea.evaluate import collect_oof,summarize
from fovea.inference import load_checkpoint,predict_records,make_submission
from fovea.data import stage_settings
from fovea.policies import policy_settings,refinement_windows
from fovea.ensemble import combine_predictions


RECIPES={'previous_winner':{'prior':1.},'repeat_once':{'repeat1':1.},'repeat_twice':{'repeat2':1.},
    'baseline_repeat_equal':{'baseline':.5,'repeat1':.5},'repeats_equal':{'repeat1':.5,'repeat2':.5},
    'prior_repeat_equal':{'prior':.5,'repeat1':.5}}


def saved_winner(manifest,baseline,name):
    root=Path(__file__).resolve().parent
    directory=root/'reports'/name
    selected=json.loads((directory/'selection.json').read_text(encoding='utf8'))
    if name=='optimization_v4':
        data=json.loads((directory/(selected['selected']+'_oof.json')).read_text(encoding='utf8'))
        assert data['manifest_hash']==manifest_hash(manifest)
        return {int(k):p for k,p in data['predictions'].items()}
    parent='optimization_v4' if name=='detector_v5' else 'detector_v5'
    components={'baseline':baseline,'v4':saved_winner(manifest,baseline,parent)}
    for part in ('detector','refined'):
        data=json.loads((directory/(part+'_oof.json')).read_text(encoding='utf8'))
        assert data['manifest_hash']==manifest_hash(manifest)
        components[part]={int(k):p for k,p in data['predictions'].items()}
    return combine_predictions(components,selected['weights'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute-inference',action='store_true')
    args=parser.parse_args()
    if not args.execute_inference:parser.error('Requires --execute-inference; this never trains.')
    root=Path(__file__).resolve().parent
    import os
    os.chdir(root)
    started=time.monotonic();torch.set_num_threads(4)
    parent=json.loads((root/'runs/detector_v6/status.json').read_text(encoding='utf8'))
    assert parent['phase']=='completed','Wait until training and parent inference finish.'
    report=root/'reports/iterative_refinement';report.mkdir(exist_ok=True)
    manifest=read_manifest(root/'prepared/manifest.json')
    records,baseline=collect_oof(manifest,root/'runs/accuracy_5h','fine')
    write_json(report/'predeclared_plan.json',dict(recipes=RECIPES,manifest_hash=manifest_hash(manifest),
        rule='Use corresponding-fold predicted fine points as crop centers, reuse the exact held-out fine checkpoint, one or two additional passes. Test follows the same per-fold cascade before averaging.',
        warning='An additional validation comparison after earlier experiments; selection remains optimistic.'))
    def clip(records,pred):
        return {r['id']:dict(xy=np.clip(pred[r['id']]['xy'],[0,0],[r['width']-1,r['height']-1]).tolist()) for r in records}
    def fine_model(fold):
        model,checkpoint=load_checkpoint(root/f'runs/accuracy_5h/fold_{fold}/fine_best.pt','cuda')
        assert checkpoint['manifest_hash']==manifest_hash(manifest) and checkpoint['fold']==fold
        settings=policy_settings(stage_settings(checkpoint['config'],'fine'),'integral_fixed')
        return model,settings
    def repeat(model,settings,records,pred):
        centers={i:p['xy'] for i,p in pred.items()}
        return clip(records,predict_records(model,records,'fine',settings,'cuda',centers=centers,tta=True))
    components={'baseline':baseline,'prior':saved_winner(manifest,baseline,'detector_v6'),'repeat1':{},'repeat2':{}}
    for fold in range(5):
        valid=[r for r in records if r['fold']==fold]
        model,settings=fine_model(fold)
        first=repeat(model,settings,valid,{r['id']:baseline[r['id']] for r in valid})
        second=repeat(model,settings,valid,first)
        components['repeat1'].update(first);components['repeat2'].update(second)
        del model;torch.cuda.empty_cache()
    candidates={name:dict(weights=weights,metrics=summarize(records,combine_predictions(components,weights))) for name,weights in RECIPES.items()}
    selected=min(candidates,key=lambda name:candidates[name]['metrics']['rmse'])
    chosen=candidates[selected]
    write_json(report/'selection.json',dict(selected=selected,**chosen,candidates=candidates))
    for key,points in components.items():
        write_json(report/(key+'_oof.json'),dict(predictions=points,manifest_hash=manifest_hash(manifest)))
    for name,candidate in candidates.items():print(name,candidate['metrics']['rmse'],flush=True)
    test=[r for r in manifest['records'] if r['split']=='test']
    import csv
    prior={r['id']:dict(xy=[0.,0.]) for r in test}
    with Path(parent['submission']).open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            ident,axis=row['ImageID'].split('_Fovea_');prior[int(ident)]['xy'][0 if axis=='X' else 1]=float(row['value'])
    test_components={'prior':prior}
    if set(chosen['weights'])-{'prior'}:
        members={'baseline':[],'repeat1':[],'repeat2':[]}
        for fold in range(5):
            coarse,checkpoint=load_checkpoint(root/f'runs/accuracy_5h/fold_{fold}/coarse_best.pt','cuda')
            assert checkpoint['manifest_hash']==manifest_hash(manifest) and checkpoint['fold']==fold
            coarse_pred=predict_records(coarse,test,'coarse',stage_settings(checkpoint['config'],'coarse'),'cuda',tta=True)
            del coarse;torch.cuda.empty_cache()
            fine,settings=fine_model(fold)
            centers,sides=refinement_windows(coarse_pred,settings)
            initial=clip(test,predict_records(fine,test,'fine',settings,'cuda',centers=centers,crop_sides=sides,tta=True))
            first=repeat(fine,settings,test,initial)
            second=repeat(fine,settings,test,first)
            for name,points in zip(members,(initial,first,second)):members[name].append(points)
            del fine;torch.cuda.empty_cache()
        for name,points in members.items():
            test_components[name]=combine_predictions({str(i):p for i,p in enumerate(points)},{str(i):.2 for i in range(5)})
    predictions=combine_predictions(test_components,chosen['weights'])
    submission=root/'submissions/submission_recommended_20261021.csv'
    make_submission(manifest['template'],predictions,submission,{r['id']:(r['width'],r['height']) for r in test})
    write_json(submission.with_suffix('.diagnostics.json'),dict(predictions=predictions,selected=selected,weights=chosen['weights'],oof_rmse=chosen['metrics']['rmse'],manifest_hash=manifest_hash(manifest)))
    desktop=Path('C:/Users/Lenovo/Desktop/2026task3_提交文件')/submission.name
    if desktop.exists():raise FileExistsError('Do not overwrite existing recommended submission.')
    shutil.copyfile(submission,desktop)
    assert hashlib.sha256(submission.read_bytes()).digest()==hashlib.sha256(desktop.read_bytes()).digest()
    write_json(report/'completion.json',dict(phase='completed',submission=str(submission),desktop_submission=str(desktop),selected=selected,
        oof_rmse=chosen['metrics']['rmse'],wall_seconds=time.monotonic()-started,sha256=hashlib.sha256(submission.read_bytes()).hexdigest()))


if __name__=='__main__':main()
