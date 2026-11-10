"""Queued five-fold detector experiment; final selection still uses OOF only."""
import argparse,copy,ctypes,datetime,hashlib,json,os,shutil,sys,time,traceback,gc
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute-training',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Resume a paused queue before its first detector fold starts.')
    parser.add_argument('--variant',choices=['standard','low_lr'],default='standard')
    args=parser.parse_args()
    if not args.execute_training:parser.error('Explicit --execute-training is required.')
    root=Path(__file__).resolve().parent;os.chdir(root)
    sys.stdout.reconfigure(encoding='utf8');sys.stderr.reconfigure(encoding='utf8')
    work=root.parent.parent/'work'
    config_dir=work/'ultralytics_config';config_dir.mkdir(exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(config_dir)
    os.environ['MPLCONFIGDIR']=str(work/'matplotlib_config')
    os.environ['ULTRALYTICS_AUTOINSTALL']='false'
    font_dir=config_dir/'Ultralytics';font_dir.mkdir(exist_ok=True)
    font=Path('C:/Windows/Fonts/arial.ttf')
    if font.exists() and not (font_dir/'Arial.ttf').exists():shutil.copyfile(font,font_dir/'Arial.ttf')
    from fovea.prepare import read_manifest,write_json
    manifest=read_manifest('prepared/manifest.json')
    low_lr=args.variant=='low_lr'
    run_name='detector_v6' if low_lr else 'detector_v5'
    job=root/'runs'/run_name;report_dir=root/'reports'/run_name
    learning_rate=.0001 if low_lr else .001
    bias_warmup=0. if low_lr else .1
    job.mkdir(parents=True,exist_ok=True);report_dir.mkdir(parents=True,exist_ok=True)
    status_path=job/'status.json'
    if args.resume:
        previous=json.loads(status_path.read_text(encoding='utf-8-sig'))
        if previous['phase']!='paused' or list(job.glob('fold_*')):
            raise RuntimeError('This entry resumes only a paused detector queue without trained folds.')
    elif status_path.exists():raise FileExistsError('Detector job already exists.')
    lock=(job/'active.lock').open('x',encoding='utf8');lock.write(str(os.getpid()));lock.flush()
    started=time.monotonic();state=dict(pid=os.getpid(),started_at=datetime.datetime.now().astimezone().isoformat())
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.datetime.now().astimezone().isoformat(),wall_seconds=time.monotonic()-started,**extra)
        temporary=status_path.with_suffix('.tmp');write_json(temporary,state);temporary.replace(status_path)
        print(json.dumps(dict(phase=phase,**extra),ensure_ascii=False),flush=True)
    awake=False
    try:
        queue_status=root/('runs/detector_v5/status.json' if low_lr else 'runs/optimization_v4/status.json')
        status('queued_for_gpu',waiting_for=str(queue_status))
        while True:
            current=json.loads(queue_status.read_text(encoding='utf8'))
            if current['phase'] in ('completed','failed'):break
            if current['phase']=='paused':
                status('paused',pause_reason='Parent optimization paused; detector has not started.')
                return
            time.sleep(15)
        # Count active work only: overnight/user-requested pauses consume no budget.
        used=float(current.get('active_seconds',current.get('wall_seconds',0.)))
        if low_lr:
            parent=json.loads((root/'runs/optimization_v4/status.json').read_text(encoding='utf8'))
            post_parent=(datetime.datetime.fromisoformat(current['updated_at'])-datetime.datetime.fromisoformat(parent['updated_at'])).total_seconds()
            used=float(parent['active_seconds'])+max(0.,post_parent)
        deadline=time.time()+16200-300-used
        if time.time()>deadline-300:raise TimeoutError('Shared optimization budget exhausted before detector training.')
        if os.name=='nt':awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        import numpy as np
        import torch
        import ultralytics
        from ultralytics import YOLO,settings
        from fovea.train import manifest_hash
        from fovea.evaluate import collect_oof,summarize
        from fovea.inference import make_submission,load_checkpoint,predict_records
        from fovea.data import stage_settings
        from fovea.policies import policy_settings
        from fovea.ensemble import combine_predictions
        from fovea.detector_data import prepare_detector_folds,detector_points
        torch.set_num_threads(4)
        settings.update(dict(sync=False,weights_dir=str(work/'pretrained_cache'),datasets_dir=str(work/'detector_data'),runs_dir=str(job)))
        yamls=prepare_detector_folds(manifest,work/'detector_data')
        records,baseline=collect_oof(manifest,root/'runs/accuracy_5h','fine')
        recipes=dict(baseline={'baseline':1.},detector={'detector':1.},baseline_detector={'baseline':.5,'detector':.5},
            v4={'v4':1.},v4_detector={'v4':.5,'detector':.5},
            detector_refined={'refined':1.},baseline_refined={'baseline':.5,'refined':.5},
            detector_and_refined={'detector':.5,'refined':.5},v4_refined={'v4':.5,'refined':.5})
        write_json(report_dir/'predeclared_plan.json',dict(recipes=recipes,manifest_hash=manifest_hash(manifest),
            pretrained_source='Official Ultralytics YOLO11s COCO detector; no external retinal task weights.',
            weights_sha256=hashlib.sha256((work/'pretrained_cache/yolo11s.pt').read_bytes()).hexdigest(),
            ultralytics=ultralytics.__version__,imgsz=960,epochs=160,batch=4,lr0=learning_rate,lrf=.01,warmup_bias_lr=bias_warmup,
            prior_selection='detector_v5 winner' if low_lr else 'optimization_v4 winner',
            confidence=.001,miss_policy='Automatic corresponding-fold baseline prediction; no zero threshold.',
            selection='Held-out point-coordinate RMSE of predicted box center, not detection mAP.'))
        components={'baseline':baseline};detector_predictions={};refined_predictions={}
        def refine(records,points,fold):
            model,checkpoint=load_checkpoint(root/f'runs/accuracy_5h/fold_{fold}/fine_best.pt','cuda')
            if checkpoint['manifest_hash']!=manifest_hash(manifest) or checkpoint['fold']!=fold:
                raise ValueError('Refinement checkpoint manifest/fold mismatch.')
            options=policy_settings(stage_settings(checkpoint['config'],'fine'),'integral_fixed')
            pred=predict_records(model,records,'fine',options,'cuda',centers={i:p['xy'] for i,p in points.items()},tta=True)
            del model;torch.cuda.empty_cache()
            return pred
        for fold in range(5):
            valid=[r for r in records if r['fold']==fold]
            allowance=(deadline-time.time())/(5-fold)
            if allowance<60:raise TimeoutError('Insufficient shared budget for all detector folds.')
            fold_deadline=time.time()+allowance
            directory=job/f'fold_{fold}';directory.mkdir(exist_ok=True)
            best_path=directory/'point_best.pt'
            history=[];seen=set();best=float('inf');best_pred=None
            status('training',fold=fold,fold_budget_seconds=allowance)
            model=YOLO(str(work/'pretrained_cache/yolo11s.pt'))
            def time_check(trainer):
                if time.time()>=fold_deadline:trainer.stop=True
            def evaluate_point(trainer):
                nonlocal best,best_pred
                epoch=trainer.epoch
                if epoch in seen:return
                if epoch!=0 and (epoch+1)%5 and epoch+1<160 and not trainer.stop:return
                seen.add(epoch)
                # AutoBackend fuses layers; evaluate a copy to protect EMA state.
                wrapper=YOLO('yolo11s.yaml')
                wrapper.model=copy.deepcopy(trainer.ema.ema).float().eval()
                results=wrapper.predict(source=[r['path'] for r in valid],imgsz=960,conf=.001,iou=.5,max_det=1,
                    device=0,batch=4,half=True,verbose=False,save=False,augment=False)
                pred=detector_points(results,valid,baseline)
                score=float(np.sqrt(np.mean([(np.array(pred[r['id']]['xy'])-r['xy'])**2 for r in valid])))
                history.append(dict(epoch=epoch+1,rmse=score,misses=sum(p['detector_miss'] for p in pred.values())))
                if score<best:
                    best,best_pred=score,pred
                    snapshot=dict(model=copy.deepcopy(trainer.ema.ema).half().cpu(),ema=None,
                        epoch=epoch,train_args=vars(trainer.args),date=datetime.datetime.now().isoformat(),version=ultralytics.__version__,
                        fovea_manifest_hash=manifest_hash(manifest),fovea_fold=fold,fovea_rmse=score)
                    temporary=best_path.with_suffix('.tmp');torch.save(snapshot,temporary);temporary.replace(best_path)
                    del snapshot
                write_json(directory/'point_history.json',history)
                print(f'POINT VALIDATION fold {fold} epoch {epoch+1}: RMSE={score:.4f}, best={best:.4f}',flush=True)
                del wrapper,results
                torch.cuda.empty_cache()
            model.add_callback('on_train_batch_end',time_check)
            model.add_callback('on_fit_epoch_end',evaluate_point)
            model.train(data=str(yamls[fold]),epochs=160,imgsz=960,batch=4,nbs=4,device=0,workers=2,
                optimizer='AdamW',lr0=learning_rate,lrf=.01,momentum=.9,weight_decay=.0005,cos_lr=True,warmup_epochs=3,warmup_bias_lr=bias_warmup,
                patience=0,seed=20261024+fold,deterministic=False,cache='ram',amp=True,
                mosaic=0.,mixup=0.,copy_paste=0.,degrees=10.,translate=.05,scale=.25,fliplr=.5,flipud=0.,
                hsv_h=.015,hsv_s=.3,hsv_v=.3,close_mosaic=0,val=False,save=False,plots=False,verbose=False,
                project=str(directory),name='train',exist_ok=False)
            if best_pred is None or not best_path.exists():raise RuntimeError('No detector checkpoint was validated.')
            detector_predictions.update(best_pred)
            write_json(directory/'detector_oof.json',dict(stage='detector',fold=fold,manifest_hash=manifest_hash(manifest),predictions=best_pred,rmse=best))
            del model
            gc.collect()
            torch.cuda.empty_cache()
            refined=refine(valid,best_pred,fold)
            refined_predictions.update(refined)
            write_json(directory/'refined_oof.json',dict(stage='refined',fold=fold,manifest_hash=manifest_hash(manifest),predictions=refined))
        components['detector']=detector_predictions
        components['refined']=refined_predictions
        v4_report=root/'reports/optimization_v4/selection.json'
        if v4_report.exists():
            chosen_v4=json.loads(v4_report.read_text(encoding='utf8'))['selected']
            data=json.loads((v4_report.parent/(chosen_v4+'_oof.json')).read_text(encoding='utf8'))
            if data['manifest_hash']!=manifest_hash(manifest):raise ValueError('v4 OOF manifest mismatch.')
            components['v4']={int(k):v for k,v in data['predictions'].items()}
        else:components['v4']=baseline
        if low_lr:
            previous_dir=root/'reports/detector_v5'
            previous=json.loads((previous_dir/'selection.json').read_text(encoding='utf8'))
            previous_components={'baseline':baseline,'v4':components['v4']}
            for component in ('detector','refined'):
                data=json.loads((previous_dir/(component+'_oof.json')).read_text(encoding='utf8'))
                if data['manifest_hash']!=manifest_hash(manifest):raise ValueError('Previous detector OOF manifest mismatch.')
                previous_components[component]={int(k):v for k,v in data['predictions'].items()}
            components['v4']=combine_predictions(previous_components,previous['weights'])
        results={name:dict(weights=weights,metrics=summarize(records,combine_predictions(components,weights))) for name,weights in recipes.items()}
        selected=min(results,key=lambda name:results[name]['metrics']['rmse'])
        chosen=results[selected]
        write_json(report_dir/'selection.json',dict(selected=selected,**chosen,candidates=results,
            warning='OOF model and checkpoint selection is optimistic. New leaderboard score unknown.'))
        status('predicting_test',selected=selected,selected_oof_rmse=chosen['metrics']['rmse'])
        test=[r for r in manifest['records'] if r['split']=='test']
        def from_csv(path):
            import csv
            pred={r['id']:dict(xy=[0.,0.]) for r in test}
            for row in csv.DictReader(path.open(encoding='utf-8-sig',newline='')):
                ident,axis=row['ImageID'].split('_Fovea_');pred[int(ident)]['xy'][0 if axis=='X' else 1]=float(row['value'])
            return pred
        test_components={'baseline':from_csv(root/'submissions/submission_accuracy_5h.csv')}
        if 'v4' in chosen['weights']:
            file=root/('submissions/submission_optimized_final.csv' if low_lr else 'submissions/submission_optimized_v4.csv')
            test_components['v4']=from_csv(file) if file.exists() else test_components['baseline']
        if 'detector' in chosen['weights'] or 'refined' in chosen['weights']:
            members=[];refined_members=[]
            for fold in range(5):
                detector=YOLO(str(job/f'fold_{fold}'/'point_best.pt'))
                result=detector.predict(source=[r['path'] for r in test],imgsz=960,conf=.001,iou=.5,max_det=1,
                    device=0,batch=4,half=True,verbose=False,save=False,augment=False)
                points=detector_points(result,test,test_components['baseline'])
                members.append(points)
                del detector,result;torch.cuda.empty_cache()
                if 'refined' in chosen['weights']:refined_members.append(refine(test,points,fold))
            test_components['detector']=combine_predictions({str(i):p for i,p in enumerate(members)},{str(i):.2 for i in range(5)})
            if refined_members:test_components['refined']=combine_predictions({str(i):p for i,p in enumerate(refined_members)},{str(i):.2 for i in range(5)})
        pred=combine_predictions(test_components,chosen['weights'])
        submission=root/('submissions/submission_optimized_low_lr.csv' if low_lr else 'submissions/submission_optimized_final.csv')
        make_submission(manifest['template'],pred,submission,{r['id']:(r['width'],r['height']) for r in test})
        write_json(report_dir/'detector_oof.json',dict(manifest_hash=manifest_hash(manifest),predictions=detector_predictions,metrics=summarize(records,detector_predictions)))
        write_json(report_dir/'refined_oof.json',dict(manifest_hash=manifest_hash(manifest),predictions=refined_predictions,metrics=summarize(records,refined_predictions)))
        write_json(submission.with_suffix('.diagnostics.json'),dict(predictions=pred,selected=selected,weights=chosen['weights'],oof_rmse=chosen['metrics']['rmse'],manifest_hash=manifest_hash(manifest)))
        desktop=Path('C:/Users/Lenovo/Desktop/2026task3_提交文件')/submission.name
        if desktop.exists():raise FileExistsError('Final desktop submission already exists; not overwriting.')
        shutil.copyfile(submission,desktop)
        assert hashlib.sha256(submission.read_bytes()).digest()==hashlib.sha256(desktop.read_bytes()).digest()
        status('completed',submission=str(submission),desktop_submission=str(desktop),selected_oof_rmse=chosen['metrics']['rmse'])
    except BaseException as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        lock.close();(job/'active.lock').unlink(missing_ok=True)
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
