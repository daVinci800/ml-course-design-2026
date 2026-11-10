"""Reproducible report experiment; original historical scripts remain separate.

Run with --data-root, --cache-root, --weights, --output and --execute.
No test labels are read. Exact pixel duplicates are grouped; conflicting groups excluded.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse, copy, csv, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

CLASSES=['Angry','Fear','Happy','Sad','Surprise','Neutral']

def save_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

class CBAM(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.fc=nn.Sequential(nn.Conv2d(c,c//16,1,bias=False),nn.ReLU(),nn.Conv2d(c//16,c,1,bias=False))
        self.spatial=nn.Conv2d(2,1,7,padding=3,bias=False)
    def forward(self,x):
        x=x*torch.sigmoid(self.fc(x.mean((2,3),keepdim=True))+self.fc(x.amax((2,3),keepdim=True)))
        return x*torch.sigmoid(self.spatial(torch.cat([x.mean(1,keepdim=True),x.amax(1,keepdim=True)],1)))

class Model(nn.Module):
    def __init__(self,weights=None):
        super().__init__()
        kwargs={'pretrained_cfg_overlay':{'file':str(weights)}} if weights else {}
        self.backbone=timm.create_model('convnext_tiny.in12k_ft_in1k',pretrained=weights is not None,num_classes=0,**kwargs)
        self.cbam=CBAM(768)
        self.head=nn.Sequential(nn.Dropout(.3),nn.Linear(768,6))
    def forward(self,x):
        f=self.cbam(self.backbone.forward_features(x)).mean((2,3))
        return self.head(f),f

class Data(Dataset):
    def __init__(self,rows,cache_root,train):
        self.rows=rows; self.cache_root=Path(cache_root)
        common=[transforms.Resize((224,224))]
        if train: common += [transforms.RandomHorizontalFlip(),transforms.RandAugment(2,9)]
        common += [transforms.ToTensor()]
        if train: common += [transforms.RandomErasing(.25,scale=(.02,.1),ratio=(.3,3.3))]
        common += [transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
        self.transform=transforms.Compose(common)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(self.cache_root/str(r['label'])/r['filename']) as im:
            x=self.transform(im.convert('RGB'))
        return x,r['label'],i

def worker_seed(worker):
    seed=torch.initial_seed()%2**32
    random.seed(seed);np.random.seed(seed)
    torch.set_num_threads(1)

def metrics(y,p):
    cm=np.zeros((6,6),dtype=int)
    for a,b in zip(y,p.argmax(1)):cm[a,b]+=1
    per=[]
    for k,name in enumerate(CLASSES):
        tp=int(cm[k,k]);support=int(cm[k].sum());pred=int(cm[:,k].sum())
        pr=tp/max(pred,1);re=tp/max(support,1)
        per.append(dict(class_name=name,support=support,precision=pr,recall=re,f1=2*pr*re/max(pr+re,1e-12)))
    return dict(samples=len(y),correct=int(np.trace(cm)),accuracy=float(np.trace(cm)/len(y)),macro_f1=float(np.mean([r['f1'] for r in per])),per_class=per,confusion_matrix=cm.tolist())

@torch.no_grad()
def evaluate(model,loader,device,tta=False):
    model.eval();ys=[];ps=[];loss=0
    for x,y,_ in loader:
        x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z,_=model(x); prob=z.float().softmax(1)
            if tta:prob=(prob+model(x.flip(-1))[0].float().softmax(1))/2
            loss+=float(nn.functional.cross_entropy(z.float(),y,reduction='sum'))
        ys.extend(y.cpu().tolist());ps.append(prob.cpu().numpy())
    p=np.concatenate(ps);m=metrics(np.asarray(ys),p);m['cross_entropy']=loss/len(ys)
    return m,p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',required=True);ap.add_argument('--cache-root',required=True)
    ap.add_argument('--weights',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--execute',action='store_true')
    ap.add_argument('--resume',action='store_true',help='Restore the complete last epoch checkpoint')
    args=ap.parse_args()
    if not args.execute:raise SystemExit('Training requires --execute')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'history.jsonl').exists() and not args.resume:raise RuntimeError('Refusing to overwrite an existing experiment')
    resume_ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False) if args.resume else None
    seed=20261110;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=True
    groups={};all_rows=[]
    for label,name in enumerate(CLASSES):
        for path in sorted((Path(args.data_root)/'train'/name).glob('*.jpg')):
            with Image.open(path) as im: pixels=np.asarray(im.convert('L'))
            sha=hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
            r=dict(filename=path.name,class_name=name,label=label,sha256=sha)
            all_rows.append(r);groups.setdefault(sha,[]).append(r)
    excluded=[];valid_groups=[]
    for key,rows in groups.items():
        if len({r['label'] for r in rows})>1:
            for r in rows:r['split']='excluded_conflicting_label';excluded.append(r)
        else:valid_groups.append(rows)
    rng=np.random.default_rng(seed)
    for label in range(6):
        gs=[g for g in valid_groups if g[0]['label']==label];rng.shuffle(gs)
        target=round(sum(len(g) for g in gs)*.2);n=0
        for g in gs:
            split='validation' if n<target else 'train'
            if split=='validation':n+=len(g)
            for r in g:r['split']=split
    if args.resume:
        prior_manifest=list(csv.DictReader((out/'manifest.csv').open(encoding='utf-8')))
        if [{k:str(v) for k,v in r.items()} for r in all_rows]!=prior_manifest:raise RuntimeError('Dataset or split changed')
    else:
        with (out/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    train=[r for r in all_rows if r['split']=='train'];val=[r for r in all_rows if r['split']=='validation']
    assert not ({r['sha256'] for r in train}&{r['sha256'] for r in val})
    audit=dict(total=len(all_rows),train=len(train),validation=len(val),excluded_conflicting_labels=len(excluded),unique_pixel_groups=len(groups),duplicate_extra_samples=len(all_rows)-len(groups),class_counts={n:sum(r['class_name']==n for r in all_rows) for n in CLASSES},train_counts={n:sum(r['class_name']==n for r in train) for n in CLASSES},validation_counts={n:sum(r['class_name']==n for r in val) for n in CLASSES},pixel_hash_overlap=0)
    save_json(out/'data_audit.json',audit);print(json.dumps(audit),flush=True)
    config=vars(args)|dict(seed=seed,model='convnext_tiny.in12k_ft_in1k',attention='CBAM',optimizer='AdamW',learning_rate=.0001,weight_decay=.01,warmup_epochs=3,mixup_alpha=.2,center_loss_weight=.01,ema_decay=.999,focal_gamma=2.,focal_definition='class_weight_times_unweighted_focal',precision='bf16',selection='best validation macro F1; accuracy breaks ties',manifest_sha256=hashlib.sha256((out/'manifest.csv').read_bytes()).hexdigest(),pretrained_sha256=hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),device=torch.cuda.get_device_name(0),torch=torch.__version__,timm=timm.__version__,validation_tta=False)
    config.pop('resume',None)
    if args.resume:
        if config!=resume_ck['config']:raise RuntimeError('Resume configuration mismatch')
    else:save_json(out/'config.json',config)
    generator=torch.Generator().manual_seed(seed)
    dl=DataLoader(Data(train,args.cache_root,True),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0,worker_init_fn=worker_seed,generator=generator)
    vl=DataLoader(Data(val,args.cache_root,False),batch_size=args.batch_size*2,shuffle=False,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0,worker_init_fn=worker_seed)
    device='cuda';model=Model(None if args.resume else args.weights).to(device);ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    centers=nn.Parameter(torch.randn(6,768,device=device))
    opt=torch.optim.AdamW(list(model.parameters())+[centers],lr=.0001,weight_decay=.01)
    counts=np.bincount([r['label'] for r in train],minlength=6)
    cw=torch.tensor(len(train)/(6*counts),dtype=torch.float32,device=device)
    def focal(z,y):
        ce=nn.functional.cross_entropy(z.float(),y,reduction='none')
        return (cw[y]*(1-torch.exp(-ce)).pow(2)*ce).mean()
    best=(-1,-1);start_epoch=0;start=time.perf_counter()
    if args.resume:
        model.load_state_dict(resume_ck['raw_model']);ema.load_state_dict(resume_ck['ema_model'])
        with torch.no_grad():centers.copy_(resume_ck['centers'].to(device))
        opt.load_state_dict(resume_ck['optimizer']);start_epoch=resume_ck['epoch']
        old=json.loads((out/'metrics.json').read_text(encoding='utf-8'));best=(old['macro_f1'],old['accuracy'])
        history=[json.loads(s) for s in (out/'history.jsonl').read_text().splitlines() if s.strip()]
        if [r['epoch'] for r in history]!=list(range(1,start_epoch+1)):raise RuntimeError('History and checkpoint epochs disagree')
        torch.set_rng_state(resume_ck['torch_rng']);torch.cuda.set_rng_state_all(resume_ck['cuda_rng'])
        np.random.set_state(resume_ck['numpy_rng']);random.setstate(resume_ck['python_rng']);generator.set_state(resume_ck['loader_rng'])
        save_json(out/'resume_record.json',dict(resumed_after_epoch=start_epoch,checkpoint_sha256=hashlib.sha256((out/'last.pt').read_bytes()).hexdigest(),note='Restored raw model, EMA, optimizer, centers and main RNG states. Worker augmentation RNG is restarted; not bitwise identical to an uninterrupted run. Partial next epoch discarded.'))
        print('Restored checkpoint after epoch '+str(start_epoch),flush=True)
        del resume_ck
    for epoch in range(start_epoch,args.epochs):
        if epoch<3:factor=(epoch+1)/3
        else:factor=.5*(1+math.cos(math.pi*(epoch-3)/max(1,args.epochs-3)))
        for pg in opt.param_groups:pg['lr']=.0001*factor
        model.train();total=0;loss_sum=0;t0=time.perf_counter()
        for step,(x,y,_) in enumerate(dl):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            lam=float(np.random.beta(.2,.2));lam=max(lam,1-lam);order=torch.randperm(x.shape[0],device=device)
            xa=lam*x+(1-lam)*x[order]
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):z,f=model(xa)
            f=f.float();loss=lam*focal(z,y)+(1-lam)*focal(z,y[order])+.01*(lam*(f-centers[y]).pow(2).sum(1).mean()+(1-lam)*(f-centers[y[order]]).pow(2).sum(1).mean())
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(list(model.parameters())+[centers],5);opt.step()
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.001)
            loss_sum+=float(loss.detach())*len(y);total+=len(y)
            if (step+1)%150==0:print(f'epoch {epoch+1} step {step+1}/{len(dl)} seconds {time.perf_counter()-t0:.1f}',flush=True)
        m,prob=evaluate(ema,vl,device)
        row=dict(epoch=epoch+1,train_loss=loss_sum/total,validation_accuracy=m['accuracy'],validation_macro_f1=m['macro_f1'],validation_cross_entropy=m['cross_entropy'],learning_rate=opt.param_groups[0]['lr'],seconds=time.perf_counter()-t0)
        with (out/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        ck=dict(raw_model=model.state_dict(),ema_model=ema.state_dict(),centers=centers.detach(),optimizer=opt.state_dict(),epoch=epoch+1,config=config,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),loader_rng=generator.get_state())
        torch.save(ck,out/'last.pt')
        rank=(m['macro_f1'],m['accuracy'])
        if rank>best:
            best=rank;torch.save(dict(model=ema.state_dict(),epoch=epoch+1,config=config),out/'best.pt')
            save_json(out/'metrics.json',m|dict(best_epoch=epoch+1,elapsed_seconds=time.perf_counter()-start))
            np.save(out/'validation_probabilities.npy',prob)
            with (out/'validation_predictions.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f);w.writerow(['filename','truth','prediction']+['p_'+n for n in CLASSES])
                for r,p in zip(val,prob):w.writerow([r['filename'],r['label'],int(p.argmax())]+p.tolist())
    history=[json.loads(s) for s in (out/'history.jsonl').read_text().splitlines() if s.strip()]
    summary=dict(status='completed',epochs=args.epochs,elapsed_seconds_last_session=time.perf_counter()-start,completed_epoch_seconds=sum(r['seconds'] for r in history),best_validation_macro_f1=best[0],best_validation_accuracy=best[1],peak_allocated_memory_mb=torch.cuda.max_memory_allocated()/2**20,resumed_after_epoch=start_epoch,partial_epoch_time_excluded=True)
    save_json(out/'training_summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
