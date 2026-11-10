from pathlib import Path
import json,csv,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
BASE=Path(__file__).resolve().parents[1];PKG=BASE/'outputs/ml_course_design_code';OUT=BASE/'work/figures';OUT.mkdir(parents=True,exist_ok=True)
FONTPATH=r'C:\Windows\Fonts\msyh.ttc'
def font(n):return ImageFont.truetype(FONTPATH,n)
BLUE='#24527A';ORANGE='#BB5C29';GREEN='#277360';GRAY='#64748B'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def text(d,xy,s,n=25,fill='#202B35',anchor=None):d.text(xy,str(s),font=font(n),fill=fill,anchor=anchor)
def chart(name,series,ymax=1.,ylabel='准确率',xlabel='训练轮次',size=(1700,660)):
    im=Image.new('RGB',size,'white');d=ImageDraw.Draw(im);l,t,r,b=110,125,size[0]-65,size[1]-90
    xmax=max(len(s[1]) for s in series)
    for j in range(6):
        v=ymax*j/5;yy=b-(b-t)*j/5;d.line((l,yy,r,yy),fill='#DCE2E8',width=2);text(d,(l-18,yy),f'{v:.1f}',25,anchor='rm')
    d.line((l,t,l,b,r,b),fill='#8795A3',width=2)
    for x in range(0,xmax+1,max(1,xmax//6)):
        xx=l+(r-l)*x/max(xmax,1);text(d,(xx,b+20),str(x),25,anchor='mt')
    text(d,(l,12),ylabel,26);text(d,(r,b+55),xlabel,25,anchor='rt')
    xx=l
    for title,vals,color in series:
        d.line((xx,t-45,xx+44,t-45),fill=color,width=5);text(d,(xx+57,t-45),title,25,anchor='lm');xx+=max(300,len(title)*26+95)
        pts=[(l+(r-l)*(i+1)/xmax,b-(b-t)*v/ymax) for i,v in enumerate(vals)]
        if len(pts)>1:d.line(pts,fill=color,width=5)
        for x,y in pts:
            if len(pts)<20:d.ellipse((x-5,y-5,x+5,y+5),fill=color)
    im.save(OUT/name)
def confusion(name,cm,labels):
    cm=np.asarray(cm);n=len(cm);cell=103;l=245;t=80;w=l+n*cell+80;h=t+n*cell+135
    im=Image.new('RGB',(w,h),'white');d=ImageDraw.Draw(im)
    for i in range(n):
        text(d,(l-20,t+i*cell+cell/2),labels[i],24,anchor='rm')
        text(d,(l+i*cell+cell/2,t+n*cell+28),labels[i],23,anchor='mt')
        for j in range(n):
            val=cm[i,j];v=val/max(1,cm[i].sum());col=tuple(int(246*(1-v)+c*v) for c in (35,80,118))
            box=(l+j*cell,t+i*cell,l+(j+1)*cell-3,t+(i+1)*cell-3)
            d.rectangle(box,fill=col);text(d,(box[0]+cell/2,box[1]+cell/2),str(val),30,'white' if v>.5 else '#1E293B',anchor='mm')
    text(d,(l,t-45),'真实类别 ↓    预测类别 →',26)
    im.save(OUT/name)
def bars(name,categories,series,ymax=40,ylabel='坐标 RMSE 像素'):
    im=Image.new('RGB',(1700,630),'white');d=ImageDraw.Draw(im);l,t,r,b=110,100,1620,530
    for j in range(5):
        yy=b-(b-t)*j/4;d.line((l,yy,r,yy),fill='#DFE5EA',width=2);text(d,(l-20,yy),f'{ymax*j/4:.0f}',25,anchor='rm')
    text(d,(l,t-65),ylabel,26)
    for k,(label,values,color) in enumerate(series):
        x=l+700+k*340;d.rectangle((x,t-55,x+35,t-26),fill=color);text(d,(x+48,t-40),label,25,anchor='lm')
        for i,v in enumerate(values):
            cx=l+(r-l)*(i+.5)/len(categories)+(k-(len(series)-1)/2)*87
            yy=b-(b-t)*v/ymax;d.rectangle((cx-35,yy,cx+35,b),fill=color);text(d,(cx,yy-12),f'{v:.2f}',23,anchor='mb')
    for i,s in enumerate(categories):text(d,(l+(r-l)*(i+.5)/len(categories),b+30),s,25,anchor='mt')
    im.save(OUT/name)
def pipeline(name,blocks):
    w=1700;h=320;n=len(blocks);im=Image.new('RGB',(w,h),'white');d=ImageDraw.Draw(im)
    bw=280 if n<=5 else 225;gap=(w-80-n*bw)/max(1,n-1)
    for i,lines in enumerate(blocks):
        x=40+i*(bw+gap);d.rounded_rectangle((x,52,x+bw,265),radius=10,outline='#64748B',width=3,fill='#F4F7FA')
        for j,s in enumerate(lines):text(d,(x+bw/2,90+j*48),s,25,anchor='mm')
        if i<n-1:
            x1=x+bw+7;x2=x+bw+gap-7;y=157;d.line((x1,y,x2,y),fill=BLUE,width=4);d.polygon([(x2,y),(x2-12,y-9),(x2-12,y+9)],fill=BLUE)
    im.save(OUT/name)
def cropfit(im,size):
    im=im.convert('RGB');im.thumbnail(size);canvas=Image.new('RGB',size,'#F6F6F6');canvas.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2));return canvas
def fovea_examples():
    root=Path(r'C:\Users\Lenovo\Documents\Codex\2026-10-20\1-0-0-2-3-rules\outputs\fovea_solution')
    rec=[r for r in load(root/'prepared/manifest.json')['records'] if r['split']=='train'];pred=load(root/'reports/detector_v6/detector_oof.json')['predictions']
    ranked=sorted([(float(np.linalg.norm(np.asarray(pred[str(r['id'])]['xy'])-r['xy'])),r) for r in rec],key=lambda x:x[0])
    picks=[ranked[0],ranked[len(ranked)//2],ranked[-1]]
    im=Image.new('RGB',(1680,580),'white');d=ImageDraw.Draw(im)
    for i,(err,r) in enumerate(picks):
        original=Image.open(r['path']).convert('RGB');ratio=min(520/original.width,440/original.height);small=original.resize((round(original.width*ratio),round(original.height*ratio)))
        off=(i*560+(560-small.width)//2,65+(440-small.height)//2);im.paste(small,off)
        tx,ty=r['xy'];px,py=pred[str(r['id'])]['xy'];x=off[0]+tx*ratio;y=off[1]+ty*ratio
        d.line((x-13,y,x+13,y),fill='#00FF88',width=4);d.line((x,y-13,x,y+13),fill='#00FF88',width=4)
        x=off[0]+px*ratio;y=off[1]+py*ratio;d.ellipse((x-11,y-11,x+11,y+11),outline='#FFBB22',width=4)
        text(d,(i*560+280,35),f"ID {r['id']}  欧氏误差 {err:.2f} px",25,anchor='mm')
    text(d,(840,548),'绿色十字为真实标注    黄色圆圈为折外预测',26,anchor='mm');im.save(OUT/'fovea_examples.png')
    (OUT/'fovea_examples.json').write_text(json.dumps([dict(id=r['id'],euclidean_error=e) for e,r in picks]),encoding='utf-8')
def vessel_examples():
    root=Path(r'C:\Users\Lenovo\Documents\Codex\2026-10-31\2d-0-255-rules-2026-11\outputs\retina_vessel')
    with Path(r'C:\Users\Lenovo\Desktop\2026task4_提交结果\折外验证逐图结果.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    rows=sorted(rows,key=lambda r:float(r['native_vessel_dice_at_selected_threshold']));picks=[rows[len(rows)//2],rows[0]]
    im=Image.new('RGB',(1700,1220),'white');d=ImageDraw.Draw(im)
    for j,r in enumerate(picks):
        sid=r['Id'];image=Image.open(root/f'data/segmentation/train/image/{sid}.jpg').convert('RGB');label=Image.open(root/f'data/segmentation/train/label/{sid}.jpg').convert('L')
        paths=list((root/'runs/main').glob(f'fold*/oof_native/{sid}.npy'))
        # Calibration uses newly computed 8-direction TTA probabilities, not cached no-TTA training outputs.
        tta_paths=list((root/'artifacts').rglob(f'{sid}.npy'))
        split=load(root/'artifacts/splits.json')['folds'];fold=next(x['fold'] for x in split if sid in x['val_ids'])
        # Display saved no-TTA OOF probability at the model-selection threshold 0.5.
        prob=np.load(root/f'runs/main/fold{fold}/oof_native/{sid}.npy').squeeze()
        pred=Image.fromarray((prob>=.5).astype('uint8')*255)
        for k,(src,title) in enumerate([(image,'原始眼底图'),(label,'真实血管标注'),(pred,'折外预测 阈值 0.5')]):
            im.paste(cropfit(src,(530,470)),(15+k*565,65+j*610));text(d,(280+k*565,35+j*610),title,27,anchor='mm')
        truth=np.asarray(label)>=128;mask=prob>=.5;dice=2*(mask&truth).sum()/max(1,mask.sum()+truth.sum())
        text(d,(850,570+j*610),f'ID {sid}    此图展示无 TTA 的折外预测    Dice {dice:.4f}',25,anchor='mm')
    im.save(OUT/'vessel_examples.png')
def main():
    plant=PKG/'task01_plant/evidence'
    vals=[]
    for run,title,col in [('round2_fast_resnet50','ResNet50 验证均值',BLUE),('round3_effnet_b3','EfficientNet B3 验证均值',ORANGE)]:
        histories=[load(plant/run/f'fold_{i}/metrics.json') for i in range(5)]
        vals.append((title,np.mean([[r['val']['acc'] for r in h] for h in histories],axis=0).tolist(),col))
    chart('plant_curve.png',vals)
    m=load(plant/'round3_effnet_resnet_blend/evaluation_50_50/metrics.json')
    rows=list(csv.reader((plant/'round3_effnet_resnet_blend/evaluation_50_50/confusion_matrix.csv').open(encoding='utf-8-sig')))
    confusion('plant_confusion.png',[[int(x) for x in r[1:6]] for r in rows[1:]],['B grass','Wheat','S bent','Mayweed','S beet'])
    fr=BASE/'work/fer_retrain'
    if (fr/'history.jsonl').exists():
        h=[json.loads(s) for s in (fr/'history.jsonl').read_text().splitlines() if s.strip()]
        chart('fer_curve.png',[('验证准确率',[r['validation_accuracy'] for r in h],BLUE),('验证 Macro F1',[r['validation_macro_f1'] for r in h],ORANGE)],1.)
        fm=load(fr/'metrics.json');confusion('fer_confusion.png',fm['confusion_matrix'],['Angry','Fear','Happy','Sad','Surprise','Neutral'])
    fm=load(PKG/'task03_fovea/evidence/detector_v6/selection.json')['metrics']
    bars('fovea_folds.png',[f'第 {i+1} 折' for i in range(5)],[('热图基线',[6.315104,23.473980,32.936216,21.349150,8.546561],ORANGE),('YOLO11s',[fm['per_fold_rmse'][str(i)] for i in range(5)],BLUE)],40)
    v=PKG/'task04_vessel/evidence';series=[]
    colors=[BLUE,ORANGE,GREEN,'#8B5E97','#6E7E36']
    for i in range(5):
        h=[json.loads(s) for s in (v/f'fold{i}/history.jsonl').read_text().splitlines() if s.strip()]
        # Resume history may contain restarted epochs. Keep the latest record per epoch.
        hs={r['epoch']:r for r in h};series.append((f'第 {i+1} 折',[hs[k]['validation_vessel_dice'] for k in sorted(hs)],colors[i]))
    chart('vessel_curve.png',series,1.,'原尺寸验证 Dice',size=(1700,660))
    pipeline('plant_model.png',[['RGB 输入','补边和缩放'],['并行两条分支','ResNet50 与 B3'],['各分支五类概率','四方向增强'],['同折概率等权平均','五折模型集成'],['类别概率最大值','输出植物类别']])
    pipeline('fer_model.png',[['CLAHE 灰度','RGB 224×224'],['ConvNeXt Tiny','768×7×7'],['CBAM','通道与空间注意力'],['全局平均池化','768 维特征'],['Dropout 0.3','六类线性分类']])
    pipeline('fovea_model.png',[['原始眼底图','缩放输入 960'],['YOLO11s','单类黄斑检测'],['最高置信度框','计算框中心'],['未检出自动回退','对应折热图模型'],['还原原图坐标','五折坐标平均']])
    pipeline('vessel_model.png',[['RGB 与 CLAHE','四通道输入'],['残差编码器','24 48 96 192 384'],['跳跃连接与解码','空间注意力'],['主头与辅助头','像素级概率'],['五折与八方向','均值后阈值化']])
    fovea_examples();vessel_examples();print('Figures written',OUT)
if __name__=='__main__':main()
