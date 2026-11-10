# -*- coding: utf-8 -*-
"""Build the four-chapter report from verified experiment evidence."""
from pathlib import Path
import json,csv,math
import numpy as np
from docx import Document
from docx.shared import Inches,Pt,Cm,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

BASE=Path(__file__).resolve().parents[1];OUT=BASE/'outputs';FIG=BASE/'work/figures';PKG=OUT/'ml_course_design_code'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fmt(x,n=4):return f'{x:.{n}f}'
def pct(x):return f'{100*x:.2f}%'
fer=load(BASE/'work/fer_retrain/metrics.json');audit=load(BASE/'work/fer_retrain/data_audit.json')
fh=[json.loads(s) for s in (BASE/'work/fer_retrain/history.jsonl').read_text().splitlines() if s.strip()]
plant=load(PKG/'task01_plant/evidence/round3_effnet_resnet_blend/evaluation_50_50/metrics.json')
fm=load(PKG/'task03_fovea/evidence/detector_v6/selection.json')['metrics']
vm=load(PKG/'task04_vessel/evidence/训练与提交核验.json')
repo=load(BASE/'work/github_result.json')['url'] if (BASE/'work/github_result.json').exists() else '待完成 GitHub 上传后补充仓库链接'
doc=Document();sec=doc.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=Cm(2.1);sec.bottom_margin=Cm(1.9);sec.left_margin=Cm(2.4);sec.right_margin=Cm(2.4)
sec.header_distance=Cm(.8);sec.footer_distance=Cm(.9)
for name,size,face,bold in [('Normal',11.5,'宋体',False),('Title',25,'黑体',True),('Subtitle',15,'黑体',False),('Heading 1',18,'黑体',True),('Heading 2',13,'黑体',True),('Heading 3',12,'黑体',True),('Caption',10,'宋体',False)]:
    st=doc.styles[name];st.font.name='Times New Roman';st.font.size=Pt(size);st.font.bold=bold;st.font.color.rgb=RGBColor(0,0,0)
    st.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),face)
    st.paragraph_format.space_after=Pt(6 if name=='Normal' else 8)
    st.paragraph_format.line_spacing=1.28 if name=='Normal' else 1.12
    if name.startswith('Heading'):st.paragraph_format.space_before=Pt(10)
doc.styles['Normal'].paragraph_format.first_line_indent=Pt(23)
doc.core_properties.title='机器学习课程设计报告'
doc.core_properties.subject='植物分类 人脸情感分类 中央凹定位 眼底血管分割'
doc.core_properties.author='姜广阔'
doc.core_properties.keywords='20245621 机器学习 课程设计'
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');footer._p.append(f)
sec.different_first_page_header_footer=True
def p(s,bold=False,indent=True):
    para=doc.add_paragraph();para.paragraph_format.widow_control=True
    if not indent:para.paragraph_format.first_line_indent=Pt(0)
    run=para.add_run(s);run.bold=bold;return para
def h(s,level=2):return doc.add_heading(s,level)

def repo_link(prefix='',center=False):
    q=p(prefix,indent=False)
    if center:q.alignment=WD_ALIGN_PARAGRAPH.CENTER
    if repo.startswith('https://'):
        link=OxmlElement('w:hyperlink');link.set(qn('r:id'),q.part.relate_to(repo,RT.HYPERLINK,is_external=True))
        run=OxmlElement('w:r');pr=OxmlElement('w:rPr');color=OxmlElement('w:color');color.set(qn('w:val'),'000000');pr.append(color);run.append(pr)
        tx=OxmlElement('w:t');tx.text=repo;run.append(tx);link.append(run);q._p.append(link)
    else:q.add_run(repo)
    return q
def page(force=False):
    if force:doc.add_page_break()
def caption(s):
    q=doc.add_paragraph(s,'Caption');q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.first_line_indent=Pt(0)
    q.paragraph_format.keep_with_next=s.startswith('表')
    return q
def image(name,cap,width=6.45):
    q=doc.add_paragraph();q.paragraph_format.first_line_indent=Pt(0);q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.keep_with_next=True;q.paragraph_format.space_after=Pt(2)
    r=q.add_run();r.add_picture(str(FIG/name),width=Inches(width))
    # Meaningful alt text in OOXML.
    for el in r._r.iter(qn('wp:docPr')):el.set('descr',cap)
    caption(cap)
def table(headers,rows,widths=None):
    t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    if widths:
        for c,w in zip(t.columns,widths):c.width=Cm(w)
    for i,head in enumerate(headers):t.rows[0].cells[i].text=str(head)
    for row in rows:
        cells=t.add_row().cells
        for c,v in zip(cells,row):c.text=str(v)
    for ri,row in enumerate(t.rows):
        pr=row._tr.get_or_add_trPr();ns=OxmlElement('w:cantSplit');pr.append(ns)
        if ri==0:rep=OxmlElement('w:tblHeader');pr.append(rep)
        for ci,c in enumerate(row.cells):
            c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr=c._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
            for edge in ['top','left','bottom','right']:
                b=OxmlElement('w:'+edge);b.set(qn('w:val'),'single');b.set(qn('w:sz'),'5');b.set(qn('w:color'),'D9D9D9');borders.append(b)
            tcpr.append(borders);m=OxmlElement('w:tcMar')
            for edge,num in [('top','70'),('bottom','70'),('left','95'),('right','95')]:
                e=OxmlElement('w:'+edge);e.set(qn('w:w'),num);e.set(qn('w:type'),'dxa');m.append(e)
            tcpr.append(m)
            if ri==0:
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E7EEF5');tcpr.append(shade)
            for para in c.paragraphs:
                para.paragraph_format.first_line_indent=Pt(0);para.paragraph_format.line_spacing=1.16;para.paragraph_format.space_after=Pt(2)
                para.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER
                if len(t.rows)<=9:para.paragraph_format.keep_with_next=ri<len(t.rows)-1
                for r in para.runs:r.font.size=Pt(10.5);r.bold=ri==0
    spacer=doc.add_paragraph();spacer.paragraph_format.space_after=Pt(0);spacer.paragraph_format.line_spacing=Pt(2);spacer.paragraph_format.space_before=Pt(0);spacer.add_run().font.size=Pt(1)
    return t
def eq(s):
    mapping={'L = λ Lfocal(z, ya) + (1 − λ) Lfocal(z, yb) + 0.01 Lcenter':'fer_loss',
             'x = (xleft + xright) / 2      y = (ytop + ybottom) / 2':'box_center',
             'RMSE = √[ Σ ((x̂ − x)² + (ŷ − y)²) / (2N) ]':'rmse',
             'L = 0.4 LBCE + 0.5 LDice + 0.1 r LclDice + Laux':'vessel_loss',
             'Dice = 2 |P ∩ G| / (|P| + |G|)':'dice'}
    file=FIG/(mapping[s]+'.png')
    q=doc.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.first_line_indent=Pt(0)
    if file.exists():
        from PIL import Image
        im=Image.open(file);width=min(6.0,im.width/600);q.add_run().add_picture(str(file),width=Inches(width))
    else:q.add_run(s).font.name='Cambria Math'

# Cover.
q=doc.add_paragraph();q.paragraph_format.space_after=Pt(62)
q=doc.add_paragraph('机器学习课程设计报告','Title');q.alignment=WD_ALIGN_PARAGRAPH.CENTER
q=doc.add_paragraph('图像分类 定位与医学影像分割','Subtitle');q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.space_after=Pt(36)
for s in ['第一章  植物分类','第二章  人脸情感分类','第三章  中央凹检测定位','第四章  眼底血管分割']:
    q=p(s,indent=False);q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.space_after=Pt(14)
q=doc.add_paragraph();q.paragraph_format.space_after=Pt(22)
for s in ['姓    名   姜广阔','学    号   20245621','课    程   机器学习课程设计','日    期   2026 年 11 月 10 日']:
    q=p(s,indent=False);q.alignment=WD_ALIGN_PARAGRAPH.CENTER
q=doc.add_paragraph();q.paragraph_format.space_after=Pt(15)
q=p('代码仓库',True,False);q.alignment=WD_ALIGN_PARAGRAPH.CENTER
repo_link(center=True)

page(True);h('摘要',2)
p('本课程设计围绕植物类别识别、人脸情感分类、眼底中央凹定位和眼底血管分割开展四项实验。针对任务的输出形式和数据规模，分别采用迁移学习分类器与概率融合、ConvNeXt-Tiny 与注意力模块、黄斑框检测与坐标解码、带空间注意力的残差 U-Net。各章依次说明数据、模型、实验设置、训练过程和结果分析，并保留可复查的配置与逐样本预测。')
p(f'植物分类的 500 张训练图像采用分层五折验证，ResNet50 与 EfficientNet-B3 等权融合的折外准确率为 92.00%，Macro-F1 为 0.9202。人脸情感分类重新执行固定分组留出实验，在 {audit["validation"]} 张验证图像上，最佳模型准确率为 {pct(fer["accuracy"])}、Macro-F1 为 {fmt(fer["macro_f1"])}。中央凹定位的五折坐标 RMSE 为 16.579950 像素。血管分割的原尺寸折外 Dice 为 0.817872，按官方缩放规则构建的代理 Dice 为 0.842116。')
p('结果表明，数据划分、预处理和评价口径会直接影响结论：情感数据中的完全重复图像应整组划分，中央凹定位需要检查少数大误差样本，血管分割需要核对标签极性与缩放规则。上述验证结果均参与了模型或阈值选择，不能直接解释为独立测试集性能。本文将已完成的实验、历史反馈和后续改进建议明确区分。')
p('关键词  迁移学习  注意力机制  折外验证  中央凹定位  血管分割',indent=False)
h('目录',2)
for s in ['第一章  植物分类','第二章  人脸情感分类','第三章  中央凹检测定位','第四章  眼底血管分割','参考文献']:
    p(s,indent=False)
p('正文共四章，每项任务单独成章。源代码按 task01_plant 至 task04_vessel 组织，每章末列出对应的复查文件。',indent=False)

# Chapter 1 page 1.
page(True);h('第一章 植物分类',1);h('1.1 任务定义与数据分析')
p('本任务输入一张植物幼苗图像，输出五个物种之一。图像包含不同生长状态、植株尺度及砂砾背景，主要难点是小样本条件下的类内变化与禾本科植物之间的形态相似。训练集仅提供物种标签，未提供生长阶段标签，因此本实验只建立物种分类模型。')
caption('表 1.1 数据集规模与类别')
table(['类别','训练图像数','主要形态'],[['Black-grass',100,'细长条带叶'],['Common wheat',100,'较宽条带叶'],['Loose Silky-bent',100,'细软簇生叶'],['Scentless Mayweed',100,'分裂或羽状真叶'],['Sugar beet',100,'宽阔椭圆叶']],[5.1,3.0,8.0])
p('训练图像共 500 张，测试图像共 378 张，每类训练样本均为 100 张，不需要类别加权或过采样。878 张图像均能解码，其中 869 张为 RGB、9 张为 RGBA；RGBA 的透明通道恒为 255，统一转换为 RGB。原始尺寸共有 518 种，宽度为 49～2840 像素，且 852 张为正方形。')
p('文件哈希和感知哈希检查未发现明显的完全重复或近重复图像，但该检查不能替代植株身份分组。不同类别的原始尺寸、绿色植被占比存在差异，背景中还包含标尺和花盆边缘，模型可能学习到与类别相关的采集特征。因此不输入文件名、宽高等元信息，并通过几何与颜色增强降低背景依赖。')
h('1.2 模型结构')
p('最终方案采用两个并行的 ImageNet 预训练分类分支：ResNet50 使用残差瓶颈结构提取特征，末端全局池化后得到 2048 维向量；EfficientNet-B3 以 MBConv 和通道注意力提取特征，末端为 1536 维向量。各自将分类层替换为五类线性层，输出 softmax 概率。两者独立训练，在相同验证样本上按 0.5∶0.5 融合概率。[1][2]')
image('plant_model.png','图 1.1 植物分类的双分支推理流程',6.45)

# Chapter1 page2.
page();h('1.3 实验方案与训练参数')
p('采用固定种子 2026 的分层五折划分，每折 400 张训练、100 张验证，验证集中每类 20 张。每个样本只使用未训练过该样本的对应折模型生成折外预测，即 OOF。融合时只组合两个分支在同一验证折上的概率，避免把其他折训练过该样本的模型引入评估。')
p('训练前按长宽比补成正方形，再缩放至 288×288，使用 ImageNet 均值和标准差归一化。增强包含 D4 旋转翻转、0.9～1.1 尺度仿射、5% 平移和轻度颜色扰动。主实验使用普通交叉熵，标签平滑为 0；不把早期建议中的标签平滑或 ROI 裁剪当作最终已采用配置。')
caption('表 1.2 两个分类分支的实际训练设置')
table(['参数','ResNet50 第二轮','EfficientNet-B3 第三轮'],[['输入尺寸','288×288','288×288'],['物理批量与累积','16 与 2','16 与 2'],['训练轮数','45','30'],['仅训练分类头','前 3 轮','前 2 轮'],['优化器','AdamW','AdamW'],['权重衰减','0.01','0.01'],['初始分类头学习率','0.001','0.001'],['解冻后主干最大学习率','0.0001','0.0001'],['解冻后分类头学习率','0.0003','0.0003'],['检查点选择','固定末轮原始权重','固定末轮原始权重']],[5.0,5.55,5.55])
p('分类头预热后解冻主干，学习率采用过渡预热与余弦下降，梯度范数裁剪为 1。梯度累积后的有效批量为 32。两个正式分支均按既定轮数完成五折训练，不从每轮验证值中挑选最高点作为分支结果；学习率按成功的优化器更新推进。')
image('plant_curve.png','图 1.2 五折平均验证准确率随训练轮次变化',6.45)
p('曲线为五个验证折同一轮的平均值，尚未叠加最终推理增强。曲线后段的细小波动不能直接等同于融合后 OOF 指标；后者需对全部 500 张对应折预测统一计算。')

# Chapter1 page3.
page();h('1.4 实验结果与对照')
caption('表 1.3 基于同一固定划分的植物分类结果')
table(['方案','OOF 准确率','说明'],[['首轮 ResNet50','64.60%','保留的早期基线'],['第二轮 ResNet50','90.00%','288 输入与四旋转 TTA'],['第二轮 D4 TTA','89.80%','增加视图未改善'],['ResNet50 在 320 尺度推理','90.80%','同一训练权重'],['EfficientNet-B3','90.80%','四旋转 TTA'],['两个分支等权融合','92.00%','460 / 500 正确']],[7.0,3.2,5.9])
p('固定等权融合较两个单分支均增加 6 个正确样本，准确率提升 1.20 个百分点，说明两种特征提取器的错误具有一定互补性。融合后 Macro-F1 为 0.9202，多类对数损失为 0.2247。该比较使用同一批 OOF 记录，包含模型与推理策略选择影响；不能把最高分视为无偏泛化估计。')
image('plant_confusion.png','图 1.3 融合模型的 OOF 混淆矩阵  行为真实类别 列为预测类别',5.25)
p('Black-grass 与 Loose Silky-bent 之间共有 29 次相互混淆，占全部 40 个错误的 72.5%。Common wheat 有 4 张被判为 Black-grass；Sugar beet 有 2 张被判为 Common wheat。Scentless Mayweed 在此划分的 100 张 OOF 样本上全部正确，但不能据此推断该类在新采集条件下必然达到 100%。')

# Chapter1 page4.
page();h('1.5 结果分析与方法局限')
caption('表 1.4 最终融合模型的分类别表现')
table(['类别','Precision','Recall','F1'],[[r['class_name'],fmt(r['precision']),fmt(r['recall']),fmt(r['f1'])] for r in plant['per_class']],[6.5,3.2,3.2,3.2])
p('五折准确率依次为 93%、89%、92%、95% 和 91%，说明小样本划分仍有明显波动。以 460 / 500 计算的 Wilson 95% 区间约为 89.29%～94.07%，但该区间并未校正反复模型选择和样本相关性，主要用于说明有限样本的不确定性。先前设置的 95% 总体准确率目标尚未达到，不能用第 4 折的 95% 代替总体结果。')
p('首轮结果较低，后续更换训练日程、解冻方式及输入尺度后准确率提高。但这些因素同时变化，本实验未提供逐因素控制的消融，因此不能将全部提升归因于某一种增强或模型结构。D4 增强比四旋转略低，也表明“增加推理视图必然提高精度”并不成立。')
p('从混淆关系看，细长叶片类别是主要瓶颈。低分辨率幼苗的叶宽、叶鞘与弯曲形态容易在缩放中丢失；背景与拍摄尺度又可能提供捷径。可进一步检查误分类原图的植株覆盖率与生长状态，并在有可靠植株编号时进行分组验证。基于绿色区域的辅助裁剪具有可解释性，但必须通过独立对照验证，避免裁掉细叶。')
h('1.6 设计特点与复查方法')
p('本任务的设计特点是将两种现有主干用于互补融合，并以固定划分、逐样本概率和完整 OOF 覆盖支撑比较。这里的创新属于任务适配和实验组织，不属于提出新的残差网络或 EfficientNet 结构。')
p('测试推理对各分支的五个折模型分别求概率均值，再按相同的 0.5∶0.5 权重融合，输出 378 行类别预测。测试标签未参与融合选择。训练数据、模型权重保留在本机，代码包保存固定划分、训练配置及评估证据。')
p('复查入口为 task01_plant/src；固定划分为 data/folds_v2.csv；训练日志在 evidence/round2_fast_resnet50 与 evidence/round3_effnet_b3；最终指标、混淆矩阵和 OOF 文件位于 evidence/round3_effnet_resnet_blend。检查样本数量应为 500、每个样本恰好出现一次，并与验证折标记对应。',indent=False)

# Chapter2 page1.
page(True);h('第二章 人脸情感分类',1);h('2.1 任务定义与重训数据')
p('本任务根据人脸图像识别 Angry、Fear、Happy、Sad、Surprise 和 Neutral 六类情感，提交标签依次为 0～5。原图分辨率较低，表情类别之间存在相似性，训练标签还可能包含噪声。除准确率外，采用 Macro-F1 和逐类召回率，避免样本较多的 Happy 类掩盖少数类别的错误。')
p('此前保留了 ConvNeXt-Tiny 与 CBAM 的源码，但原脚本将权重与 OOF 结果保存在临时目录，未保留可核验的训练产物。本章结果来自 2026 年 11 月 10 日重新执行的固定分组留出实验。原脚本中的两种子五折方案只作为历史设计说明，本次未重新运行该十模型方案。')
caption('表 2.1 情感数据数量与本次划分')
table(['类别','原始训练数','本次训练数','验证数'],[[n,audit['class_counts'][n],audit['train_counts'][n],audit['validation_counts'][n]] for n in ['Angry','Fear','Happy','Sad','Surprise','Neutral']]+[['合计',audit['total'],audit['train'],audit['validation']]],[5.3,3.6,3.6,3.6])
p(f'原始训练集共 {audit["total"]:,} 张，测试集为 7,065 张。逐图解码为灰度像素后计算 SHA-256，得到 {audit["unique_pixel_groups"]:,} 个像素组，重复组额外样本共 {audit["duplicate_extra_samples"]:,} 张。发现同图异标签组，排除其中全部 {audit["excluded_conflicting_labels"]} 张图像；其余同像素组不拆分，按类别与固定种子 20261110 分配约 80% 训练和 20% 验证。')
p('最终训练 22,533 张、验证 5,632 张，两部分像素哈希交集为 0。此策略防止完全相同的图像跨集合，但没有按人物身份分组，近重复图像或同一人物的不同帧仍可能跨集合。本实验保留有效组中的重复样本，因此评价以图像为单位，并非每个独立人物等权。')
h('2.2 预处理与增强')
p('先对灰度图像进行 CLAHE 局部对比度增强，clipLimit 为 2.0、网格为 8×8，再复制为 RGB 三通道并缓存。输入缩放为 224×224，使用 ImageNet 归一化。训练时水平翻转、RandAugment 两操作且强度为 9、概率 0.25 的随机擦除；验证仅执行固定缩放和归一化，不使用 TTA。')

# Chapter2 page2.
page();h('2.3 模型结构与损失设计')
p('分类主干为 timm 的 convnext_tiny.in12k_ft_in1k，使用本机缓存的通用图像预训练权重，不使用外部表情分类权重。ConvNeXt-Tiny 的四级主干在 224 输入下输出约 768×7×7 的特征图，随后加入 CBAM。通道注意力由全局均值和最大值经共享两层 1×1 卷积计算，压缩比为 16；空间注意力将通道均值与最大值拼接，使用 7×7 卷积得到空间权重。[3][4]')
image('fer_model.png','图 2.1 本次人脸情感分类模型结构',6.45)
p('注意力后的特征经全局平均池化得到 768 维向量，连接 Dropout 0.3 和六类线性层。768 维特征同时用于 Center Loss，使同类特征靠近可学习中心。该设计希望突出眼部、眉部和嘴部等区域，但本次未单独进行 CBAM 消融，不能声称注意力带来了已测定的提升。')
p('分类损失采用类别加权 Focal Loss，γ 为 2，类别权重仅根据本次训练子集统计，为训练总量除以六倍该类数量。调制项中的真实类别概率由未加权交叉熵计算，之后再乘类别权重，避免把加权损失误作概率。[5]')
eq('L = λ Lfocal(z, ya) + (1 − λ) Lfocal(z, yb) + 0.01 Lcenter')
p('Mixup 系数来自 Beta(0.2, 0.2)，并取 λ 与 1−λ 中较大者。混合图像对两种标签的 Focal Loss 加权；Center Loss 也对两个标签中心按同样权重求和。相较历史源码，本次明确修正了加权 Focal 中的概率定义，以及 Mixup 时只使用第一个中心标签的问题。上述修改属于实现一致性修正，未经对照不单独归因提分。[6]')
caption('表 2.2 本次实际训练配置')
table(['项目','配置'],[['优化器与权重衰减','AdamW，0.01'],['学习率','1×10⁻⁴；前 3 轮预热，随后余弦下降'],['轮数与批量','12 轮，batch size 32'],['EMA','每步更新，衰减 0.999'],['梯度与精度','梯度范数上限 5，BF16 混合精度'],['选择规则','验证 Macro-F1 最大，准确率用于并列判断']],[5.2,10.9])

# Chapter2 page3.
page();h('2.4 训练过程与验证结果')
p(f'训练运行于 RTX 4070 Laptop GPU，PyTorch 2.7.1 CUDA 12.6，timm 1.0.22。每轮训练约 705 个批次，模型参数与类别中心共同更新。训练完成 {len(fh)} 轮，最佳模型为第 {fer["best_epoch"]} 轮。第 4 轮后暂停过一次，从该轮完整检查点恢复；未完成的第 5 轮更新被丢弃后重跑。')
p('恢复包含原始模型、EMA、优化器、类别中心、主进程随机数状态及 DataLoader 生成器状态。多进程增强工作线程在恢复后重建，其增强随机序列未完全保留，因此不声称与不中断训练逐位相同。每轮日志和检查点的对应关系保存在记录中，后续结果只采用实际完成的轮次。')
image('fer_curve.png','图 2.2 重新训练的验证准确率与 Macro-F1',6.45)
caption('表 2.3 固定留出验证的实际结果')
table(['指标','结果'],[['验证图像数',fer['samples']],['正确图像数',fer['correct']],['准确率',pct(fer['accuracy'])],['Macro-F1',fmt(fer['macro_f1'])],['未加权验证交叉熵',fmt(fer['cross_entropy'])],['模型选择轮次',fer['best_epoch']],['多数类恒定预测准确率',pct(max(audit['validation_counts'].values())/audit['validation'])]],[6.6,9.5])
p('多数类基线始终预测验证集中最多的 Happy 类，只反映类别先验，不使用图像信息。当前模型超过该基线，说明特征具有表情区分能力。训练损失同时包含 Focal、Mixup 和 Center Loss，验证交叉熵则是不加权的普通交叉熵，两者定义不同，不能直接用数值高低判断训练与验证差距。')
p('最佳模型由同一验证集逐轮选出，故本表是用于选择的留出验证成绩。尚未构建额外独立测试集，也没有本次重训的 Kaggle 实测得分；不能把此准确率解释为比赛成绩或历史多种子五折结果。')

# Chapter2 page4.
page();h('2.5 分类错误与实验分析')
image('fer_confusion.png','图 2.3 最佳模型在固定留出验证集上的混淆矩阵',5.5)
caption('表 2.4 验证集各情感类别表现')
table(['类别','Precision','Recall','F1'],[[r['class_name'],fmt(r['precision']),fmt(r['recall']),fmt(r['f1'])] for r in fer['per_class']],[6.5,3.2,3.2,3.2])
recall_best=max(fer['per_class'],key=lambda r:r['recall']);recall_low=min(fer['per_class'],key=lambda r:r['recall'])
p(f'各类中，{recall_best["class_name"]} 的召回率最高，为 {pct(recall_best["recall"])}；{recall_low["class_name"]} 最低，为 {pct(recall_low["recall"])}。表情强度较低、遮挡、姿态变化和标签边界模糊均可能导致混淆，但仅凭混淆矩阵不能确认单个错误的成因，需要回看对应原图。')
p('本实验的主要特点是明确排除同图异标签冲突，并将完全重复样本整组划分，使结果具有更清晰的复查依据。Focal Loss、Center Loss、CBAM 和 EMA 均为已有方法的组合，本次没有完成逐模块消融，不能把综合结果解释为各模块均有效。后续应优先建立人物身份分组或独立数据源验证，再比较简化基线与注意力模型。')
p('复查入口为 task02_fer/train_report.py；evidence/manifest.csv 保存图像、类别、哈希和划分；history.jsonl 为逐轮日志；metrics.json、validation_predictions.csv 和 resume_record.json 分别保存结果、逐图概率及续训说明。best.pt 和 last.pt 在本机训练目录保存，未放入 GitHub 代码包。',indent=False)

# Chapter3 page1.
page(True);h('第三章 中央凹检测定位',1);h('3.1 任务与标注分析')
p('本任务在眼底彩色图像中估计中央凹的横纵坐标。训练数据同时提供点坐标与黄斑区域框，测试提交采用每张图相邻的 X、Y 两行。训练图像共 80 张，测试图像共 20 张。分辨率包含 1956×1934 与 2992×2000 两种，评价必须在还原后的原图坐标上计算，不能把网络输入上的像素误差直接作为最终误差。')
p('固定五折划分每折验证 16 张，训练 64 张，按相机分辨率尽量均衡；明确的重复图像整组分配。数据没有可靠的患者编号，无法完全排除同一患者不同图像跨折。80 张训练图的中央凹均可见，缺少真实不可见负例，因此不能评价“不可见时输出零”的识别能力。')
h('3.2 模型结构与定位路径')
p('早期基线采用 ResNet34 编码器与热图解码器，先在全图定位，再在粗预测附近裁剪精修区域。热图目标由点坐标生成高斯分布，输出峰值再解码并还原原图坐标。该方案提供点级监督，也作为检测器未检出时的自动回退来源。')
p('最终保留的主要方法为低学习率 YOLO11s 单类黄斑框检测器，使用官方 COCO 权重初始化，将训练标注框归一化为 YOLO 格式。检测器包含主干、多尺度特征融合与检测头；本实验未改写 YOLO11s 的网络模块，主要改变任务类别、训练参数及坐标评价方式。[9]')
image('fovea_model.png','图 3.1 检测框中心定位及自动回退流程',6.45)
p('推理选择有限坐标且置信度最高的有效框，输出左右边界均值为 x、上下边界均值为 y。未检出时自动读取对应验证折的热图基线结果，不人工修改坐标，不因低置信度将坐标强制置零。五折测试模型均在原图坐标系下预测，再对坐标等权平均。')
eq('x = (xleft + xright) / 2      y = (ytop + ybottom) / 2')
p('检测框中心与人工点标注并非必然一致，因此模型选择直接依据点坐标 RMSE，而非检测 mAP。框标注的中心偏差会形成误差来源，也限制了仅依赖检测框的精度上限。')

# Chapter3 page2.
page();h('3.3 实验方案与训练过程')
caption('表 3.1 低学习率 YOLO11s 的实际设置')
table(['参数','取值'],[['Ultralytics 版本','8.3.221'],['模型初始化','官方 YOLO11s COCO 权重'],['输入尺度与批量','imgsz 960，batch size 4'],['每折训练上限','160 轮'],['优化器','AdamW'],['初始学习率与末端比率','0.0001 与 0.01'],['权重衰减','0.0005'],['预热','3 轮，warmup_bias_lr 为 0'],['几何增强','±10° 旋转、5% 平移、尺度扰动、水平翻转'],['混合增强','Mosaic、Mixup、Copy-Paste 均关闭'],['模型选择','验证折点坐标 RMSE']],[6.1,10.0])
p('关闭大幅拼接增强是为了保留眼底解剖结构与单个黄斑区域的空间关系。检测置信度下限为 0.001，低阈值用于降低漏检，再通过点坐标误差评估保留模型。每折训练 160 轮，但保存的最佳轮次可能较早；最优轮次依据对应验证折选择，不能将末轮拟合情况等同于泛化最优。')
caption('表 3.2 五折检测模型选择记录')
table(['验证折','所选轮次','最终轮次','验证漏检回退数'],[['第 1 折',60,160,'1 / 16'],['第 2 折',140,160,'0 / 16'],['第 3 折',40,160,'0 / 16'],['第 4 折',45,160,'0 / 16'],['第 5 折',20,160,'0 / 16']],[4.5,3.4,3.4,4.8])
p('五折预测拼合后恰好覆盖 80 张训练图，计算横纵坐标误差、欧氏距离分布和阈值内命中率。本文的坐标 RMSE 将 x 与 y 共 160 个坐标误差合并求均方根；平均欧氏距离则先对每张图求二维距离再平均，两者定义不同。')
eq('RMSE = √[ Σ ((x̂ − x)² + (ŷ − y)²) / (2N) ]')
p('历史优化还比较了 ResNet34 和 ConvNeXt 热图、重复精定位、检测与精定位融合、直接坐标分支及两阶段检测。验证分数同时参与检查点与方案筛选，试验次数较多，存在选择偏差。只有保留原始全部方案记录，才能合理解释最后分数。')

# Chapter3 page3.
page();h('3.4 结果对照与误差分布')
caption('表 3.3 主要定位方案的五折坐标 RMSE')
table(['方案','OOF RMSE 像素'],[['初始热图精定位基线','20.997774'],['较早检测器','21.120902'],['低学习率检测器','16.579950'],['低学习率检测与原热图等权融合','16.987303'],['检测后再次精修','22.154215'],['冻结检测器的坐标分支','15.873620'],['两阶段检测器','53.241400']],[11.4,4.7])
p('低学习率检测器相对热图基线的坐标 RMSE 下降约 21.04%。后续坐标分支虽在汇总值上低于 16.579950，但未通过事先设定的跨折稳定性门槛，未据此替换已有较好在线版本。两阶段模型的大误差说明增加结构复杂度并不必然适合仅 80 张训练图的条件。')
image('fovea_folds.png','图 3.2 热图基线与低学习率检测器的逐折 RMSE',6.45)
caption('表 3.4 检测方案的综合误差指标')
table(['指标','数值'],[['坐标 RMSE','16.579950 px'],['x 坐标 RMSE','20.435566 px'],['y 坐标 RMSE','11.496830 px'],['平均与中位欧氏距离','14.554281 与 7.623930 px'],['95% 分位与最大欧氏距离','55.540415 与 104.496162 px'],['20 px 与 40 px 内比例','81.25% 与 91.25%']],[7.1,9.0])
p('第 3 折仍明显困难，最大欧氏误差超过 100 像素。平均欧氏距离高于中位数，说明误差呈长尾分布；横向误差也明显大于纵向。整体 RMSE 的改善仍伴随少数样本较大的残差，不能仅用一个平均数判断定位稳定性。')

# Chapter3 page4.
page();h('3.5 可视化与历史提交反馈')
image('fovea_examples.png','图 3.3 低误差 中位附近及最大误差的训练图折外定位',6.45)
p('示例根据折外欧氏误差排序选择低误差、中位附近和最大误差样本，绿色十字表示点标注，黄色圆圈表示预测。展示的是训练样本的对应验证折输出，没有使用测试图标注。高误差图应联合检查图像对比度、黄斑框与点标注一致性、病变及检测框尺度，不能仅凭图像外观给出确定因果解释。')
caption('表 3.5 历史提交反馈的不同口径')
table(['记录','成绩与状态'],[['原较好版本','14.39947，历史提交反馈记录'],['后续融合版本','15.94521，历史反馈较差'],['10 月 27 日全量重训 v12','反馈约 17，精确小数未保存'],['本地五折检测器','16.579950，OOF 坐标 RMSE']],[7.1,9.0])
p('全量重训 v12 使用全部 80 张训练图，三个种子分别为 20261027、20261028 和 20261029，固定训练 160 轮，平均第 80、120、160 轮 EMA 权重；三个全量模型各占 25%，原较好预测占 25%。已有反馈表明该候选未提分，因此保留 14.39947 版本。全量模型没有独立留出集，不能沿用旧五折分数作为其验证成绩。')
p('原版五折有多折在 20～60 轮选中，而全量候选偏向后期权重，同时还改变了训练集大小与种子。由于因素同时变化，仅凭一次约 17 的公开反馈无法确认退化原因。公开榜样本数量有限，也不能用榜单总分反推逐图标签或人工修正测试坐标。')
h('3.6 设计特点与复查方法')
p('本任务的特点是把框检测输出转换为课程要求的点坐标，以真正的点误差选择模型，并为漏检设计对应折回退。主要限制是小样本、验证选择偏差与不可见负例缺失。目标 RMSE≤5 尚未达到，后续应围绕独立验证、点与框标注一致性以及离群样本改进。')
p('复查入口为 task03_fovea/optimize_detector.py，标注与划分在 prepared；evidence/detector_v6 保存选模和逐图预测；evidence/20261027_提交反馈记录.md 保存后续反馈。测试 CSV 必须含 40 行，ID 顺序与模板一致，坐标有限且在图像边界内。',indent=False)

# Chapter4 page1.
page(True);h('第四章 眼底血管分割',1);h('4.1 任务与标签核验')
p('本任务对彩色眼底图像进行血管像素分割，训练图像与标注各 20 张，测试图像 20 张。图像尺寸为宽 565、高 584 像素。数据文件虽以 .jpg 命名，实际原图为 TIFF、标签为 GIF，读取时按文件真实格式解码。训练标签只有 0 和 255 两个值，其中 255 表示血管、0 表示背景。')
p('题目对可视化结果要求“血管为 0、背景为 255”，与训练标签极性相反。模型内部始终以 1 表示血管；展示图可反色满足文字要求，CSV 则按血管前景编码。不能将反色展示图直接作为提交编码输入，否则前景会颠倒。每张训练图血管占全图约 6.03%～11.59%，属于明显的前景稀疏任务。')
h('4.2 模型结构')
p('采用自行实现的轻量残差 U-Net，参数约 466 万，未使用预训练或外部训练图像。网络受 U-Net 跳跃连接和空间注意力启发，但不是 SA-UNet、IterNet 等论文的完整复现。输入为 RGB 加绿色通道 CLAHE，共四通道；像素归一化到 −1～1。[7]')
image('vessel_model.png','图 4.1 残差 U-Net 的编码 解码与集成流程',6.45)
caption('表 4.1 分割网络主要组成')
table(['模块','结构与输出'],[['输入','RGB 三通道与绿色 CLAHE 一通道'],['编码器','五级通道数 24、48、96、192、384'],['残差块','两层 3×3 卷积、GroupNorm、SiLU 与残差支路'],['下采样','2×2 最大池化，共四次'],['解码器','双线性上采样、跳跃拼接、残差块、空间注意力'],['主输出','1×1 卷积，单通道像素 logits'],['辅助输出','1/2 与 1/4 尺度的两层深监督']],[5.1,11.0])
p('训练增强后使用 592×592 画布，验证和推理保留 584×565 原尺寸。网络在右侧和底部补齐到 16 的倍数，输出裁回原尺寸，减少反复缩放对细血管的影响。GroupNorm 适合 batch size 2 的小批量设置；Dropout 为 0.1，空间注意力用于解码特征加权。')

# Chapter4 page2.
page();h('4.3 实验方案与训练过程')
p('采用整图五折交叉验证，每折 16 张训练、4 张验证，按标签血管比例排序后分桶分配，使各折前景比例尽量平衡。不采用同一图像的不同局部块跨训练与验证的划分。缺少患者 ID，仍不能保证同一患者的不同图像不会跨折。')
p('增强同时作用于图像与标签，包括 90° 旋转翻转、小角度旋转、尺度与平移。图像采用双线性插值，标签采用最近邻插值，避免生成混合类别。图像还进行适度伽马、颜色和噪声扰动；每轮从本折 16 张图像重复采样 160 个增强样本，重复增强不增加独立样本数。')
caption('表 4.2 训练配置与实际完成轮数')
table(['项目','配置或结果'],[['优化器与学习率','AdamW，初始学习率 0.001'],['权重衰减与预热','0.0001，前 5 轮预热后余弦下降'],['批量与每轮采样','batch size 2，160 个增强样本'],['训练上限与早停','最多 160 轮，至少 40 轮，耐心 35'],['EMA 与精度','EMA 0.99，AMP；损失归约为 FP32'],['各折完成轮数','74、89、76、89、100'],['各折最佳轮次','39、54、41、54、65']],[6.0,10.1])
p('损失主体为 0.4 倍 BCE 加 0.5 倍 Soft Dice；第 10 轮后开始线性增加 soft-clDice 项，10 轮内增至 0.1。clDice 通过软骨架重叠关注细血管的拓扑一致性，使用 10 次软骨架迭代。另在两个辅助尺度上分别以 0.2、0.1 权重加入 BCE 与 Dice 的等权组合。[8]')
eq('L = 0.4 LBCE + 0.5 LDice + 0.1 r LclDice + Laux')
p('每折用固定阈值 0.5 下的原尺寸验证 Dice 选择最佳 EMA 模型，最终再统一用折外概率校准输出阈值。阈值校准与模型选择分开记录，避免把最终校准指标混成逐轮选模指标。')
image('vessel_curve.png','图 4.2 五折训练的原尺寸验证 Dice 曲线 阈值固定为 0.5',6.45)

# Chapter4 page3.
page();h('4.4 评价口径与实验结果')
p('最终使用五个折模型等权平均概率，并使用八方向旋转翻转测试增强。对训练图的折外校准，每张图只使用没有训练过该图的模型生成概率。Dice 为两倍预测与标注交集除以两者像素数之和；本文按图像计算再取均值，不与全像素汇总 Dice 混用。')
eq('Dice = 2 |P ∩ G| / (|P| + |G|)')
p('官方转换脚本先将标签以双线性插值缩放到 512×512，再把所有非零像素作为前景，这会使边缘附近的微小非零值也计入前景。为核对这一规则，本实验同时报告原尺寸血管 Dice 与该规则下的代理 Dice。代理指标依赖对服务端标签处理的假设，不能称为已核实的 Kaggle 得分。')
caption('表 4.3 五折训练及最终折外评估')
table(['指标','结果','评价条件'],[['第 1 折最佳 Dice','0.823504','无 TTA，固定阈值 0.5'],['第 2 折最佳 Dice','0.840497','无 TTA，固定阈值 0.5'],['第 3 折最佳 Dice','0.809787','无 TTA，固定阈值 0.5'],['第 4 折最佳 Dice','0.781094','无 TTA，固定阈值 0.5'],['第 5 折最佳 Dice','0.823291','无 TTA，固定阈值 0.5'],['最终原尺寸 OOF Dice','0.817872','八方向 TTA，阈值 0.53'],['官方缩放代理 OOF Dice','0.842116','八方向 TTA，阈值 0.07']],[6.8,3.1,6.2])
p('原尺寸与代理指标分别在 0.01～0.99 范围以 0.01 步长校准，最优阈值为 0.53 和 0.07。这两个数值对应不同的标签空间与输出处理，不能把 0.842116 高于 0.817872 解释为同一测试任务上的提升。两者还都参与阈值选择，不能作为无偏独立测试成绩。')
p('在 CSV 阈值 0.07 下，原尺寸血管 Dice 为 0.790159；在原尺寸阈值 0.53 下，官方缩放代理 Dice 为 0.759545。这一交叉比较说明同一概率图的最佳阈值依赖评价定义。原尺寸输出用于观察细血管，提交输出先缩放概率到 512×512 再阈值化，两者分别导出。')
p('最终生成 20 行 RLE 预测，按行展开、1-based 起点编码。已检查 ID 顺序、起点与长度合法性、非重叠和 PNG 解码往返一致。折外校准与最终推理导出分别用时约 10.0 秒和 21.7 秒；各折含续训，保存的末次会话耗时不能相加冒充完整训练时长。')

# Chapter4 page4.
page();h('4.5 分割效果与局限分析')
image('vessel_examples.png','图 4.3 原图 标注与无 TTA 折外分割示例 展示阈值为 0.5',6.3)
p('示例采用各图对应验证折保存的概率，展示中等与较困难样本；图中 Dice 按展示阈值重新计算，不使用最终校准分数替换。血管主干通常比细小分支更容易识别，分叉、低对比度末梢和病变附近区域需重点检查。预测是否改善连通性还需专门统计，不能仅凭加入 clDice 就宣称拓扑性能已提升。')
p('本任务的设计特点是保留原分辨率细节、联合绿色 CLAHE 与 RGB、使用小批量友好的归一化与两层深监督，并把标签极性、缩放和编码纳入验证。没有进行强制连通、填孔或删除小连通域的后处理，避免把真实细血管当作孤立噪声移除。网络各模块的独立收益仍需控制变量消融检验。')
h('4.6 四项任务的综合认识与复查')
p('四项实验的共同经验是先保证数据和评价可靠，再讨论结构复杂度。植物分类的互补融合在固定 OOF 上有效，但更复杂的推理增强并不总能提高分数；情感分类需要控制重复图像泄漏；定位结果要关注长尾误差与不可见样本；分割则必须区分标签极性和不同分辨率下的指标。改进方案只有在相同划分与评价条件下比较，才有充分依据。')
p('血管分割复查入口为 task04_vessel/main.py，网络与损失位于 vessel/model.py 和 losses.py；evidence/fold0 至 fold4 保存逐轮日志和划分；训练与提交核验.json、折外验证逐图结果.csv 保存最终原始结果。四项代码统一随报告保留，GitHub 仓库链接见封面。',indent=False)

# References, unnumbered: four chapters remain exactly four.
page(True);h('参考文献',2)
refs=[
'[1] He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. CVPR, 2016.',
'[2] Tan M, Le Q V. EfficientNet Rethinking Model Scaling for Convolutional Neural Networks. ICML, 2019.',
'[3] Liu Z, Mao H, Wu C Y, et al. A ConvNet for the 2020s. CVPR, 2022.',
'[4] Woo S, Park J, Lee J Y, Kweon I S. CBAM Convolutional Block Attention Module. ECCV, 2018.',
'[5] Lin T Y, Goyal P, Girshick R, He K, Dollar P. Focal Loss for Dense Object Detection. ICCV, 2017.',
'[6] Wen Y, Zhang K, Li Z, Qiao Y. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV, 2016.',
'[7] Ronneberger O, Fischer P, Brox T. U-Net Convolutional Networks for Biomedical Image Segmentation. MICCAI, 2015.',
'[8] Shit S, Paetzold J C, Sekuboyina A, et al. clDice A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. CVPR, 2021.',
'[9] Ultralytics. Ultralytics YOLO11 documentation and software. https://docs.ultralytics.com/models/yolo11/  使用版本 8.3.221.',
'[10] Wightman R. PyTorch Image Models timm. https://github.com/huggingface/pytorch-image-models  本次情感重训使用版本 1.0.22.'
]
for s in refs:p(s,indent=False)
h('实验记录与代码对应说明',2)
p('本文表格和曲线由本机实际保存的配置、训练日志及逐样本评估文件生成。植物分类以最终融合 OOF 文件为依据；情感分类以本次重训完成记录为依据；定位的在线成绩标注为历史反馈记录；血管分割区分原尺寸和官方缩放代理口径。方法分析中的可能原因和下一步建议不视为已完成实验。',indent=False)
repo_link('GitHub 仓库  ')
for root in [doc.element,doc.styles.element]:
    for el in list(root.iter(qn('w:pBdr'))):el.getparent().remove(el)
path=OUT/'机器学习课程设计报告_姜广阔_20245621.docx';doc.save(path)
print(path);print('Paragraphs',len(doc.paragraphs),'tables',len(doc.tables),'images',len(doc.inline_shapes),'completed FER epochs',len(fh))
