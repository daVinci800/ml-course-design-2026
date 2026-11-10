"""No training: only no-grad forwards, deterministic array checks and mocked I/O."""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from vessel.data import augment,make_folds,preprocess,load_label
from vessel.inference import transform,inverse_transform,binary_dice,predict_probability
from vessel.losses import dice_loss,cldice_loss,segmentation_loss
from vessel.model import VesselUNet
from vessel.rle import encode_rle,decode_rle,official_foreground,masks_to_csv,read_ids,resize_probability
from vessel.runtime import load_config

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)


class RLETests(unittest.TestCase):
    def test_documented_examples_and_edges(self):
        x=np.zeros((4,4),bool); x.ravel()[:3]=True; x.ravel()[9:14]=True
        self.assertEqual(encode_rle(x),'1 3 10 5')
        self.assertTrue(np.array_equal(decode_rle('1 3 10 5',(4,4)),x))
        self.assertEqual(encode_rle(np.zeros((2,2))), '')
        self.assertEqual(encode_rle(np.ones((2,2))), '1 4')
        self.assertEqual(encode_rle(np.array([[0,1],[1,0]])), '2 2')
        for s in ['0 1','1 0','-1 2','1','1 3 3 2','17 1','9 2 1 1']:
            with self.assertRaises(ValueError): decode_rle(s,(4,4))

    def test_official_converter_equivalence(self):
        official_path=ROOT/'data/segmentation/segmentation_to_csv.py'
        spec=importlib.util.spec_from_file_location('official',official_path)
        official=importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
        rng=np.random.default_rng(100)
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            for i,shape in enumerate([(584,565),(512,512),(7,11)]):
                for polarity in [False,True]:
                    mask=rng.random(shape)<0.08
                    if polarity: mask=~mask
                    p=Path(directory)/f'{i}_{polarity}.png'; Image.fromarray(mask.astype(np.uint8)*255).save(p); paths.append(p)
            official_strings=official.turn_to_str([str(p) for p in paths])
            for path,expected in zip(paths,official_strings):
                ours=encode_rle(official_foreground(path))
                self.assertEqual(ours,expected)
                self.assertTrue(np.array_equal(decode_rle(ours),official_foreground(path)))

    def test_csv_order_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); masks=root/'masks'; masks.mkdir()
            sample=root/'sample.csv'; sample.write_text('Id,Predicted\n2,\n1,\n',encoding='utf-8')
            for sid in ['1','2']: Image.fromarray(np.zeros((512,512),np.uint8)).save(masks/f'{sid}.png')
            rows=masks_to_csv(masks,sample,root/'out.csv')
            self.assertEqual([r['Id'] for r in rows],['2','1'])
            self.assertEqual(rows[0]['Predicted'],'')
            Image.fromarray(np.zeros((20,20),np.uint8)).save(masks/'1.png')
            with self.assertRaises(ValueError): masks_to_csv(masks,sample,root/'out.csv')


class ModelTests(unittest.TestCase):
    def test_shape_loss_and_unchanged_weights(self):
        config=load_config(ROOT/'configs/main.json'); config['base_channels']=8
        model=VesselUNet(4,8,0.1).eval()
        x=torch.randn(2,4,67,83); y=torch.zeros(2,1,67,83); y[:,:,20:40,30:32]=1
        before={k:v.clone() for k,v in model.state_dict().items()}
        with torch.inference_mode():
            result=model(x)
            loss,parts=segmentation_loss(result,y,config,30)
        self.assertEqual(result['logits'].shape,y.shape)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_topology_and_empty_masks(self):
        x=torch.zeros(1,1,32,32); x[:,:,4:28,15:17]=1
        with torch.inference_mode():
            self.assertLess(float(dice_loss(x,x)),1e-5)
            self.assertLess(float(cldice_loss(x,x,10)),1e-5)
            self.assertTrue(torch.isfinite(cldice_loss(x*0,x*0,10)))
            broken=x.clone(); broken[:,:,14:18]=0
            self.assertGreater(float(cldice_loss(broken,x,10)),float(cldice_loss(x,x,10)))

    def test_tta_rectangular_inverse(self):
        x=torch.arange(5*7).reshape(1,1,5,7)
        for k in range(4):
            for flip in [False,True]:
                self.assertTrue(torch.equal(inverse_transform(transform(x,k,flip),k,flip),x))
        class PixelModel(torch.nn.Module):
            def forward(self,z): return {'logits':z[:,:1]}
        inp=torch.randn(4,17,23)
        a=predict_probability(PixelModel(),inp,torch.device('cpu'),tta=False)
        b=predict_probability(PixelModel(),inp,torch.device('cpu'),tta=True)
        self.assertTrue(np.allclose(a,b,atol=1e-6))


class DataTests(unittest.TestCase):
    def test_all_actual_labels_and_folds(self):
        root=ROOT/'data/segmentation'
        for p in (root/'train/label').glob('*.jpg'):
            a=load_label(p)
            self.assertGreater(float(a.mean()),0.01); self.assertLess(float(a.mean()),0.3)
        a=make_folds(root); b=make_folds(root)
        self.assertEqual(a,b)
        held=[]
        for split in a['folds']:
            self.assertEqual(len(split['train_ids']),16); self.assertEqual(len(split['val_ids']),4)
            self.assertFalse(set(split['train_ids'])&set(split['val_ids']))
            held+=split['val_ids']
        self.assertEqual(len(held),20); self.assertEqual(len(set(held)),20)

    def test_augmentation_alignment_and_preprocess(self):
        config=load_config(ROOT/'configs/main.json')
        x=np.zeros((55,47,3),np.uint8); y=np.zeros((55,47),np.float32)
        x[10:40,20:25]=255; y[10:40,20:25]=1
        for seed in range(12):
            rgb,mask=augment(x.copy(),y.copy(),np.random.default_rng(seed))
            self.assertEqual(rgb.shape[:2],mask.shape)
            self.assertEqual(set(np.unique(mask)),{0.,1.})
            # Photo augmentation changes intensities, not geometry.
            bright=rgb.mean(-1)>128
            self.assertGreater(binary_dice(bright,mask),0.85)
            tensor=preprocess(rgb,config)
            self.assertEqual(tensor.shape[0],4); self.assertTrue(torch.isfinite(tensor).all())
            self.assertGreaterEqual(float(tensor.min()),-1); self.assertLessEqual(float(tensor.max()),1)


if __name__=='__main__': unittest.main()
