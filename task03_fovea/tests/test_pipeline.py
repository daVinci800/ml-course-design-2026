"""Verification only: no optimizer, real-data training, or pretrained downloads."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
from fovea.geometry import roi_matrix,transform_points,warp,decode_norm,box_mask
from fovea.model import Localizer
from fovea.loss import localization_loss
from fovea.data import FoveaDataset
from fovea.inference import make_submission,load_checkpoint,predict_records
from fovea.prepare import read_manifest

torch.set_num_threads(2)


class GeometryTests(unittest.TestCase):
    def test_round_trip_with_rotation_flip_padding(self):
        points=np.array([[0,0],[734.031331,1016.817066],[2991,1999],[-5,50]])
        for flip in [False,True]:
            for angle in [-12,0,11]:
                a=roi_matrix([1400,1000],640,512,angle,flip)
                norm=(transform_points(a,points)+0.5)/512
                np.testing.assert_allclose(decode_norm(norm,a,512),points,atol=1e-10)

    def test_pixel_center_warp_agrees_with_transformed_coordinate(self):
        yy,xx=np.mgrid[:96,:96]
        image=Image.fromarray(np.stack([xx*2,yy*2,np.zeros_like(xx)],axis=2).astype(np.uint8))
        for flip in [False,True]:
            a=roi_matrix([47.5,47.5],72,64,angle=8,flip=flip)
            output=np.asarray(warp(image,a,64))
            for q in [[22,25],[35,40],[31,31]]:
                original=transform_points(np.linalg.inv(a),np.array(q))
                np.testing.assert_allclose(output[q[1],q[0],:2],original*2,atol=1.2)

    def test_transformed_box_rasterization(self):
        a=roi_matrix([31.5,31.5],64,64,flip=True)
        mask=box_mask([8,8,23,23],a,64)
        self.assertEqual(mask.sum(),16)
        self.assertEqual(mask[3,12],1)
        self.assertEqual(mask[3,3],0)


class ModelTests(unittest.TestCase):
    def test_synthetic_forward_loss_gradient(self):
        torch.manual_seed(3)
        model=Localizer('resnet18',False,64)
        output=model(torch.randn(2,3,64,64))
        self.assertEqual(output['logits'].shape,(2,1,16,16))
        self.assertTrue(torch.isfinite(output['xy']).all())
        self.assertTrue(((output['xy']>=0)&(output['xy']<=1)).all())
        batch=dict(target=torch.tensor([[0.3,0.55],[0.6,0.4]]),side=torch.tensor([2000.,640.]),
                   coord_valid=torch.ones(2),presence=torch.ones(2),box=torch.zeros(2,1,16,16),box_valid=torch.ones(2))
        batch['box'][:,:,6:10,6:10]=1
        loss,parts=localization_loss(output,batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()  # Graph test only. No optimizer is created or stepped.
        self.assertTrue(torch.isfinite(model.heatmap.weight.grad).all())
        self.assertGreater(float(model.heatmap.weight.grad.abs().sum()),0)
        self.assertEqual(parts['presence'],0)

    def test_masked_invisible_point_has_no_coordinate_gradient(self):
        logits=torch.randn(1,1,16,16,requires_grad=True)
        xy=torch.tensor([[0.5,0.5]],requires_grad=True)
        output=dict(logits=logits,xy=xy,box=torch.zeros(1,1,16,16,requires_grad=True),presence=torch.zeros(1,requires_grad=True))
        batch=dict(target=torch.zeros(1,2),side=torch.tensor([2000.]),coord_valid=torch.zeros(1),
                   presence=torch.zeros(1),box=torch.zeros(1,1,16,16),box_valid=torch.zeros(1))
        loss,_=localization_loss(output,batch,presence_enabled=True)
        loss.backward()
        self.assertEqual(float(xy.grad.abs().sum()),0)
        self.assertGreater(float(output['presence'].grad),0)

    def test_convnext_forward(self):
        model=Localizer('convnext_tiny',False,64).eval()
        with torch.no_grad():
            output=model(torch.zeros(1,3,64,64))
        self.assertEqual(output['logits'].shape,(1,1,16,16))

    def test_checkpoint_roundtrip_scratch(self):
        with tempfile.TemporaryDirectory() as temp:
            model=Localizer('resnet18',False,64).eval()
            path=Path(temp)/'model.pt'
            torch.save(dict(format='fovea-v1',config=dict(backbone='resnet18',pretrained=False,decoder_channels=64),model=model.state_dict()),path)
            loaded,_=load_checkpoint(path,'cpu')
            with torch.no_grad():
                x=torch.randn(1,3,64,64)
                torch.testing.assert_close(model(x)['xy'],loaded(x)['xy'])

    def test_checkpoint_batchnorm_structure_without_download(self):
        from torchvision import models
        with tempfile.TemporaryDirectory() as temp:
            model=Localizer('resnet18',False,64)
            model.encoder=models.resnet18(weights=None)
            model.encoder.fc=torch.nn.Identity()
            model.eval()
            path=Path(temp)/'model.pt'
            torch.save(dict(format='fovea-v1',config=dict(backbone='resnet18',pretrained=True,decoder_channels=64),model=model.state_dict()),path)
            loaded,_=load_checkpoint(path,'cpu')
            with torch.no_grad():
                x=torch.randn(1,3,64,64)
                torch.testing.assert_close(model(x)['xy'],loaded(x)['xy'])


class DataAndSubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=read_manifest('prepared/manifest.json')

    def test_real_folds_group_duplicates_and_balance_camera(self):
        records=[r for r in self.manifest['records'] if r['split']=='train']
        self.assertEqual(len(records),80)
        for fold in range(5):
            a=[r for r in records if r['fold']==fold]
            b=[r for r in records if r['fold']!=fold]
            self.assertEqual(len(a),16)
            self.assertFalse({r['decoded_sha256'] for r in a}&{r['decoded_sha256'] for r in b})
            self.assertEqual(sum(r['width']==1956 for r in a),1)
        pair=[r for r in records if r['id'] in (39,77)]
        self.assertEqual(pair[0]['fold'],pair[1]['fold'])

    def test_real_coordinate_crop_roundtrip(self):
        records=[r for r in self.manifest['records'] if r['split']=='train']
        cfg=dict(size=128,repeats=1,crop_side=640)
        for stage in ['coarse','fine']:
            centers={r['id']:np.array(r['xy'])+[45,-27] for r in records}
            ds=FoveaDataset(records,stage,cfg,False,centers=centers)
            for i,r in enumerate(records):
                # Actual images, target transformation, mask dimensions, inverse map.
                sample=ds[i]
                restored=decode_norm(sample['target'].numpy(),sample['matrix'].numpy(),cfg['size'])
                np.testing.assert_allclose(restored,r['xy'],atol=0.0002)
                self.assertEqual(sample['image'].shape,(3,128,128))
                self.assertEqual(sample['box'].shape,(1,32,32))

    def test_fine_validation_refuses_ground_truth_center_fallback(self):
        r=next(r for r in self.manifest['records'] if r['split']=='train')
        ds=FoveaDataset([r],'fine',dict(size=128,crop_side=640),False)
        with self.assertRaisesRegex(ValueError,'Missing predicted coarse'):
            ds[0]

    def test_submission_template_order_and_zero_pair(self):
        records=[r for r in self.manifest['records'] if r['split']=='test']
        predictions={r['id']:[r['width']/2,r['height']/2] for r in reversed(records)}
        predictions[records[0]['id']]=[0.,0.]
        dimensions={r['id']:(r['width'],r['height']) for r in records}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'out.csv'
            make_submission(self.manifest['template'],predictions,path,dimensions)
            rows=list(csv.DictReader(path.open(newline='')))
            template=list(csv.DictReader(Path(self.manifest['template']).open(newline='')))
            self.assertEqual([r['ImageID'] for r in rows],[r['ImageID'] for r in template])
            self.assertEqual(len(rows),40)
            self.assertEqual(float(rows[0]['value']),0)
            self.assertEqual(float(rows[1]['value']),0)
            del predictions[records[0]['id']]
            with self.assertRaises(ValueError):
                make_submission(self.manifest['template'],predictions,path,dimensions)

    def test_submission_rejects_nonfinite_or_out_of_bounds(self):
        records=[r for r in self.manifest['records'] if r['split']=='test']
        predictions={r['id']:[100.,100.] for r in records}
        dimensions={r['id']:(r['width'],r['height']) for r in records}
        with tempfile.TemporaryDirectory() as temp:
            for bad in [[float('nan'),100.],[-1,100.],[99999,100.]]:
                predictions[records[0]['id']]=bad
                with self.assertRaises(ValueError):
                    make_submission(self.manifest['template'],predictions,Path(temp)/'out.csv',dimensions)

    def test_metric_matches_flattened_submission(self):
        from fovea.train import rmse
        records=[dict(id=1,xy=[10,20]),dict(id=2,xy=[30,40])]
        predictions={1:dict(xy=[13,24]),2:dict(xy=[30,40])}
        self.assertAlmostEqual(rmse(records,predictions),2.5)
        euclidean_rms=np.sqrt(np.mean([25,0]))
        self.assertAlmostEqual(euclidean_rms,np.sqrt(2)*rmse(records,predictions))

    def test_checkpoint_data_identity_allows_relocation_but_rejects_changed_gt(self):
        import copy
        from fovea.train import manifest_hash
        moved=copy.deepcopy(self.manifest)
        moved['root']='D:/relocated/detection'
        moved['template']='D:/relocated/detection/sample_submission.csv'
        for r in moved['records']:
            r['path']='D:/relocated/'+Path(r['path']).name
        self.assertEqual(manifest_hash(moved),manifest_hash(self.manifest))
        train=next(r for r in moved['records'] if r['split']=='train')
        train['xy'][0]+=1
        self.assertNotEqual(manifest_hash(moved),manifest_hash(self.manifest))

    def test_predict_original_coordinates_with_tta(self):
        class Center(torch.nn.Module):
            def forward(self,x):
                return dict(xy=torch.tensor([[0.5,0.5]],device=x.device),presence=torch.tensor([0.0],device=x.device))
        r=next(r for r in self.manifest['records'] if r['split']=='test')
        prediction=predict_records(Center(),[r],'coarse',dict(size=64),'cpu',tta=True)[r['id']]
        np.testing.assert_allclose(prediction['xy'],r['fov_center'],atol=1e-6)
        self.assertLess(prediction['tta_spread'],1e-6)


if __name__=='__main__':
    unittest.main(verbosity=2)
