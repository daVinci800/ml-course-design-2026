import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from fovea.point_branch import CoordinateLoss, extract_points, freeze_batchnorm_stats


class PointBranchTests(unittest.TestCase):
    def test_frozen_bn_statistics_keep_trainable_feature_gradients(self):
        model=torch.nn.Sequential(torch.nn.Conv2d(3,4,3,padding=1),torch.nn.BatchNorm2d(4))
        mean=model[1].running_mean.clone()
        variance=model[1].running_var.clone()
        model.train()
        freeze_batchnorm_stats(model)
        model(torch.randn(2,3,8,8)+5).square().mean().backward()
        torch.testing.assert_close(model[1].running_mean,mean)
        torch.testing.assert_close(model[1].running_var,variance)
        self.assertGreater(model[0].weight.grad.abs().sum().item(),0)
        self.assertGreater(model[1].weight.grad.abs().sum().item(),0)

    def test_large_error_has_restoring_gradient(self):
        prediction=torch.tensor([[[40.,50.,1.]]],requires_grad=True)
        loss=CoordinateLoss()(prediction,torch.tensor([[[10.,10.,2.]]]),torch.tensor([[True]]),torch.tensor([[400.]]))
        loss.backward()
        self.assertGreater(loss.item(),0)
        self.assertTrue((prediction.grad[0,0,:2]>0).all())
        self.assertEqual(prediction.grad[0,0,2].item(),0)

    def test_coordinate_loss_masks_invisible_and_handles_no_targets(self):
        p=torch.tensor([[[2.,3.,1.]]],requires_grad=True)
        loss=CoordinateLoss()(p,torch.zeros_like(p),torch.tensor([[False]]),torch.tensor([[1.]]))
        self.assertEqual(loss.item(),0)
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_keypoint_coordinates_used_instead_of_box_center(self):
        result=SimpleNamespace(boxes=SimpleNamespace(conf=torch.tensor([.7,.8]),__len__=lambda:2),
                               keypoints=SimpleNamespace(xy=torch.tensor([[[20.,30.]],[[40.,50.]]])))
        class Boxes:
            conf=torch.tensor([.7,.8])
            def __len__(self): return 2
        result.boxes=Boxes()
        p=extract_points([result],[dict(id=1,width=100,height=100)],{})
        self.assertEqual(p[1]['xy'],[40.,50.])
        self.assertFalse(p[1]['detector_miss'])

    def test_miss_falls_back_to_automatic_baseline(self):
        result=SimpleNamespace(boxes=None,keypoints=None)
        p=extract_points([result],[dict(id=1,width=100,height=100)],{1:dict(xy=[12.,34.])})
        self.assertEqual(p[1]['xy'],[12.,34.])
        self.assertTrue(p[1]['detector_miss'])

    def test_pose_folds_preserve_duplicate_groups_and_precise_xy(self):
        import optimize_point_v8 as module
        records=[dict(id=i,split='train',fold=i%5,group=str(i),path='',width=100,height=100,box=[10,20,40,50],xy=[24.2,36.7]) for i in range(5)]
        records.append(dict(records[0],id=5))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for r in records:
                path=root/f'{r["id"]}.jpg'
                path.write_bytes(b'test')
                r['path']=str(path)
            with patch.object(module,'WORK',root): paths=module.prepare(dict(records=records))
            for fold,path in enumerate(paths):
                train=(path.parent/f'train_{fold}.txt').read_text().splitlines()
                valid=(path.parent/f'val_{fold}.txt').read_text().splitlines()
                self.assertFalse(set(train)&set(valid))
                if fold==0:
                    self.assertEqual(len(valid),2)
                    self.assertTrue(all(Path(p).stem in ('0000','0005') for p in valid))
            values=(paths[0].parent/'labels/0000.txt').read_text().split()
            np.testing.assert_allclose([float(x) for x in values[-3:]],[.242,.367,2])


if __name__=='__main__': unittest.main()
