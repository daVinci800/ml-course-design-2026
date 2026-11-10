import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from fovea.decoding import decode_heatmap
from fovea.policies import policy_settings,refinement_windows
from fovea.inference import predict_records
from fovea.data import FoveaDataset
from fovea.geometry import decode_norm
from fovea.budget import planned_steps
from fovea.ablate import ablate
from unittest.mock import patch


def gaussian(x,y,w=32,h=28,sigma=2):
    gy,gx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    return (-((gx-x)**2+(gy-y)**2)/(2*sigma**2))[None,None].float()


class UpgradeTests(unittest.TestCase):
    def test_dark_recovers_subpixel_gaussian_without_training(self):
        target=torch.tensor([[13.27,12.68]])
        decoded=decode_heatmap(gaussian(*target[0]),'dark',blur_sigma=0)*torch.tensor([32,28])-0.5
        torch.testing.assert_close(decoded,target,atol=2e-5,rtol=0)
        blurred=decode_heatmap(gaussian(*target[0]),'dark')*torch.tensor([32,28])-0.5
        torch.testing.assert_close(blurred,target,atol=.01,rtol=0)

    def test_dark_avoids_global_average_between_two_synthetic_peaks(self):
        first=gaussian(8.2,12.4).exp()
        second=gaussian(25.2,12.4).exp()*.65
        logits=(first+second).clamp_min(1e-15).log()
        integral=decode_heatmap(logits)*torch.tensor([32,28])-.5
        dark=decode_heatmap(logits,'dark')*torch.tensor([32,28])-.5
        self.assertGreater(float(integral[0,0]),13)
        self.assertLess(abs(float(dark[0,0])-8.2),.05)

    def test_flip_preserves_pixel_center_convention(self):
        logits=gaussian(13.27,12.68)
        for mode in ['integral','dark']:
            a,b=decode_heatmap(logits,mode),decode_heatmap(logits.flip(-1),mode)
            torch.testing.assert_close(a[:,0],1-b[:,0],atol=1e-6,rtol=0)
            torch.testing.assert_close(a[:,1],b[:,1],atol=1e-6,rtol=0)

    def test_flat_border_small_maps_and_invalid_inputs(self):
        flat=decode_heatmap(torch.zeros(2,1,28,32),'dark')
        torch.testing.assert_close(flat,torch.full((2,2),.5))
        border=decode_heatmap(gaussian(0,0),'dark',0)
        torch.testing.assert_close(border,torch.tensor([[.5/32,.5/28]]))
        small=torch.zeros(1,1,3,3)
        torch.testing.assert_close(decode_heatmap(small,'dark'),decode_heatmap(small))
        with self.assertRaises(ValueError): decode_heatmap(small,'unknown')
        with self.assertRaises(ValueError): decode_heatmap(small,'dark',-1)

    def test_integral_decoder_retains_differentiable_training_path(self):
        logits=torch.randn(2,1,12,12,requires_grad=True)
        decode_heatmap(logits).square().sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0)

    def test_dual_decoding_reuses_network_forward_and_matches_separate_passes(self):
        class Peak(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls=0
            def forward(self,x):
                self.calls+=1
                logits=gaussian(13.27,12.68).expand(len(x),-1,-1,-1)
                return dict(logits=logits,xy=decode_heatmap(logits),presence=torch.zeros(len(x)))
        manifest=json.loads(Path('prepared/manifest.json').read_text(encoding='utf8'))
        records=[r for r in manifest['records'] if r['split']=='train'][:2]
        settings=dict(size=64,inference_batch_size=4)
        model=Peak()
        dual=predict_records(model,records,'coarse',settings,'cpu',decoder_methods=('integral','dark'))
        calls=model.calls
        self.assertEqual(calls,2)
        for method in ['integral','dark']:
            single=predict_records(model,records,'coarse',dict(settings,decoder=method),'cpu')
            self.assertEqual(dual[method],single)
        self.assertEqual(model.calls,3*calls)

    def test_ablation_uses_only_matching_validation_fold_and_rejects_mismatch(self):
        import hashlib
        from fovea.train import manifest_hash
        manifest=json.loads(Path('prepared/manifest.json').read_text(encoding='utf8'))
        manifest=copy.deepcopy(manifest)
        manifest['n_folds']=2
        manifest['records']=[r for r in manifest['records'] if r['split']=='train' and r['fold'] in (0,1)]
        key=manifest_hash(manifest)
        with tempfile.TemporaryDirectory() as temp:
            for fold in (0,1):
                d=Path(temp)/f'fold_{fold}'; d.mkdir()
                (d/'coarse_best.pt').write_bytes(b'fake-coarse')
                (d/'fine_best.pt').write_bytes(b'fake-fine')
            def load(path,device):
                fold=int(Path(path).parent.name[-1])
                stage='coarse' if 'coarse' in Path(path).name else 'fine'
                return object(),dict(fold=fold,stage=stage,manifest_hash=key,
                    coarse_sha256=hashlib.sha256(b'fake-coarse').hexdigest(),
                    config=dict(coarse=dict(size=64),fine=dict(size=64,crop_side=512)))
            def predict(model,records,stage,settings,device,**kwargs):
                self.assertEqual(len({r['fold'] for r in records}),1)
                self.assertTrue(all(r['split']=='train' for r in records))
                pred={r['id']:dict(xy=r['xy'],tta_spread=5.) for r in records}
                return {name:pred for name in kwargs['decoder_methods']} if 'decoder_methods' in kwargs else pred
            with patch('fovea.ablate.load_checkpoint',side_effect=load),patch('fovea.ablate.predict_records',side_effect=predict) as mocked,patch('builtins.print'):
                report=ablate(manifest,temp,Path(temp)/'out','cpu')
                self.assertEqual(mocked.call_count,6)
                self.assertFalse(report['training_started'])
                for stats in report['policies'].values(): self.assertEqual(stats['rmse'],0)
            def wrong(path,device):
                model,ckpt=load(path,device); ckpt['manifest_hash']='changed'; return model,ckpt
            with patch('fovea.ablate.load_checkpoint',side_effect=wrong):
                with self.assertRaises(ValueError): ablate(manifest,temp,Path(temp)/'bad','cpu')

    def test_adaptive_sides_are_bounded_and_do_not_consult_labels(self):
        settings=dict(crop_side=512)
        cp={1:dict(xy=[700,800],tta_spread=0),2:dict(xy=[900,600],tta_spread=20),3:dict(xy=[600,500],tta_spread=500)}
        centers,sides=refinement_windows(cp,policy_settings(settings,'dark_adaptive'))
        self.assertEqual(sides,{1:512,2:592,3:768})
        self.assertEqual(centers,{k:v['xy'] for k,v in cp.items()})
        _,fixed=refinement_windows(cp,policy_settings(settings,'integral_fixed'))
        self.assertEqual(set(fixed.values()),{512})
        self.assertEqual(settings,dict(crop_side=512))
        with self.assertRaises(ValueError): refinement_windows(cp,dict(crop_side=512,adaptive_crop_max=256))

    def test_adaptive_crops_use_actual_affine_to_restore_original_pixels(self):
        class Center(torch.nn.Module):
            def forward(self,x):
                return dict(xy=torch.full((len(x),2),.5),presence=torch.zeros(len(x)))
        manifest=json.loads(Path('prepared/manifest.json').read_text(encoding='utf8'))
        records=[r for r in manifest['records'] if r['split']=='train'][:2]
        centers={r['id']:[900,800] for r in records}
        sides={records[0]['id']:512,records[1]['id']:768}
        pred=predict_records(Center(),records,'fine',dict(size=64,crop_side=512,inference_batch_size=4),'cpu',centers=centers,crop_sides=sides)
        for r in records: np.testing.assert_allclose(pred[r['id']]['xy'],centers[r['id']],atol=1e-8)

    def test_wide_training_crops_keep_tensor_resolution_and_coordinate_roundtrip(self):
        manifest=json.loads(Path('prepared/manifest.json').read_text(encoding='utf8'))
        r=next(r for r in manifest['records'] if r['split']=='train')
        settings=dict(size=64,crop_side=512,wide_crop_probability=1,wide_crop_scale_range=[1.5,1.5])
        ds=FoveaDataset([r],'fine',settings,training=True)
        for epoch in range(4):
            ds.epoch=epoch
            batch=ds[0]
            self.assertEqual(batch['image'].shape,(3,64,64))
            self.assertGreater(float(batch['side']),680)
            restored=decode_norm(batch['target'].numpy(),batch['matrix'].numpy(),64)
            np.testing.assert_allclose(restored,r['xy'],atol=.0001)

    def test_upgrade_keeps_same_training_step_budget(self):
        manifest=json.loads(Path('prepared/manifest.json').read_text(encoding='utf8'))
        config=json.loads(Path('configs/accuracy_5h.json').read_text(encoding='utf8'))
        self.assertEqual(planned_steps(manifest,config,list(range(5)))['max_optimizer_steps'],32800)
        self.assertEqual(config['time_budget_seconds'],16200)


if __name__=='__main__': unittest.main(verbosity=2)
