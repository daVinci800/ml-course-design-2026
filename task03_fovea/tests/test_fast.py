import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from fovea.budget import RunBudget,planned_steps,fit_epoch_limit
from fovea.cache import build_cache,render_record,cache_path,mapped_image,compact_image,global_matrix,global_cache_path
from fovea.data import FoveaDataset,EpochSampler,worker_init,stage_settings
from fovea.geometry import roi_matrix,warp
from fovea.inference import predict_records
from fovea.model import Localizer
from fovea.train import epoch_sigma,update_ema


class FastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=json.loads(Path('prepared/manifest.json').read_text(encoding='utf-8'))
        cls.records=[r for r in cls.manifest['records'] if r['split']=='train']
        cls.fast=json.loads(Path('configs/fast_5h.json').read_text(encoding='utf-8'))

    def test_exact_step_reduction_keeps_five_folds(self):
        old=json.loads(Path('configs/resnet34.json').read_text(encoding='utf-8'))
        before=planned_steps(self.manifest,old,list(range(5)))
        after=planned_steps(self.manifest,self.fast,list(range(5)))
        self.assertEqual(before['max_optimizer_steps'],275200)
        self.assertEqual(after['max_optimizer_steps'],32800)
        self.assertEqual(len(after['stages']),10)
        self.assertEqual(after['time_budget_seconds'],4.5*3600)

    def test_budget_reserves_future_stages_and_survives_resume(self):
        tick=[100.]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'budget.json'
            budget=RunBudget(path,1000,clock=lambda:tick[0])
            self.assertEqual(budget.stage_deadline([1,1],60),540)
            tick[0]=300
            budget.save()
            tick[0]=900
            resumed=RunBudget(path,1000,resume=True,clock=lambda:tick[0])
            self.assertEqual(resumed.remaining,800)
            tick[0]+=50
            self.assertEqual(resumed.remaining,750)
            with self.assertRaises(FileExistsError):
                RunBudget(path,1000,clock=lambda:tick[0])
            with self.assertRaises(ValueError):
                RunBudget(path,1100,resume=True,clock=lambda:tick[0])
            tick[0]+=740
            with self.assertRaises(TimeoutError):
                resumed.stage_deadline([1,1],60)

    def test_throughput_shortens_schedule_without_exceeding_epoch_cap(self):
        self.assertEqual(fit_epoch_limit(90,3,10,300),31)
        self.assertEqual(fit_epoch_limit(90,3,10,10000),90)
        self.assertEqual(fit_epoch_limit(90,3,10,0),3)
        with self.assertRaises(ValueError):
            fit_epoch_limit(90,3,0,100)

    def test_cached_pixels_and_transformed_crops_match_original(self):
        r=self.records[0]
        with tempfile.TemporaryDirectory() as temp:
            info=build_cache([r,r],temp)
            self.assertEqual(info['created'],1)
            self.assertEqual(info['reused'],1)
            with Image.open(r['path']) as image:
                image=image.convert('RGB')
                for center,side in [(r['fov_center'],r['fov_side']),([-20,20],512),(r['xy'],512),([-1000,-1000],100)]:
                    for flip in [False,True]:
                        a=roi_matrix(center,side,128,angle=11,flip=flip)
                        direct=np.asarray(warp(image,a,128)).astype(int)
                        cached=np.asarray(render_record(r,a,128,temp)).astype(int)
                        self.assertLessEqual(np.max(abs(direct-cached)),1)
            mapped_image.cache_clear()
            compact_image.cache_clear()

    def test_compact_global_cache_keeps_the_same_coordinate_frame(self):
        r=self.records[0]
        a=roi_matrix(r['fov_center'],r['fov_side'],128,angle=11,flip=True)
        with tempfile.TemporaryDirectory() as temp:
            build_cache([r],temp)
            rendered=np.asarray(render_record(r,a,128,temp,coarse=True))
            source=Image.fromarray(np.load(global_cache_path(r,temp),allow_pickle=False))
            reference=np.asarray(warp(source,a @ np.linalg.inv(global_matrix(r)),128))
            np.testing.assert_array_equal(rendered,reference)
            mapped_image.cache_clear()
            compact_image.cache_clear()

    def test_duplicate_denoising_only_changes_training_targets(self):
        pair=[r for r in self.records if r['id'] in [39,77]]
        settings=dict(size=64,crop_side=512,average_duplicate_targets=True)
        train=FoveaDataset(pair,'fine',settings,training=True)
        mean=np.mean([r['xy'] for r in pair],axis=0)
        np.testing.assert_allclose(train.target_overrides[39],mean)
        np.testing.assert_allclose(train.target_overrides[77],mean)
        self.assertEqual(train.sample_weights[39],0.5)
        valid=FoveaDataset(pair,'fine',settings,training=False,centers={r['id']:r['xy'] for r in pair})
        self.assertEqual(valid.target_overrides,{})
        self.assertNotEqual(pair[0]['xy'],pair[1]['xy'])

    def test_sigma_sharpens_without_changing_old_configuration(self):
        settings=self.fast['fine']
        self.assertEqual(epoch_sigma(settings,0),1.8)
        self.assertEqual(epoch_sigma(settings,settings['epochs']-1),1.0)
        self.assertGreater(epoch_sigma(settings,10),epoch_sigma(settings,40))
        self.assertEqual(epoch_sigma(dict(sigma=1.7,epochs=70),10),1.7)

    def test_epoch_indices_change_with_persistent_workers(self):
        settings=dict(size=64,repeats=1)
        ds=FoveaDataset(self.records[:4],'coarse',settings,training=True,seed=3)
        sampler=EpochSampler(ds,3)
        loader=DataLoader(ds,batch_size=2,sampler=sampler,num_workers=2,persistent_workers=True,worker_init_fn=worker_init)
        first={int(i):v.clone() for batch in loader for i,v in zip(batch['id'],batch['image'])}
        ds.epoch=1
        second={int(i):v.clone() for batch in loader for i,v in zip(batch['id'],batch['image'])}
        self.assertEqual(set(first),set(second))
        self.assertTrue(any(not torch.equal(first[k],second[k]) for k in first))
        ds.epoch=0
        repeated={int(i):v.clone() for batch in loader for i,v in zip(batch['id'],batch['image'])}
        for ident in first:
            torch.testing.assert_close(first[ident],repeated[ident])
        del loader

    def test_batched_tta_matches_single_view_batches(self):
        class Center(torch.nn.Module):
            def forward(self,x):
                return dict(xy=torch.full((len(x),2),0.5,device=x.device),presence=torch.zeros(len(x),device=x.device))
        records=self.records[:3]
        settings=dict(size=64,inference_batch_size=1)
        single=predict_records(Center(),records,'coarse',settings,'cpu')
        settings['inference_batch_size']=4
        batched=predict_records(Center(),records,'coarse',settings,'cpu')
        self.assertEqual(single,batched)

    def test_foreach_ema_matches_reference_on_synthetic_parameters(self):
        model=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.BatchNorm1d(3))
        ema=copy.deepcopy(model)
        with torch.no_grad():
            for p in ema.parameters():
                p.zero_()
            model[1].running_mean.fill_(2)
        expected=[p.detach().clone()*0.02 for p in model.parameters()]
        update_ema(ema,model,0.98)
        for actual,reference in zip(ema.parameters(),expected):
            torch.testing.assert_close(actual,reference)
        torch.testing.assert_close(ema[1].running_mean,model[1].running_mean)


if __name__=='__main__':
    unittest.main(verbosity=2)
