import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import final_round as final


class FinalRoundGeometryTests(unittest.TestCase):
    def test_json_write_retries_windows_sharing_violation(self):
        replace=Path.replace
        calls=[]
        def transient(path,target):
            calls.append(path)
            if len(calls)<3: raise PermissionError('Windows sharing violation')
            return replace(path,target)
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'status.json'
            final.write(target,dict(epoch=1))
            with patch.object(Path,'replace',transient),patch.object(final.time,'sleep'):
                final.write(target,dict(epoch=2))
            self.assertEqual(final.read(target),dict(epoch=2))
            self.assertEqual(len(calls),3)
            self.assertFalse(list(Path(td).glob('*.tmp')))

    def test_json_write_can_update_when_windows_denies_replacement(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'status.json'
            final.write(target,dict(epoch=1))
            with patch.object(Path,'replace',side_effect=PermissionError('Replace denied')),patch.object(final.time,'sleep'):
                final.write(target,dict(epoch=2))
            self.assertEqual(final.read(target),dict(epoch=2))
            self.assertFalse(list(Path(td).glob('*.tmp')))

    def test_process_liveness_distinguishes_exit_from_running(self):
        import os
        import subprocess
        import sys
        from run_final_round import process_alive
        self.assertTrue(process_alive(os.getpid()))
        self.assertFalse(process_alive(None))
        self.assertFalse(process_alive(0))
        child=subprocess.Popen([sys.executable,'-c','pass'])
        child.wait(timeout=15)
        self.assertFalse(process_alive(child.pid))

    def test_missing_roi_detector_preserves_automatic_coarse_point(self):
        import cv2
        from types import SimpleNamespace
        class EmptyBoxes:
            def __len__(self): return 0
        class MissingModel:
            def predict(self,**kwargs): return [SimpleNamespace(boxes=EmptyBoxes())]
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'image.png'
            cv2.imwrite(str(path),np.zeros((80,100,3),dtype=np.uint8))
            records=[dict(id=1,path=str(path),height=80,width=100)]
            coarse={1:dict(xy=[20.,30.])}
            result=final.predict(MissingModel(),records,{1:dict(xy=[80.,40.])},space='roi',centers=coarse)
            self.assertTrue(result[1]['detector_miss'])
            self.assertEqual(result[1]['xy'],coarse[1]['xy'])

    def test_dense_center_ignores_different_objects(self):
        from fovea.box_fusion import dense_center
        result=dense_center([[0,0,10,10],[1,0,11,10],[30,30,40,40]],[1,.5,.9])
        np.testing.assert_allclose(result,[5.2,5])

    def test_dense_center_rejects_invalid_hypotheses(self):
        from fovea.box_fusion import dense_center
        self.assertIsNone(dense_center([[0,0,0,10],[np.nan,0,10,10]],[1,1]))
        with self.assertRaises(ValueError): dense_center([[0,0,10,10]],[])

    def test_dense_center_filters_low_relative_confidence(self):
        from fovea.box_fusion import dense_center
        np.testing.assert_allclose(dense_center([[0,0,10,10],[1,0,11,10]],[1,.01]),[5,5])

    def test_crop_origin_and_padding(self):
        image=np.arange(10*12*3,dtype=np.uint8).reshape(10,12,3)
        view,(x,y)=final.crop(image,[1,2],8)
        self.assertEqual((x,y),(-3,-2))
        self.assertEqual(view.shape,(8,8,3))
        np.testing.assert_array_equal(view[2,3],image[0,0])
        np.testing.assert_array_equal(view[:2],0)

    def test_crop_identity_inside(self):
        image=np.arange(10*12*3,dtype=np.uint8).reshape(10,12,3)
        view,origin=final.crop(image,[6,5],6)
        self.assertEqual(origin,(3,2))
        np.testing.assert_array_equal(view,image[2:8,3:9])

    def test_invalid_boxes_are_rejected(self):
        from fovea.detector_data import yolo_box
        with self.assertRaises(ValueError):
            yolo_box([-1,0,10,10],100,100)

    def test_fold_lists_keep_all_augmented_views_together(self):
        import cv2
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            records=[]
            for i in range(5):
                path=root/f'{i}.png'
                cv2.imwrite(str(path),np.zeros((1400,1600,3),dtype=np.uint8))
                records.append(dict(id=i,path=str(path),split='train',fold=i,group=str(i),
                    xy=[800,700],box=[700,600,900,800],fov_center=[800,700],fov_side=1300))
            with patch.object(final,'WORK',root/'work'):
                yamls=final.prepare_data(dict(records=records),'roi')
            for fold,yaml in enumerate(yamls):
                dest=yaml.parent
                tr=(dest/f'train_{fold}.txt').read_text().splitlines()
                va=(dest/f'val_{fold}.txt').read_text().splitlines()
                self.assertEqual(len(tr),16)
                self.assertEqual(len(va),4)
                self.assertTrue(all(Path(p).stem.startswith(f'{fold:04d}_') for p in va))
                self.assertTrue(set(tr).isdisjoint(va))
                self.assertTrue(all(not Path(p).stem.startswith(f'{fold:04d}_') for p in tr))


if __name__=='__main__':
    unittest.main()
