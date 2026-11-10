import unittest
import numpy as np
import torch

from optimize_rcnn_v10 import coordinate_box, transform_box, RetinaDataset


class TwoStageGeometryTests(unittest.TestCase):
    def test_box_supervision_uses_precise_coordinate_and_original_extent(self):
        r=dict(box=[10,20,50,60],xy=[31.4,38.2],width=100,height=100)
        box=coordinate_box(r)
        np.testing.assert_allclose((box[:2]+box[2:])/2,r['xy'])
        np.testing.assert_allclose(box[2:]-box[:2],[40,40])

    def test_affine_box_centre_tracks_point_under_rotation(self):
        import cv2
        box=np.array([10.,20.,50.,60.])
        matrix=cv2.getRotationMatrix2D((40,40),7,1.1)
        transformed=transform_box(box,matrix)
        centre=np.r_[(box[:2]+box[2:])/2,1]
        np.testing.assert_allclose((transformed[:2]+transformed[2:])/2,matrix@centre)

    def test_dataset_validation_does_not_require_or_access_labels(self):
        r=dict(id=1,width=100,height=80)
        image=np.ones((80,100,3),dtype=np.uint8)*128
        tensor,target,ident=RetinaDataset([r],{1:(image,1,1)})[0]
        self.assertEqual(target,{})
        self.assertEqual(ident,1)
        self.assertEqual(tensor.shape,(3,80,100))
        torch.testing.assert_close(tensor,torch.full_like(tensor,128/255))

    def test_training_augmentation_is_epoch_reproducible(self):
        r=dict(id=1,width=300,height=280,box=[110,100,170,160],xy=[141.2,131.5])
        cache={1:(np.ones((280,300,3),dtype=np.uint8)*128,1,1)}
        dataset=RetinaDataset([r],cache,True,seed=7)
        first=dataset[0]
        second=dataset[0]
        torch.testing.assert_close(first[0],second[0])
        torch.testing.assert_close(first[1]['boxes'],second[1]['boxes'])
        dataset.epoch=1
        third=dataset[0]
        self.assertFalse(torch.equal(first[1]['boxes'],third[1]['boxes']))


if __name__=='__main__': unittest.main()
