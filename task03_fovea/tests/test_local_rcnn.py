import unittest
import numpy as np
from rcnn_refine_v11 import choose_local


class CoarseGuidedSelectionTests(unittest.TestCase):
    def test_stronger_distant_false_positive_does_not_override_local_candidate(self):
        point,missing=choose_local([[10,10,20,20],[100,100,120,120]],[.8,.99],[15,15],20)
        np.testing.assert_allclose(point,[15,15])
        self.assertFalse(missing)

    def test_only_distant_boxes_fall_back_to_automatic_coarse_point(self):
        point,missing=choose_local([[100,100,120,120]],[.99],[15,15],20)
        np.testing.assert_allclose(point,[15,15])
        self.assertTrue(missing)

    def test_empty_and_invalid_candidates_fall_back(self):
        for boxes,scores in [([],[]),([[np.nan,0,2,2]],[.9])]:
            point,missing=choose_local(boxes,scores,[12,34],20)
            np.testing.assert_allclose(point,[12,34])
            self.assertTrue(missing)


if __name__=='__main__': unittest.main()
