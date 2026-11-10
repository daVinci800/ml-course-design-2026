import unittest
import torch
from fovea.detector_data import yolo_box,detector_points


class DetectorTests(unittest.TestCase):
    def test_box_normalization_uses_original_image_dimensions(self):
        self.assertEqual(yolo_box([100,200,300,400],1000,500),[.2,.6,.2,.4])
        with self.assertRaises(ValueError):yolo_box([300,200,100,400],1000,500)

    def test_top_box_center_and_automatic_missing_detection_fallback(self):
        class Boxes:
            xyxy=torch.tensor([[0,0,20,20],[100,200,140,260]],dtype=torch.float32)
            conf=torch.tensor([.1,.9])
            def __len__(self):return 2
        class Result:
            def __init__(self,boxes):self.boxes=boxes
        records=[dict(id=1,width=1000,height=800),dict(id=2,width=1000,height=800)]
        pred=detector_points([Result(Boxes()),Result(None)],records,{2:dict(xy=[300,400])})
        self.assertEqual(pred[1]['xy'],[120,230]);self.assertFalse(pred[1]['detector_miss'])
        self.assertEqual(pred[2]['xy'],[300,400]);self.assertTrue(pred[2]['detector_miss'])


if __name__=='__main__':unittest.main(verbosity=2)
