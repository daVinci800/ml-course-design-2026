"""Exercise export/calibration with an analytic model, no learned checkpoints."""
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from vessel.data import make_folds
from vessel.pipeline import calibrate,predict
from vessel.rle import resize_probability,decode_rle
from vessel.runtime import training_fingerprint,load_config,save_json

ROOT=Path(__file__).resolve().parents[1]


class AnalyticModel(torch.nn.Module):
    """No parameters, optimizer or training: a deterministic test function."""
    def forward(self,x):
        return {'logits':x[:,1:2]*3}


class PipelineTests(unittest.TestCase):
    def test_oof_ensemble_both_polarities_and_rle(self):
        data=ROOT/'data/segmentation'
        config=load_config(ROOT/'configs/main.json')
        folds=make_folds(data,config['folds'],config['seed'])
        fingerprint=training_fingerprint(data)
        folds['training_fingerprint']=fingerprint
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)/'run'; splitfile=Path(directory)/'splits.json'; save_json(splitfile,folds)
            for split in folds['folds']:
                p=run/f'fold{split["fold"]}/best.pt'; p.parent.mkdir(parents=True)
                # Bytes only support hash provenance; torch.load is never called.
                p.write_bytes(b'ANALYTIC TEST FIXTURE - NOT MODEL WEIGHTS')
            def analytic_load(path,device):
                fold=int(Path(path).parent.name.replace('fold',''))
                return AnalyticModel().to(device),{'config':config,'split':folds['folds'][fold],
                    'training_fingerprint':fingerprint,'completed_epochs':1}
            common={'device':'cpu','splits':str(splitfile),'data':str(data),'run_dir':str(run),'tta':True}
            with patch('vessel.pipeline.checkpoint_load',side_effect=analytic_load),contextlib.redirect_stdout(io.StringIO()):
                calibrate(SimpleNamespace(**common,metric='official-proxy'))
                calibration=json.loads((run/'calibration.json').read_text(encoding='utf-8'))
                self.assertEqual(calibration['num_oof_images'],20)
                self.assertEqual(len(list((run/'oof_tta').glob('*.npy'))),20)
                for foreground in ['vessel','background']:
                    output=Path(directory)/foreground
                    predict(SimpleNamespace(**common,output=str(output),csv_foreground=foreground))
                    self.assertEqual(len(list((output/'submission_masks_512').glob('*.png'))),20)
                    prob=np.load(output/'probabilities/1.npy')
                    vessel=prob>=calibration['native_threshold']
                    native=np.asarray(Image.open(output/'masks_vessel0_native/1.png'))
                    self.assertTrue(np.array_equal(native==0,vessel))
                    sub=np.asarray(Image.open(output/'submission_masks_512/1.png'))
                    vessel512=resize_probability(prob)>=calibration['threshold']
                    self.assertTrue(np.array_equal(sub>0,vessel512 if foreground=='vessel' else ~vessel512))
                    import csv
                    with open(output/'submission.csv',newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
                    self.assertEqual([r['Id'] for r in rows],[str(i) for i in range(1,21)])
                    self.assertTrue(np.array_equal(decode_rle(rows[0]['Predicted']),sub>0))
                with self.assertRaises(ValueError):
                    predict(SimpleNamespace(**{**common,'tta':False},output=str(Path(directory)/'bad'),csv_foreground='vessel'))
                (run/'fold0/best.pt').write_bytes(b'changed fixture')
                with self.assertRaises(ValueError):
                    predict(SimpleNamespace(**common,output=str(Path(directory)/'changed'),csv_foreground='vessel'))

    def test_cli_training_lock_creates_no_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)/'never_created'
            p=subprocess.run([sys.executable,str(ROOT/'main.py'),'train','--fold','0','--run-dir',str(run)],capture_output=True,text=True,cwd=ROOT)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('Training is locked',p.stderr)
            self.assertFalse(run.exists())


if __name__=='__main__': unittest.main()
