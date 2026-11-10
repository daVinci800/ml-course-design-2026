import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json

import full_refit_v12 as refit


class AllTrainingDataIsolationTests(unittest.TestCase):
    def test_full_refit_uses_all_eighty_labels_and_excludes_test_images(self):
        records=[dict(id=i,split='train' if i<=80 else 'test') for i in range(1,101)]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            def prepare_folds(manifest,destination):
                (destination/'images').mkdir(parents=True)
            with patch.object(refit,'WORK',root),patch('fovea.detector_data.prepare_detector_folds',prepare_folds):
                path=refit.prepare(dict(records=records))
            settings=json.loads(path.read_text(encoding='utf8'))
            rows=(path.parent/settings['train']).read_text().splitlines()
            self.assertEqual(len(rows),80)
            self.assertEqual({int(Path(p).stem) for p in rows},set(range(1,81)))
            self.assertEqual(settings['train'],settings['val'])


if __name__=='__main__': unittest.main()
