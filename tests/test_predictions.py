"""Lossless logits, native-grid reconstruction, and calibration without inference."""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import eval as evaluation
from checkpoints import save_checkpoint
from dataset import VizWizGroundingDataset, collate_samples, make_loader
from diagnostics import thresholds
from metrics import evaluate_counts, native_logits, overlap_counts
from models.factory import build_model
from predictions import LogitWriter, SavedLogits, evaluation_complete
from tests.support import fixture_data


class SavedPredictions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_lossless_logits_reproduce_native_counts_at_every_threshold(self):
        raw = torch.tensor([[[[-2.1, -.001], [.0003, 3.14]]],
                            [[[1.75, -3.1], [.125, -.875]]]], dtype=torch.float32)
        batch = {'filename':['a.jpg','nested/b.jpg'], 'original_mask':[
                 torch.ones(1,5,7), (torch.arange(24).reshape(1,4,6)%3==0).float()]}
        class Fixed(nn.Module):
            def forward(self, batch):
                return {'logits':raw}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = LogitWriter(root/'logits', 'val', 'test-sha', 336)
            counts = evaluate_counts(Fixed(), [batch], torch.device('cpu'), root, logit_writer=writer)
            writer.finish()
            saved = SavedLogits(root)
            with np.load(root/'logits/a.npz', allow_pickle=False) as arrays:
                self.assertEqual(arrays['logits'].dtype, np.float32)
                np.testing.assert_array_equal(arrays['logits'], raw[0,0].numpy())
                np.testing.assert_array_equal(arrays['original_size'], [5,7])
                self.assertEqual(arrays['filename'].item(), 'a.jpg')
            for name, logits, truth in native_logits(raw, batch):
                restored = saved.native(name)
                torch.testing.assert_close(restored, logits, rtol=0, atol=0)
                for p in thresholds.THRESHOLDS:
                    cutoff = math.log(p/(1-p))
                    self.assertEqual(overlap_counts(restored>cutoff, truth),
                                     overlap_counts(logits>cutoff, truth))
                self.assertEqual(counts[name], overlap_counts(restored>0, truth))

    def test_saved_calibration_matches_live_inference_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data = fixture_data(root/'data')
            torch.manual_seed(42)
            model = build_model('ftc', tiny=True).eval()
            checkpoint = root/'best.pt'
            save_checkpoint(checkpoint, {'format_version':1, 'epoch':17,
                'model_state_dict':model.state_dict(), 'experiment_config':model.experiment_config,
                'run_config':{'image_size':32}})
            for split in ('val','test'):
                evaluation.eval_main(['--checkpoint',str(checkpoint),'--data-root',str(data),
                    '--dataset',split,'--device','cpu','--num-workers','0',
                    '--batch-size','2','--output-dir',str(root/split)])
                result = json.loads((root/split/'metrics.json').read_text())
                self.assertEqual(result['summary']['checkpoint_epoch'],17)
                self.assertEqual(set(result['per_sample_metrics']),set(result['per_sample']))
                self.assertTrue(evaluation_complete(root/split,result['summary']['checkpoint_sha256'],split))
            dataset = VizWizGroundingDataset(data,'val',32)
            args = SimpleNamespace(num_workers=0,device='cpu')
            live = thresholds.sweep(model,make_loader(dataset,args,2),torch.device('cpu'),thresholds.THRESHOLDS)
            cached = thresholds.sweep_saved(SavedLogits(root/'val'),dataset,'cpu',thresholds.THRESHOLDS)
            self.assertEqual(live,cached)
            with patch.object(thresholds,'load_inference',side_effect=AssertionError('must not load weights')):
                common = ['--data-root',str(data),'--device','cpu']
                thresholds.main(['validation','--predictions',str(root/'val'),*common,
                                 '--output-dir',str(root/'calibration')])
                selection = root/'calibration/selection.json'
                thresholds.main(['test','--predictions',str(root/'test'),*common,
                                 '--selection',str(selection),'--output-dir',str(root/'calibrated-test')])
            report = json.loads((root/'calibration/results.json').read_text())
            self.assertEqual(report['counts'],live)
            fixed = json.loads((root/'calibrated-test/results.json').read_text())
            self.assertEqual(fixed['counts']['0.50'],json.loads((root/'test/metrics.json').read_text())['counts'])
            self.assertEqual(json.loads(selection.read_text())['split'],'val')

    def test_partial_export_is_not_marked_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = LogitWriter(root/'logits','val','sha',336)
            writer.save('a.jpg',torch.zeros(2,2),(5,7))
            writer.finish()
            metrics = {'summary':{'checkpoint_sha256':'sha'}, 'per_sample':{'a.jpg':0.},
                       'logits':'logits/index.json'}
            (root/'metrics.json').write_text(json.dumps(metrics))
            self.assertTrue(evaluation_complete(root,'sha','val'))
            (root/'logits/a.npz').unlink()
            self.assertFalse(evaluation_complete(root,'sha','val'))
            with self.assertRaisesRegex(ValueError,'different checkpoint'):
                evaluation_complete(root,'other-sha','val')


if __name__ == '__main__':
    unittest.main()
