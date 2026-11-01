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
from dataset import VizWizGroundingDataset, make_loader
from diagnostics import thresholds
from metrics import evaluate_counts, native_logits, overlap_counts
from models.factory import build_model
from predictions import LogitWriter, SavedLogits
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
                self.assertEqual(set(arrays.files), {'logits','original_size'})
            saved.export_masks(root/'exported')
            for name, logits, truth in native_logits(raw, batch):
                from PIL import Image
                with Image.open(root/'exported'/Path(name).with_suffix('.png')) as image:
                    np.testing.assert_array_equal(np.asarray(image)>0, saved.native(name).numpy()>0)
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
            save_checkpoint(checkpoint, {'epoch':17,
                'model_state_dict':model.state_dict(), 'experiment_config':model.experiment_config,
                'run_config':{'image_size':32}})
            for split in ('val','test'):
                evaluation.eval_main(['--checkpoint',str(checkpoint),'--data-root',str(data),
                    '--dataset',split,'--device','cpu','--num-workers','0',
                    '--batch-size','2','--output-dir',str(root/split)])
                result = json.loads((root/split/'metrics.json').read_text())
                self.assertEqual(result['summary']['checkpoint']['epoch'],17)
                self.assertNotIn('per_sample_metrics',result)
                self.assertNotIn('counts',result)
                self.assertNotIn('environment',result['summary'])
                self.assertNotIn('experiment_config',result['summary'])
                self.assertIn('iou',result['per_sample']['0.jpg'])
                self.assertIn('intersection',result['per_sample']['0.jpg'])
                self.assertFalse((root/split/'0.png').exists())
                self.assertTrue((root/split/result['run']).is_file())
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
            samples=json.loads((root/'test/metrics.json').read_text())['per_sample']
            expected={name:{key:row[key] for key in ('intersection','predicted','ground_truth','pixels')} for name,row in samples.items()}
            self.assertEqual(fixed['counts']['0.50'],expected)
            self.assertEqual(json.loads(selection.read_text())['split'],'val')



if __name__ == '__main__':
    unittest.main()
