"""Offline integration fixtures are synthetic annotations, never research training data."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from PIL import Image
from research.engine import train_main, eval_main


def fixture(root):
    for split in ('train', 'val'):
        (root/split).mkdir(); (root/'binary_masks_png'/split).mkdir(parents=True)
        records = {}
        for i, size in enumerate(((17, 11), (11, 19), (13, 15))):
            Image.new('RGB', size, 'red').save(root/split/f'{i}.jpg')
            Image.new('L', size, 255 if i else 0).save(root/'binary_masks_png'/split/f'{i}.png')
            records[f'{i}.jpg'] = {'question': 'where?', 'most_common_answer': 'secret'}
        (root/f'{split}_grounding.json').write_text(json.dumps(records))
    Image.new('L', (17, 11), 255).save(root/'changed.png')
    pairs = [dict(filename='0.jpg', question1='nothing?', question2='whole?',
                  mask1='binary_masks_png/train/0.png', mask2='changed.png', relation='change'),
             dict(filename='1.jpg', question1='where?', question2='which region?',
                  mask1='binary_masks_png/train/1.png', mask2='binary_masks_png/train/1.png', relation='same')]
    (root/'pairs.json').write_text(json.dumps(pairs))


class ExtentCLITests(unittest.TestCase):
    def test_unpaired_and_paired_training_resume_and_question_only_eval(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root)
            common = ['--tiny', '--image-size', '28', '--data-root', str(root), '--device', 'cpu',
                      '--num-workers', '0', '--batch-size', '2', '--lr', '.001']
            with contextlib.redirect_stdout(io.StringIO()):
                train_main(common+['--output-dir', str(root/'unpaired'), '--num-epochs', '1'])
            out = root/'paired'
            paired = common+['--output-dir', str(out), '--pairs', str(root/'pairs.json'),
                             '--pair-delta-weight', '.5', '--pair-consistency-weight', '.5']
            process = subprocess.run([sys.executable, 'train_research.py', *paired, '--num-epochs', '1'],
                capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS='1', MPLCONFIGDIR='/tmp/vizwiz-mpl'))
            self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
            first = torch.load(out/'last.pt', weights_only=False)
            self.assertGreater(json.loads((out/'history.jsonl').read_text())['pair_delta'], 0)
            with contextlib.redirect_stdout(io.StringIO()):
                train_main(paired+['--num-epochs', '2', '--resume-checkpoint', str(out/'last.pt')])
            last = torch.load(out/'last.pt', weights_only=False)
            self.assertEqual(last['epoch'], 2)
            self.assertFalse(torch.equal(first['model_state_dict']['query_tokens'], last['model_state_dict']['query_tokens']))
            self.assertIn('pairs', last['provenance']['annotation_sha256'])
            with self.assertRaisesRegex(ValueError, 'configuration'), contextlib.redirect_stdout(io.StringIO()):
                train_main(paired+['--num-epochs', '3', '--resume-checkpoint', str(out/'last.pt'), '--area-weight', '.7'])
            # Evaluation must not reopen training-only pair artifacts.
            (root/'pairs.json').unlink(); (root/'changed.png').unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                eval_main(['--checkpoint', str(out/'best.pt'), '--data-root', str(root),
                           '--device', 'cpu', '--num-workers', '0', '--output-dir', str(root/'pred')])
            metrics = json.loads((root/'pred/metrics.json').read_text())['summary']
            self.assertEqual(metrics['text_mode'], 'question')
            self.assertEqual(metrics['num_samples'], 3)
            with Image.open(root/'pred/0.png') as prediction:
                self.assertEqual(prediction.size, (17, 11))

if __name__ == '__main__':
    unittest.main()
