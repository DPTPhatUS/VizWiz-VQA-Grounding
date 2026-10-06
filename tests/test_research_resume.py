import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from research.engine import provenance, training_parser, check_arguments


class ResumeIntegrityTests(unittest.TestCase):
    def test_annotation_provenance_detects_mask_content_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'binary_masks_png/train').mkdir(parents=True)
            mask = root/'binary_masks_png/train/a.png'
            Image.new('L', (4,4), 0).save(mask)
            (root/'train_grounding.json').write_text(json.dumps({'a.jpg': {'question': 'x'}}))
            args = SimpleNamespace(data_root=str(root), pairs=None)
            first = provenance(args)['annotation_sha256']
            Image.new('L', (4,4), 255).save(mask)
            self.assertNotEqual(first, provenance(args)['annotation_sha256'])

    def test_resume_rejects_new_directory_that_would_lose_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root/'old/last.pt'
            state.parent.mkdir()
            state.touch()
            args = training_parser().parse_args(['--resume-checkpoint', str(state),
                '--output-dir', str(root/'new')])
            with self.assertRaisesRegex(ValueError, 'same output directory'):
                check_arguments(args)

    def test_resume_rejects_older_checkpoint_with_stale_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root/'checkpoint_epoch1.pt'
            state.touch()
            args = training_parser().parse_args(['--resume-checkpoint', str(state),
                '--output-dir', str(root)])
            with self.assertRaisesRegex(ValueError, 'last.pt'):
                check_arguments(args)
