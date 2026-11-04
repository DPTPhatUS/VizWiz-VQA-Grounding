"""Lossless model-grid logits and their native-resolution reconstruction."""
import json
import os
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch
from torch.nn import functional as F

def relative_image_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '\\' in value or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError(f'Invalid prediction path: {value}')
    return path


class LogitWriter:
    """Stream separate image arrays into one compressed NPZ per split."""
    def __init__(self, path, split, checkpoint, image_size):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_suffix('.npz.tmp')
        self.archive = ZipFile(self.temporary, 'w', compression=ZIP_DEFLATED)
        self.index = {'split': split,
                      'checkpoint': checkpoint, 'image_size': image_size,
                      'dtype': 'float32', 'resolution': 'model output',
                      'resize_mode': 'bilinear', 'align_corners': False, 'files': {}}

    def save(self, filename, logits, original_size):
        filename = str(relative_image_path(filename))
        if filename in self.index['files']:
            raise ValueError(f'Duplicate prediction filename: {filename}')
        key = f'logits_{len(self.index["files"]):06d}'
        self._write_array(key, logits.detach().float().cpu().numpy())
        self.index['files'][filename] = {'array': key,
                                       'original_size': [int(v) for v in original_size]}

    def _write_array(self, key, array):
        with self.archive.open(f'{key}.npy', 'w', force_zip64=True) as stream:
            np.lib.format.write_array(stream, array, allow_pickle=False)

    def finish(self):
        self._write_array('metadata', np.asarray(json.dumps(self.index)))
        self.archive.close()
        os.replace(self.temporary, self.path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                self.finish()
        finally:
            self.archive.close()
            self.temporary.unlink(missing_ok=True)


class SavedLogits:
    def __init__(self, split_directory):
        self.arrays = np.load(Path(split_directory)/'logits.npz', allow_pickle=False)
        try:
            self.index = json.loads(self.arrays['metadata'].item())
        except Exception:
            self.close()
            raise

    def native(self, filename, device='cpu'):
        entry = self.index['files'][filename]
        logits = torch.from_numpy(self.arrays[entry['array']]).to(device=device, dtype=torch.float32)
        size = tuple(entry['original_size'])
        return F.interpolate(logits[None, None], size=size, mode='bilinear', align_corners=False)[0, 0]

    def close(self):
        self.arrays.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def export_masks(self, output, threshold=.5):
        import math
        from PIL import Image
        if not 0 < threshold < 1:
            raise ValueError('Probability threshold must be in (0, 1)')
        cutoff = math.log(threshold/(1-threshold))
        for filename in self.index['files']:
            path = Path(output)/str(relative_image_path(filename).with_suffix('.png'))
            path.parent.mkdir(parents=True, exist_ok=True)
            mask = (self.native(filename) > cutoff).numpy().astype(np.uint8)*255
            Image.fromarray(mask).save(path)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Export binary masks from saved logits without model inference')
    parser.add_argument('--predictions',required=True,help='Saved val/ or test/ directory')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--threshold',type=float,default=.5)
    args = parser.parse_args(argv)
    with SavedLogits(args.predictions) as saved:
        saved.export_masks(args.output_dir,args.threshold)


if __name__ == '__main__':
    main()
