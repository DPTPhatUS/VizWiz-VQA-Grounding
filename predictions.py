"""Lossless model-grid logits and their native-resolution reconstruction."""
import json
import os
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torch.nn import functional as F

from artifacts import write_json


def relative_logit_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '\\' in value or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError(f'Invalid prediction path: {value}')
    if path.suffix != '.npz':
        raise ValueError(f'Expected a compressed logit file: {value}')
    return path


class LogitWriter:
    def __init__(self, directory, split, checkpoint, image_size):
        self.directory = Path(directory)
        self.index = {'split': split,
                      'checkpoint': checkpoint, 'image_size': image_size,
                      'dtype': 'float32', 'resolution': 'model output',
                      'resize_mode': 'bilinear', 'align_corners': False, 'files': {}}

    def save(self, filename, logits, original_size):
        relative = relative_logit_path(str(PurePosixPath(filename).with_suffix('.npz')))
        path = self.directory/str(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, logits=logits.detach().float().cpu().numpy(),
                                original_size=np.asarray(original_size, dtype=np.int64))
        os.replace(temporary, path)
        self.index['files'][filename] = str(relative)

    def finish(self):
        write_json(self.directory/'index.json', self.index)


class SavedLogits:
    def __init__(self, split_directory):
        self.directory = Path(split_directory)/'logits'
        self.index = json.loads((self.directory/'index.json').read_text())

    def native(self, filename, device='cpu'):
        path = self.directory/str(relative_logit_path(self.index['files'][filename]))
        with np.load(path, allow_pickle=False) as saved:
            logits = torch.from_numpy(saved['logits']).to(device=device, dtype=torch.float32)
            size = tuple(int(v) for v in saved['original_size'])
        return F.interpolate(logits[None, None], size=size, mode='bilinear', align_corners=False)[0, 0]

    def export_masks(self, output, threshold=.5):
        import math
        from PIL import Image
        if not 0 < threshold < 1:
            raise ValueError('Probability threshold must be in (0, 1)')
        cutoff = math.log(threshold/(1-threshold))
        for filename in self.index['files']:
            path = Path(output)/Path(filename).with_suffix('.png')
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
    SavedLogits(args.predictions).export_masks(args.output_dir,args.threshold)


if __name__ == '__main__':
    main()
