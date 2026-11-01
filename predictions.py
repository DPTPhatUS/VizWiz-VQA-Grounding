"""Lossless model-grid logits and their native-resolution reconstruction."""
import json
import os
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torch.nn import functional as F

from artifacts import write_json

FORMAT_VERSION = 1


def relative_logit_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '\\' in value or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError(f'Invalid prediction path: {value}')
    if path.suffix != '.npz':
        raise ValueError(f'Expected a compressed logit file: {value}')
    return path


class LogitWriter:
    def __init__(self, directory, split, checkpoint_sha256, image_size):
        self.directory = Path(directory)
        self.index = {'format_version': FORMAT_VERSION, 'split': split,
                      'checkpoint_sha256': checkpoint_sha256, 'image_size': image_size,
                      'dtype': 'float32', 'resolution': 'model output',
                      'resize_mode': 'bilinear', 'align_corners': False, 'files': {}}

    def save(self, filename, logits, original_size):
        relative = relative_logit_path(str(PurePosixPath(filename).with_suffix('.npz')))
        path = self.directory/str(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, logits=logits.detach().float().cpu().numpy(),
                original_size=np.asarray(original_size, dtype=np.int64), filename=np.asarray(filename),
                checkpoint_sha256=np.asarray(self.index['checkpoint_sha256']),
                split=np.asarray(self.index['split']), format_version=np.asarray(FORMAT_VERSION))
        os.replace(temporary, path)
        self.index['files'][filename] = str(relative)

    def finish(self):
        write_json(self.directory/'index.json', self.index)


class SavedLogits:
    def __init__(self, split_directory):
        self.directory = Path(split_directory)/'logits'
        self.index = json.loads((self.directory/'index.json').read_text())
        if self.index['format_version'] != FORMAT_VERSION:
            raise ValueError('Unsupported saved-logit format')

    def native(self, filename, device='cpu'):
        path = self.directory/str(relative_logit_path(self.index['files'][filename]))
        with np.load(path, allow_pickle=False) as saved:
            if (saved['filename'].item() != filename
                    or saved['checkpoint_sha256'].item() != self.index['checkpoint_sha256']
                    or saved['split'].item() != self.index['split']):
                raise ValueError(f'Logit identity mismatch: {filename}')
            logits = torch.from_numpy(saved['logits']).to(device=device, dtype=torch.float32)
            size = tuple(int(v) for v in saved['original_size'])
        return F.interpolate(logits[None, None], size=size, mode='bilinear', align_corners=False)[0, 0]


def logit_artifacts(split_directory):
    """Paths relative to a split directory, restricted to its indexed predictions."""
    index = json.loads((Path(split_directory)/'logits/index.json').read_text())
    return ['logits/index.json', *[f'logits/{relative_logit_path(value)}' for value in index['files'].values()]]


def evaluation_complete(directory, checkpoint_sha256, split):
    """A final metrics file plus every indexed logit marks a reusable evaluation."""
    directory = Path(directory)
    metrics = directory/'metrics.json'
    if not metrics.exists():
        return False
    result = json.loads(metrics.read_text())
    if result['summary']['checkpoint_sha256'] != checkpoint_sha256:
        raise ValueError(f'Existing {split} evaluation is for a different checkpoint')
    if result.get('logits') != 'logits/index.json' or not (directory/'logits/index.json').exists():
        return False
    saved = SavedLogits(directory)
    if (saved.index['checkpoint_sha256'] != checkpoint_sha256 or saved.index['split'] != split
            or set(saved.index['files']) != set(result['per_sample'])):
        return False
    return all((directory/name).is_file() for name in logit_artifacts(directory))
