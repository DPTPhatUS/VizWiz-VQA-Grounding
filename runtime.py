"""Device transfer, parameter accounting, and reproducible source/environment records."""
import importlib.metadata
import platform
import subprocess
import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "high_priority.json"


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def environment():
    versions = {}
    for name in ('torch', 'torchvision', 'transformers', 'huggingface-hub', 'numpy', 'Pillow', 'matplotlib'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    root = PROJECT_ROOT
    def git(*args):
        result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {'python': sys.version, 'platform': platform.platform(), 'versions': versions,
            'git_commit': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
            'cuda_version': torch.version.cuda}


def parameter_counts(model):
    count = lambda module, trainable=False: sum(p.numel() for p in module.parameters() if not trainable or p.requires_grad)
    return {'total': count(model), 'trainable': count(model, True),
            'decoder_registered': count(model.decoder), 'decoder_trainable': count(model.decoder, True)}
