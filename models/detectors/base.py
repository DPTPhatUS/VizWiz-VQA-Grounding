import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseDetector(nn.Module, ABC):
    """Abstract base for pluggable object detectors.
    
    All detectors must implement forward() and train_detector().
    freeze() and unfreeze() control whether the detector receives gradients.
    load_checkpoint() is concrete: loads weights and freezes by default.
    """
    
    @abstractmethod
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Detect objects and return bounding boxes.
        
        Args:
            image: (B, 3, H, W) float [0,1] tensor
            
        Returns:
            (B, N, 4) tensor [cx, cy, w, h] normalized [0,1]
        """
        ...
    
    @abstractmethod
    def train_detector(self, data_config: str, epochs: int):
        """Train the detector on a dataset.
        
        Args:
            data_config: Path to data config (e.g., YOLO data.yaml)
            epochs: Number of training epochs
        """
        ...
    
    @abstractmethod
    def freeze(self):
        """Freeze detector weights - no gradient updates."""
        ...
    
    @abstractmethod
    def unfreeze(self):
        """Unfreeze detector weights - enable gradient updates."""
        ...
    
    def load_checkpoint(self, path: str):
        """Load detector weights from checkpoint and freeze.
        
        Args:
            path: Path to .pt checkpoint file
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(ckpt, strict=False)
        self.freeze()
