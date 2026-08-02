import torch
import torch.nn as nn
from .base import BaseDetector

class YOLODetector(BaseDetector):
    """YOLO object detector via ultralytics.
    
    Wraps ultralytics.YOLO with tensor-based forward pass
    and freeze/unfreeze control.
    """
    
    def __init__(self, model_name: str = "yolov8n", confidence: float = 0.25, topk: int = 3):
        super().__init__()
        from ultralytics import YOLO
        yolo = YOLO(model_name)
        self.model = yolo.model  # Underlying DetectionModel (not the high-level YOLO wrapper)
        self._yolo = yolo        # Keep wrapper ref for train_detector + save
        self.confidence = confidence
        self.topk = topk

    def train(self, mode: bool = True):
        """Override nn.Module.train to skip recursing into self.model.

        The underlying DetectionModel doesn't need training mode toggling
        because freeze/unfreeze already controls requires_grad.
        """
        self.training = mode
        return self
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Run YOLO inference, return top-k bounding boxes.
        
        Args:
            image: (B, 3, H, W) float [0,1] tensor
            
        Returns:
            (B, topk, 4) [cx, cy, w, h] normalized [0,1]
        """
        B = image.shape[0]
        H, W = image.shape[2], image.shape[3]
        device = image.device
        
        all_bboxes = []
        for i in range(B):
            # Convert [0,1] float tensor to numpy array for YOLO
            img_np = (image[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            
            # Run YOLO inference
            results = self._yolo.predict(img_np, conf=self.confidence, verbose=False)
            
            bboxes = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xywh  # [M, 4] in pixel coords [cx, cy, w, h]
                confs = results[0].boxes.conf    # [M]
                
                # Sort by confidence descending, take topk
                sorted_idx = confs.argsort(descending=True)
                for idx in sorted_idx[:self.topk]:
                    cx, cy, bw, bh = boxes[idx].tolist()
                    # Normalize to [0,1]
                    bboxes.append([cx / W, cy / H, bw / W, bh / H])
            
            # Pad to topk with zero bboxes
            while len(bboxes) < self.topk:
                bboxes.append([0.0, 0.0, 0.0, 0.0])
            
            all_bboxes.append(bboxes)
        
        return torch.tensor(all_bboxes, device=device, dtype=torch.float32)
    
    def freeze(self):
        """Freeze all YOLO parameters and set to eval mode."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def unfreeze(self):
        """Unfreeze all YOLO parameters and set to train mode."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()
    
    def train_detector(self, data_config: str, epochs: int):
        """Train YOLO on a data.yaml config.
        
        Args:
            data_config: Path to YOLO-format data.yaml
            epochs: Training epochs
        """
        self._yolo.train(data=data_config, epochs=epochs, verbose=True)
    
    def load_checkpoint(self, path: str):
        """Load YOLO weights from a .pt file (YOLO-format checkpoint).
        
        Loads the checkpoint's underlying DetectionModel, preserving
        the checkpoint's class count (nc) and architecture.
        """
        from ultralytics import YOLO
        loaded = YOLO(path)
        self.model = loaded.model   # Replace DetectionModel with checkpoint version
        self._yolo = loaded          # Update wrapper for predict/train/save
        self.freeze()
        del loaded
