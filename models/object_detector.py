import torch
import torch.nn as nn

class ObjectDetector(nn.Module):
    def __init__(self, model_name: str = "yolov8n", confidence: float = 0.25, topk: int = 3):
        super().__init__()
        from ultralytics import YOLO
        yolo = YOLO(model_name)
        self.model = yolo.model   # Underlying DetectionModel
        self._yolo = yolo         # High-level wrapper for predict/train/save
        self.confidence = confidence
        self.topk = topk

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        H, W = image.shape[2], image.shape[3]
        device = image.device

        all_bboxes = []
        for i in range(B):
            img_np = (image[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            results = self._yolo.predict(img_np, conf=self.confidence, verbose=False)

            bboxes = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xywh
                confs = results[0].boxes.conf
                sorted_idx = confs.argsort(descending=True)
                for idx in sorted_idx[:self.topk]:
                    cx, cy, bw, bh = boxes[idx].tolist()
                    bboxes.append([cx / W, cy / H, bw / W, bh / H])

            while len(bboxes) < self.topk:
                bboxes.append([0.0, 0.0, 0.0, 0.0])
            all_bboxes.append(bboxes)

        return torch.tensor(all_bboxes, device=device, dtype=torch.float32)

    def train_detector(self, data_config: str, epochs: int):
        self._yolo.train(data=data_config, epochs=epochs, verbose=True)
