"""YOLO-based object detector for VizWiz grounding bboxes.

Wraps Ultralytics YOLO so it can be used as a drop-in ``nn.Module`` whose
``forward`` returns both boxes and confidence scores.

API
---
``forward(image)`` -> ``(boxes, scores)`` where

* ``boxes``  : ``(B, topk, 4)`` float tensor — ``[cx, cy, w, h]`` in **input
  pixel space, normalized to [0, 1] by the input image's ``(H, W)``.
* ``scores`` : ``(B, topk)``    float tensor — confidences in ``[0, 1]``.

When an image has fewer than ``topk`` detections, the remaining slots are
zero-padded so downstream code can rely on a fixed shape.
"""

import os

import torch
import torch.nn as nn

from ultralytics import YOLO


class ObjectDetector(nn.Module):
    def __init__(
        self,
        model_name: str = "yolov8n",
        confidence: float = 0.25,
        topk: int = 3,
        imgsz: int = 640,
        device: str = None,
    ):
        """Initialize the detector.

        Args:
            model_name: Path to a ``.pt`` file or a registered model name
                (e.g. ``"yolov8n"``).
            confidence: Confidence threshold passed to YOLO's NMS.
            topk: Number of top-scoring detections to keep per image.
            imgsz: Image size (square) used by YOLO's letterbox pipeline.
            device: ``"cuda"``, ``"cpu"``, or ``None`` (defer to the first
                input tensor's device on first ``forward`` call).
        """
        super().__init__()
        yolo = YOLO(model_name)
        # ``yolo.model`` is the underlying nn.Module and IS moved by .to().
        # ``yolo`` is a plain Python wrapper; its internal device pointer
        # is updated lazily on predict() based on yolo.model.device.
        self.model = yolo.model
        self._yolo = yolo
        self.confidence = confidence
        self.topk = topk
        self.imgsz = imgsz
        self._requested_device = device  # str or None

    # ------------------------------------------------------------------ #
    # Device handling                                                    #
    # ------------------------------------------------------------------ #
    def _resolve_device(self, image: torch.Tensor) -> torch.device:
        if self._requested_device is not None:
            return torch.device(self._requested_device)
        return image.device

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, image: torch.Tensor):
        """Run YOLO on a batch of images and return top-k boxes + scores.

        Args:
            image: ``(B, 3, H, W)`` float tensor with values in ``[0, 1]``.

        Returns:
            ``(boxes, scores)`` as described in the module docstring.
        """
        B, _, H, W = image.shape
        device = self._resolve_device(image)

        # Build a list of HxWx3 uint8 numpy arrays — YOLO's predict() will
        # batch them in a single CUDNN/cuBLAS call instead of being invoked
        # once per image from a Python loop.
        imgs_np = [
            (image[i].detach().permute(1, 2, 0).cpu().numpy() * 255.0)
            .clip(0, 255)
            .astype("uint8")
            for i in range(B)
        ]

        # ``device`` is forwarded so YOLO does not rely on stale state.
        results_list = self._yolo.predict(
            imgs_np,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=str(device),
            verbose=False,
        )

        all_boxes: list[list[list[float]]] = []
        all_scores: list[list[float]] = []
        for results in results_list:
            boxes: list[list[float]] = []
            scores: list[float] = []
            if results.boxes is not None and len(results.boxes) > 0:
                xywh = results.boxes.xywh
                confs = results.boxes.conf
                sorted_idx = confs.argsort(descending=True)
                for idx in sorted_idx[: self.topk]:
                    cx, cy, bw, bh = xywh[idx].tolist()
                    boxes.append([cx / W, cy / H, bw / W, bh / H])
                    scores.append(float(confs[idx].item()))
            # Pad to topk
            while len(boxes) < self.topk:
                boxes.append([0.0, 0.0, 0.0, 0.0])
                scores.append(0.0)
            all_boxes.append(boxes)
            all_scores.append(scores)

        boxes_t = torch.tensor(all_boxes, device=device, dtype=torch.float32)
        scores_t = torch.tensor(all_scores, device=device, dtype=torch.float32)
        return boxes_t, scores_t

    # ------------------------------------------------------------------ #
    # Training                                                           #
    # ------------------------------------------------------------------ #
    def train_detector(
        self,
        data_config: str,
        epochs: int,
        resume: str = None,
        save_every: int = 10,
        output_dir: str = "outputs",
        imgsz: int = None,
        device: str = None,
        batch: int = None,
        single_cls: bool = True,
    ):
        """Train (or resume training of) the underlying YOLO model.

        Args:
            data_config: Path to YOLO ``data.yaml``.
            epochs: Total number of training epochs.
            resume: Path to a checkpoint ``.pt`` to resume from. When set,
                a fresh ``YOLO`` instance is loaded from this path.
            save_every: Save a checkpoint every N epochs.
            output_dir: Ultralytics ``project`` directory.
            imgsz: Override training image size (defaults to ``self.imgsz``).
            device: ``"cuda"``, ``"cpu"``, or ``None`` (YOLO default).
            batch: Per-rank batch size or ``None`` for YOLO auto.
            single_cls: Treat all classes as a single class. Strongly
                recommended for our 1-class grounding setup.
        """
        from ultralytics import YOLO  # local import keeps top of file light

        if resume:
            # Re-instantiate from checkpoint so the underlying model
            # weights are loaded correctly.
            self._yolo = YOLO(resume)
            self.model = self._yolo.model

        train_kwargs = dict(
            data=data_config,
            epochs=epochs,
            verbose=True,
            save_period=save_every,
            resume=resume is not None,
            project=os.path.abspath(output_dir),
            name="detector",
            exist_ok=True,
            single_cls=single_cls,
        )
        train_kwargs["imgsz"] = imgsz if imgsz is not None else self.imgsz
        if device is not None:
            train_kwargs["device"] = device
        if batch is not None:
            train_kwargs["batch"] = batch

        self._yolo.train(**train_kwargs)
