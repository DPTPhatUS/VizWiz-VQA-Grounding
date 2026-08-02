"""Evaluate ObjectDetector on VizWiz val grounding bboxes.

Computes COCO-style mAP@0.5 using axis-aligned polygon bboxes as ground
truth. The detector must be loaded with a trained ``.pt`` checkpoint; a
freshly initialized YOLO produces random detections and the reported mAP
will be near 0.

Usage:
    uv run eval_detector.py --checkpoint outputs/detector/weights/best.pt
"""

import argparse
import json
import os

import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from models import ObjectDetector


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def polygon_to_bbox(vertices, width, height):
    """Tight axis-aligned bbox around the polygon, in YOLO-normalized form."""
    xs = [v["x"] for v in vertices]
    ys = [v["y"] for v in vertices]
    cx = ((min(xs) + max(xs)) / 2) / width
    cy = ((min(ys) + max(ys)) / 2) / height
    bw = (max(xs) - min(xs)) / width
    bh = (max(ys) - min(ys)) / height
    return cx, cy, bw, bh


def compute_iou(box1, box2):
    """Axis-aligned IoU between two [cx, cy, w, h] boxes in [0, 1]."""
    x1_1, y1_1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x2_1, y2_1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x1_2, y1_2 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_2, y2_2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    ix1, iy1 = max(x1_1, x1_2), max(y1_1, y1_2)
    ix2, iy2 = min(x2_1, x2_2), min(y2_1, y2_2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)

    inter = iw * ih
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_ap(precision, recall):
    """All-points AP (Pascal VOC post-2010 / sklearn style).

    ``precision`` and ``recall`` are 1-D tensors of equal length, sorted by
    descending score outside this function.  We pad both ends with
    sentinels and compute the area under the precision-envelope curve.
    """
    if precision.numel() == 0:
        return 0.0
    mrec = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
    mpre = torch.cat((torch.tensor([0.0]), precision, torch.tensor([0.0])))
    # Make the precision curve monotonically decreasing from the right.
    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
    # Sum areas where recall changes.
    i = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=True)[0]
    return float(((mrec[i + 1] - mrec[i]) @ mpre[i + 1]).item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ObjectDetector on VizWiz val bboxes"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/vizwiz")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    # ---- Data ----
    val_json = os.path.join(args.data_root, "val_grounding.json")
    with open(val_json) as f:
        val_data = json.load(f)
    image_dir = os.path.join(args.data_root, "val")

    # ---- Model ----
    detector = ObjectDetector(
        model_name=args.checkpoint,
        confidence=args.confidence,
        imgsz=args.image_size,
        device=args.device,
    )
    detector.to(args.device)
    detector.eval()

    transform = T.Compose([T.ToTensor()])

    all_results: list[tuple] = []  # (gt, [(box, score), ...])
    for filename, meta in tqdm(val_data.items(), desc="Evaluating"):
        vertices = meta.get("answer_grounding")
        # BOTH width and height are required (eval previously only checked width).
        if not vertices or "width" not in meta or "height" not in meta:
            continue

        gt = polygon_to_bbox(vertices, meta["width"], meta["height"])

        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            continue

        image = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(args.device)
        with torch.no_grad():
            pred_boxes, pred_scores = detector(image)
        # pred_boxes : (1, topk, 4)  [cx, cy, w, h] normalized
        # pred_scores: (1, topk)

        # Drop the zero-padded slots introduced by ObjectDetector for images
        # with fewer than `topk` real detections.  score == 0 is a reliable
        # marker because real confidences are strictly > 0.
        preds = []
        for box, score in zip(pred_boxes[0].tolist(), pred_scores[0].tolist()):
            if score > 0:
                preds.append((box, score))

        all_results.append((gt, preds))

    # ---- mAP computation (single class) ----
    tp: list[int] = []
    fp: list[int] = []
    scores: list[float] = []
    n_gt = len(all_results)

    for gt, preds in all_results:
        # Sort by score desc so AP behaves like a PR curve, not a per-image
        # rank-1 readout.
        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
        matched = False
        for box, score in preds_sorted:
            iou = compute_iou(gt, box)
            if iou >= args.iou_threshold and not matched:
                tp.append(1)
                fp.append(0)
                matched = True  # one GT per image — only the first match counts
            else:
                tp.append(0)
                fp.append(1)
            scores.append(score)

    if not tp:
        print("No detections above the confidence threshold — mAP is 0.")
        print(f"Evaluated {n_gt} ground-truth samples.")
        return

    indices = torch.tensor(scores).argsort(descending=True)
    tp = torch.tensor(tp, dtype=torch.float32)[indices].cumsum(0)
    fp = torch.tensor(fp, dtype=torch.float32)[indices].cumsum(0)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / max(n_gt, 1)
    ap = compute_ap(precision, recall)

    print(f"mAP@{args.iou_threshold}: {ap:.4f}")
    print(f"Evaluated {n_gt} samples")


if __name__ == "__main__":
    main()
