"""Evaluate a trained YOLO detector on VizWiz val grounding bboxes.

Computes mAP@0.5 using axis-aligned polygon bboxes as ground truth.

Usage:
    uv run eval_detector.py --checkpoint outputs/yolo_pretrained.pt
"""

import argparse
import json
import os

import torch
from tqdm import tqdm


def polygon_to_bbox(vertices, width, height):
    """Convert polygon vertices to normalized [cx, cy, w, h]."""
    xs = [v["x"] for v in vertices]
    ys = [v["y"] for v in vertices]
    cx = ((min(xs) + max(xs)) / 2) / width
    cy = ((min(ys) + max(ys)) / 2) / height
    bw = (max(xs) - min(xs)) / width
    bh = (max(ys) - min(ys)) / height
    return cx, cy, bw, bh


def compute_iou(box1, box2):
    """Compute IoU between two [cx, cy, w, h] boxes (normalized)."""
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
    """Compute Average Precision from precision-recall curve (11-point interpolation)."""
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        p = precision[recall >= t].max() if (recall >= t).any() else 0.0
        ap += p / 11.0
    return ap


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO detector on VizWiz val bboxes")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained YOLO .pt file")
    parser.add_argument("--data-root", type=str, default="data/vizwiz")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for mAP")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load val data
    val_json = os.path.join(args.data_root, "val_grounding.json")
    with open(val_json) as f:
        val_data = json.load(f)

    image_dir = os.path.join(args.data_root, "val")

    # Load YOLO
    from ultralytics import YOLO
    model = YOLO(args.checkpoint)
    model.to(args.device)

    all_results = []  # list of (gt_bbox, pred_bboxes) tuples

    for filename, meta in tqdm(val_data.items(), desc="Evaluating"):
        vertices = meta.get("answer_grounding")
        if not vertices or "width" not in meta:
            continue

        # Ground truth bbox
        gt = polygon_to_bbox(vertices, meta["width"], meta["height"])

        # Run YOLO
        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            continue

        results = model.predict(img_path, conf=args.confidence, verbose=False)
        preds = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xywhn  # [M, 4] normalized [cx, cy, w, h]
            confs = results[0].boxes.conf
            for box, conf in zip(boxes, confs):
                preds.append((box.tolist(), conf.item()))

        all_results.append((gt, preds))

    # Compute mAP
    tp = []
    fp = []
    scores = []
    n_gt = len(all_results)

    for gt, preds in all_results:
        matched = False
        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
        for box, score in preds_sorted:
            iou = compute_iou(gt, box)
            if iou >= args.iou_threshold and not matched:
                tp.append(1)
                fp.append(0)
                matched = True
            else:
                tp.append(0)
                fp.append(1)
            scores.append(score)

    if not tp:
        print("No detections found.")
        return

    # Sort by score descending
    indices = torch.tensor(scores).argsort(descending=True)
    tp = torch.tensor(tp)[indices].cumsum(0)
    fp = torch.tensor(fp)[indices].cumsum(0)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / n_gt
    ap = compute_ap(precision, recall)

    print(f"mAP@{args.iou_threshold}: {ap:.4f}")
    print(f"Evaluated {n_gt} samples")


if __name__ == "__main__":
    main()
