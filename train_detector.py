import argparse
import json
import os
import sys

from models import ObjectDetector


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on VizWiz grounding bboxes")
    parser.add_argument("--data-root", type=str, default="data/vizwiz")
    parser.add_argument("--model-name", type=str, default="yolov8n")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Resume from checkpoint .pt")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    if args.resume_checkpoint:
        print(f"Resuming from {args.resume_checkpoint}...")
        detector = ObjectDetector(model_name=args.resume_checkpoint)
    else:
        # Load JSON
        json_path = os.path.join(args.data_root, "train_grounding.json")
        if not os.path.exists(json_path):
            print(f"ERROR: {json_path} not found", file=sys.stderr)
            sys.exit(1)

        print(f"Loading {json_path}...")
        with open(json_path) as f:
            data = json.load(f)

        # Setup directories
        yolo_dir = os.path.join(args.output_dir, "yolo_data")
        img_dir = os.path.join(yolo_dir, "images", "train")
        lbl_dir = os.path.join(yolo_dir, "labels", "train")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        # Write data.yaml
        data_yaml = os.path.join(yolo_dir, "data.yaml")
        with open(data_yaml, "w") as f:
            f.write(f"path: {os.path.abspath(yolo_dir)}\n")
            f.write("train: images/train\n")
            f.write("val: images/train\n")
            f.write("nc: 1\n")
            f.write('names: ["grounded_object"]\n')

        train_img_root = os.path.join(args.data_root, "train")
        skipped = 0
        processed = 0

        for filename, meta in data.items():
            vertices = meta.get("answer_grounding")
            if not vertices:
                print(f"Warning: No answer_grounding for {filename}, skipping")
                skipped += 1
                continue

            if "width" not in meta or "height" not in meta:
                print(f"Warning: Missing dimensions for {filename}, skipping")
                skipped += 1
                continue

            xs = [v["x"] for v in vertices]
            ys = [v["y"] for v in vertices]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            w_img = meta["width"]
            h_img = meta["height"]

            x_center = ((min_x + max_x) / 2) / w_img
            y_center = ((min_y + max_y) / 2) / h_img
            bbox_w = (max_x - min_x) / w_img
            bbox_h = (max_y - min_y) / h_img

            src_img = os.path.join(train_img_root, filename)
            dst_img = os.path.join(img_dir, filename)
            if not os.path.exists(dst_img):
                if os.path.exists(src_img):
                    os.symlink(os.path.abspath(src_img), dst_img)

            label_name = filename.replace(".jpg", ".txt").replace(".png", ".txt")
            label_path = os.path.join(lbl_dir, label_name)
            with open(label_path, "w") as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

            processed += 1

        print(f"Processed {processed} entries, skipped {skipped}")

        if processed == 0:
            print("ERROR: No valid training entries found", file=sys.stderr)
            sys.exit(1)

        detector = ObjectDetector(model_name=args.model_name)

    # Train
    print(f"Training YOLO ({args.model_name}) for {args.num_epochs} epochs...")
    detector.train_detector(
        data_config=data_yaml if not args.resume_checkpoint else os.path.join(args.output_dir, "yolo_data", "data.yaml"),
        epochs=args.num_epochs,
        resume=args.resume_checkpoint,
        save_every=args.save_every,
        output_dir=args.output_dir,
    )

    print(f"Checkpoints saved to {os.path.join(args.output_dir, 'detector', 'weights')}")


if __name__ == "__main__":
    main()
