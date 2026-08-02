import argparse
import json
import os
import sys

from models import ObjectDetector


def prepare_split(json_path, image_root, label_dir, image_dir):
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    processed = 0
    for filename, meta in data.items():
        vertices = meta.get("answer_grounding")
        if not vertices or "width" not in meta or "height" not in meta:
            continue

        xs = [v["x"] for v in vertices]
        ys = [v["y"] for v in vertices]
        cx = ((min(xs) + max(xs)) / 2) / meta["width"]
        cy = ((min(ys) + max(ys)) / 2) / meta["height"]
        bw = (max(xs) - min(xs)) / meta["width"]
        bh = (max(ys) - min(ys)) / meta["height"]

        src = os.path.join(image_root, filename)
        dst = os.path.join(image_dir, filename)
        if not os.path.exists(dst) and os.path.exists(src):
            os.symlink(os.path.abspath(src), dst)

        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt"))
        with open(label_path, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        processed += 1

    return processed


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
        # Setup directories
        yolo_dir = os.path.join(args.output_dir, "yolo_data")
        data_yaml = os.path.join(yolo_dir, "data.yaml")

        # Train split
        train_count = prepare_split(
            json_path=os.path.join(args.data_root, "train_grounding.json"),
            image_root=os.path.join(args.data_root, "train"),
            label_dir=os.path.join(yolo_dir, "labels", "train"),
            image_dir=os.path.join(yolo_dir, "images", "train"),
        )

        # Val split
        val_json = os.path.join(args.data_root, "val_grounding.json")
        if os.path.exists(val_json):
            val_count = prepare_split(
                json_path=val_json,
                image_root=os.path.join(args.data_root, "val"),
                label_dir=os.path.join(yolo_dir, "labels", "val"),
                image_dir=os.path.join(yolo_dir, "images", "val"),
            )
            val_line = "val: images/val\n"
        else:
            val_count = train_count
            val_line = "val: images/train\n"

        if train_count == 0:
            print("ERROR: No valid training entries found", file=sys.stderr)
            sys.exit(1)

        with open(data_yaml, "w") as f:
            f.write(f"path: {os.path.abspath(yolo_dir)}\n")
            f.write("train: images/train\n")
            f.write(val_line)
            f.write("nc: 1\n")
            f.write('names: ["grounded_object"]\n')

        print(f"Train: {train_count}, Val: {val_count}")
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
