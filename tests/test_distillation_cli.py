"""CPU integration: strict teacher contract, resume, teacher-free student eval."""
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import torch
from research.checkpoint import save_checkpoint, read_checkpoint
from research.engine import train_main, eval_main, training_parser, check_arguments
from research.model import ResearchGrounder
from research import variant


class DistillationIntegrationTests(unittest.TestCase):
    def fixture(self, root):
        for split in ("train", "val"):
            (root/split).mkdir()
            (root/"binary_masks_png"/split).mkdir(parents=True)
            Image.new("RGB", (17,11), "red").save(root/split/"a.jpg")
            Image.new("L", (17,11), 255).save(root/"binary_masks_png"/split/"a.png")
            (root/f"{split}_grounding.json").write_text(json.dumps({
                "a.jpg": {"question": "what?", "most_common_answer": "red"}}))
        teacher = root/"teacher.pt"
        save_checkpoint(teacher, ResearchGrounder(tiny=True), None, None, 1, 0.,
                        {"text_mode": "dropout", "answer_dropout": .5, "image_size": 28})
        return teacher

    def test_training_resume_and_evaluation_without_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher = self.fixture(root)
            out = root/"student"
            args = ["--tiny", "--device", "cpu", "--image-size", "28", "--num-workers", "0",
                    "--batch-size", "1", "--data-root", str(root), "--output-dir", str(out),
                    "--teacher-checkpoint", str(teacher)]
            train_main(args + ["--num-epochs", "1"])
            train_main(args + ["--num-epochs", "2", "--resume-checkpoint", str(out/"last.pt")])
            checkpoint = read_checkpoint(out/"last.pt")
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertIn("teacher_sha256", checkpoint["run_config"])
            teacher.unlink()
            eval_main(["--checkpoint", str(out/"best.pt"), "--data-root", str(root),
                       "--device", "cpu", "--num-workers", "0", "--output-dir", str(root/"pred")])
            with Image.open(root/"pred/a.png") as prediction:
                self.assertEqual(prediction.size, (17,11))

    def test_incremental_rejects_teacher_without_answer_dropout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher = self.fixture(root)
            saved = read_checkpoint(teacher)
            saved["run_config"]["text_mode"] = "answer"
            torch.save(saved, teacher)
            args = training_parser().parse_args(["--tiny", "--image-size", "28",
                "--teacher-checkpoint", str(teacher)])
            check_arguments(args)
            with self.assertRaisesRegex(ValueError, "answer dropout"):
                variant.build_objective(args, variant.build_model(args), torch.device("cpu"))
