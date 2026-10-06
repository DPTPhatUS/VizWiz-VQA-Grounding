"""Answer-value distillation experiment plugin."""
import hashlib
from research.checkpoint import read_checkpoint, load_checkpoint
from research.model import ResearchGrounder
from research.distillation import AnswerStudent, DistillationObjective

EXPERIMENT = "answer-value-distillation"


def add_arguments(parser):
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--kd-mode", choices=["none", "ordinary", "confidence", "error", "incremental"], default="incremental")
    parser.add_argument("--kd-weight", type=float, default=1.)


def validate_args(args):
    if args.text_mode != "question" or args.architecture != "compact":
        raise ValueError("Distillation students require compact architecture and question-only text")
    if args.kd_weight < 0:
        raise ValueError("kd-weight cannot be negative")
    if args.kd_mode != "none" and not args.teacher_checkpoint:
        raise ValueError("Distillation requires --teacher-checkpoint (or --kd-mode none)")
    # run_config resume checks this hash, preventing silently changed teachers.
    if args.teacher_checkpoint:
        digest = hashlib.sha256()
        with open(args.teacher_checkpoint, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b""):
                digest.update(chunk)
        args.teacher_sha256 = digest.hexdigest()


def build_model(args):
    # Evaluation never opens the training-time teacher checkpoint.
    return AnswerStudent(args.tiny)


def build_objective(args, model, device):
    teacher = None
    if args.kd_mode != "none":
        saved = read_checkpoint(args.teacher_checkpoint)
        config = saved["experiment_config"]
        if config.get("experiment") != "controls" or config.get("tiny") != args.tiny:
            raise ValueError("Teacher must be a matching real/tiny corrected controls checkpoint")
        run = saved["run_config"]
        if args.kd_mode == "incremental" and (run.get("text_mode") != "dropout" or
                                             not 0 < run.get("answer_dropout", 0) < 1):
            raise ValueError("Incremental KD requires a teacher trained with nontrivial answer dropout")
        if run.get("image_size") != args.image_size:
            raise ValueError("Teacher/student image size must match")
        teacher = ResearchGrounder(config["architecture"], args.tiny).to(device)
        load_checkpoint(args.teacher_checkpoint, teacher)
    return DistillationObjective(teacher, args.kd_mode, args.kd_weight, args.dice_weight)
