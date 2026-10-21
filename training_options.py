"""Optional training controls shared by experiment branches; legacy defaults stay intact."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

DEFAULTS = {
    "optimizer": "adam",
    "weight_decay": 0.0,
    "backbone_lr": None,
    "lr_scheduler": "none",
    "warmup_ratio": 0.0,
    "min_lr_ratio": 0.1,
    "poly_power": 0.9,
    "freeze_vision_encoder": None,
    "freeze_text_encoder": None,
}
CONTROL_KEYS = {*DEFAULTS, "lr", "freeze_backbone"}


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1", "yes"}:
        return True
    if str(value).lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Use True or False")


class TrainingArgumentParser(argparse.ArgumentParser):
    """Resolve a flat JSON over explicit CLI options, retaining branch-specific defaults."""

    def __init__(self, *args, recipe_keys=(), **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)
        self.recipe_keys = set(recipe_keys)

    def parse_args(self, args=None, namespace=None):
        if "--optimizer" not in self._option_string_actions:
            if "--lr" not in self._option_string_actions:
                self.add_argument("--lr", type=float, default=1e-5)
            self.add_argument("--optimizer", choices=("adam", "adamw"), default=None)
            self.add_argument("--weight-decay", type=float, default=None)
            self.add_argument(
                "--backbone-lr",
                type=float,
                default=None,
                help="Optional LR for unfrozen vision/text encoders; otherwise --lr",
            )
            self.add_argument(
                "--lr-scheduler",
                choices=("none", "cosine", "linear", "poly"),
                default=None,
            )
            self.add_argument("--warmup-ratio", type=float, default=None)
            self.add_argument("--min-lr-ratio", type=float, default=None)
            self.add_argument("--poly-power", type=float, default=None)
            for name in ("backbone", "vision-encoder", "text-encoder"):
                self.add_argument(
                    f"--freeze-{name}",
                    type=boolean,
                    nargs="?",
                    const=True,
                    default=None,
                )
                self.add_argument(
                    f"--no-freeze-{name}",
                    dest=f"freeze_{name.replace('-', '_')}",
                    action="store_false",
                    default=None,
                )
            if "--config" not in self._option_string_actions:
                self.add_argument(
                    "--config",
                    help="JSON overrides CLI options; omitted keys keep branch defaults",
                )
        raw = list(sys.argv[1:] if args is None else args)
        parsed = super().parse_args(raw, namespace)
        explicit = {
            self._option_string_actions[token.split("=", 1)[0]].dest
            for token in raw
            if token.split("=", 1)[0] in self._option_string_actions
        }
        requested = {}
        if parsed.config:
            requested = json.loads(Path(parsed.config).read_text(encoding="utf-8-sig"))
            if not isinstance(requested, dict):
                self.error("Config must be a JSON object")
        actions = {a.dest: a for a in self._actions}
        unknown = requested.keys() - actions.keys() - self.recipe_keys
        if unknown or "config" in requested:
            self.error(
                f"Unknown or recursive config keys: {sorted(unknown | ({'config'} & requested.keys()))}"
            )
        for key, value in requested.items():
            if key not in actions:
                continue
            action = actions[key]
            if isinstance(
                action,
                (
                    argparse.BooleanOptionalAction,
                    argparse._StoreTrueAction,
                    argparse._StoreFalseAction,
                ),
            ) or key.startswith("freeze_"):
                valid = type(value) is bool
            elif action.nargs in ("*", "+"):
                valid = isinstance(value, list) and all(
                    type(x) is (action.type or str) for x in value
                )
            elif action.type in (int, float):
                valid = (
                    type(value) is int
                    if action.type is int
                    else type(value) in (int, float)
                )
            else:
                valid = isinstance(value, str)
            if not valid or (
                action.choices is not None and value not in action.choices
            ):
                self.error(f"Invalid config value for {key}: {value!r}")
            setattr(parsed, key, value)
        # A config shorthand outranks even an explicit component CLI switch.
        for component in ("freeze_vision_encoder", "freeze_text_encoder"):
            if component in requested:
                value = requested[component]
            elif "freeze_backbone" in requested:
                value = requested["freeze_backbone"]
            elif component in explicit:
                value = getattr(parsed, component)
            else:
                value = parsed.freeze_backbone
            setattr(parsed, component, value)
            if "freeze_backbone" in explicit or "freeze_backbone" in requested:
                explicit.add(component)
        explicit.update(requested)
        parsed._control_explicit = sorted(explicit)
        parsed._recipe_requested = {
            k: v for k, v in requested.items() if k in self.recipe_keys
        }
        for key in self.recipe_keys & CONTROL_KEYS:
            value = getattr(parsed, key, None)
            if value is not None:
                parsed._recipe_requested[key] = value
        return parsed


def recipe_config(args, defaults):
    """Keep optimization keys out of the branch's existing objective resolver."""
    if hasattr(args, "_recipe_requested"):
        return dict(args._recipe_requested)
    requested = (
        json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        if args.config
        else {}
    )
    if not isinstance(requested, dict):
        raise TypeError("Config must be a JSON object")
    return {
        k: v for k, v in requested.items() if k not in CONTROL_KEYS or k in defaults
    }


def public_arguments(args):
    return {key: value for key, value in vars(args).items() if not key.startswith("_")}


def restore_control_arguments(args, checkpoint):
    """Restore new checkpoint controls before the branch validates its historical recipe."""
    state = checkpoint.get("training_controls") if checkpoint else None
    if not state:
        return
    if state.get("version") != 1:
        raise ValueError("Unsupported training-controls checkpoint version")
    explicit = set(getattr(args, "_control_explicit", ()))
    for key, value in state["config"].items():
        if key in explicit and getattr(args, key, None) != value:
            raise ValueError(f"Resume {key} differs from checkpoint; start a new run")
        setattr(args, key, value)


def encoder_modules(model):
    """Find encoder boundaries, including coarse wrappers, without touching readers/decoders."""
    root = model.module if hasattr(model, "module") else model
    result = {"vision": [], "text": []}
    selected = []
    for name, module in root.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        kind = (
            "vision"
            if leaf in {"image_encoder", "vision_encoder"}
            else ("text" if leaf == "text_encoder" else None)
        )
        if kind and not any(name.startswith(prefix + ".") for prefix in selected):
            selected.append(name)
            result[kind].append((name, module))
    return result


class TrainingControls:
    """Encoder policy, optimizer groups and an update-based, resumable LR schedule."""

    def __init__(self, model, args, checkpoint=None, *, include_frozen=False):
        restore_control_arguments(args, checkpoint)
        self.config = {k: getattr(args, k, None) for k in DEFAULTS}
        self.config = {
            k: DEFAULTS[k]
            if v is None
            and k not in {"freeze_vision_encoder", "freeze_text_encoder", "backbone_lr"}
            else v
            for k, v in self.config.items()
        }
        self.config["lr"] = args.lr
        self._validate()
        self.encoders = encoder_modules(model)
        self._saved = checkpoint.get("training_controls") if checkpoint else None
        full_resume = checkpoint is not None and "optimizer_state_dict" in checkpoint
        if full_resume and self._saved is None:
            for key in (
                "optimizer",
                "weight_decay",
                "backbone_lr",
                "lr_scheduler",
                "warmup_ratio",
            ):
                if self.config[key] != DEFAULTS[key]:
                    raise ValueError(
                        f"Legacy resume has no {key} state; initialize a new run instead"
                    )
        for kind, modules in self.encoders.items():
            policy = self.config[f"freeze_{kind}_encoder"]
            if policy is None:
                continue
            if not modules:
                raise ValueError(f"Model has no {kind} encoder to configure")
            for name, module in modules:
                root = model.module if hasattr(model, "module") else model
                if (
                    not policy
                    and getattr(root, "experiment_config", {}).get("experiment")
                    == "gain-guided-refinement"
                ):
                    raise ValueError(
                        "Gain-guided refinement requires its fixed coarse reference; encoders cannot be unfrozen"
                    )
                if full_resume and self._saved is None:
                    was_frozen = not any(p.requires_grad for p in module.parameters())
                    if was_frozen != policy:
                        raise ValueError(
                            f"Legacy resume cannot change freezing of {name}; start a new run"
                        )
                    # Preserve every historical parameter-group member, including unused ones.
                    continue
                module.requires_grad_(not policy)
                # This CLIP layer only normalizes unused pooled CLS; keep it out of DDP.
                if not policy and kind == "vision":
                    for part_name, part in module.named_modules():
                        if part_name.endswith("vision_model.post_layernorm"):
                            part.requires_grad_(False)
                if (
                    not policy
                    and kind == "text"
                    and getattr(module, "backbone", None) == "siglip2"
                ):
                    # SigLIP's pooled projection is not consumed by token cross-attention.
                    module.model.head.requires_grad_(False)
                if policy:
                    for part in module.modules():
                        disable = getattr(part, "gradient_checkpointing_disable", None)
                        if callable(disable):
                            disable()
        self.set_encoder_modes()
        named = list(model.named_parameters())
        self.trainable_names = [name for name, p in named if p.requires_grad]
        self.parameters = [p for _, p in named if p.requires_grad]
        if not self.parameters:
            raise ValueError("No trainable parameters remain")
        if self._saved and self._saved["trainable_names"] != self.trainable_names:
            raise ValueError(
                "Resume trainable parameter layout differs from checkpoint"
            )
        grouped = (
            self.config["weight_decay"] != 0 or self.config["backbone_lr"] is not None
        )
        if grouped:
            encoder_ids = {
                id(p)
                for modules in self.encoders.values()
                for _, m in modules
                for p in m.parameters()
            }
            groups = {}
            for name, p in named:
                if not p.requires_grad:
                    continue
                backbone = id(p) in encoder_ids
                no_decay = (
                    p.ndim <= 1
                    or name.endswith(".bias")
                    or any(
                        token in name
                        for token in (
                            "position_embedding",
                            "token_embedding",
                            "pos_embed",
                            "class_embedding",
                        )
                    )
                )
                key = (backbone, no_decay)
                if key not in groups:
                    groups[key] = {
                        "params": [],
                        "lr": (self.config["backbone_lr"] or args.lr)
                        if backbone
                        else args.lr,
                        "weight_decay": 0.0
                        if no_decay
                        else self.config["weight_decay"],
                        "group_name": f"{'backbone' if backbone else 'head'}_{'no_decay' if no_decay else 'decay'}",
                    }
                groups[key]["params"].append(p)
            parameters = list(groups.values())
        else:
            changed_freezing = any(
                self.config[k] is not None
                for k in ("freeze_vision_encoder", "freeze_text_encoder")
            )
            preserve_legacy_layout = full_resume and self._saved is None
            parameters = (
                [p for _, p in named]
                if include_frozen and (not changed_freezing or preserve_legacy_layout)
                else self.parameters
            )
        optimizer_type = (
            torch.optim.AdamW
            if self.config["optimizer"] == "adamw"
            else torch.optim.Adam
        )
        self.optimizer = optimizer_type(
            parameters, lr=args.lr, weight_decay=self.config["weight_decay"]
        )
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.completed_steps = 0
        self.total_steps = self.warmup_steps = None
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                json.dumps(
                    {
                        "optimization_config": self.config,
                        "encoder_frozen": {
                            kind: self.is_encoder_frozen(kind) for kind in self.encoders
                        },
                        "trainable_parameters": sum(p.numel() for p in self.parameters),
                    }
                ),
                flush=True,
            )

    def _validate(self):
        c = self.config
        if c["optimizer"] not in {"adam", "adamw"} or c["lr_scheduler"] not in {
            "none",
            "cosine",
            "linear",
            "poly",
        }:
            raise ValueError("Unsupported optimizer or LR scheduler")
        for key in ("lr", "poly_power"):
            if not math.isfinite(c[key]) or c[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if c["backbone_lr"] is not None and (
            not math.isfinite(c["backbone_lr"]) or c["backbone_lr"] <= 0
        ):
            raise ValueError("backbone_lr must be positive")
        if not math.isfinite(c["weight_decay"]) or c["weight_decay"] < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(c["warmup_ratio"]) or not 0 <= c["warmup_ratio"] < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not math.isfinite(c["min_lr_ratio"]) or not 0 <= c["min_lr_ratio"] <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        for key in ("freeze_vision_encoder", "freeze_text_encoder"):
            if c[key] is not None and type(c[key]) is not bool:
                raise ValueError(f"{key} must be a boolean")

    def set_encoder_modes(self):
        for kind, modules in self.encoders.items():
            policy = self.config[f"freeze_{kind}_encoder"]
            if policy is not None:
                for _, module in modules:
                    module.train(not policy)

    def is_encoder_frozen(self, kind):
        return all(
            not p.requires_grad
            for _, module in self.encoders[kind]
            for p in module.parameters()
        )

    def log_learning_rates(self, epoch):
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "learning_rates": [
                            group["lr"] for group in self.optimizer.param_groups
                        ],
                    }
                ),
                flush=True,
            )

    def configure_schedule(self, epochs, updates_per_epoch, start_epoch=0):
        self.total_steps = epochs * updates_per_epoch
        if self.total_steps <= 0:
            raise ValueError("Training requires at least one optimizer update")
        self.warmup_steps = min(
            self.total_steps - 1,
            math.ceil(self.total_steps * self.config["warmup_ratio"]),
        )
        active = self.config["lr_scheduler"] != "none" or self.warmup_steps > 0
        if self._saved:
            if active and self._saved["total_steps"] != self.total_steps:
                raise ValueError(
                    "Resume cannot change a scheduled run's total optimizer-step budget"
                )
            if active and self._saved["updates_per_epoch"] != updates_per_epoch:
                raise ValueError(
                    "Resume cannot change accumulation/loader length for an LR schedule"
                )
            self.completed_steps = self._saved["completed_steps"]
        elif start_epoch and active:
            raise ValueError(
                "Checkpoint has no LR schedule state; start a new scheduled run"
            )
        self.updates_per_epoch = updates_per_epoch
        self._apply_lr()

    def _apply_lr(self):
        number = min(self.completed_steps + 1, self.total_steps)
        if self.warmup_steps and number <= self.warmup_steps:
            factor = number / self.warmup_steps
        else:
            progress = (
                (number - self.warmup_steps)
                / max(1, self.total_steps - self.warmup_steps)
                if self.warmup_steps
                else (number - 1) / max(1, self.total_steps - 1)
            )
            shape = {
                "none": lambda: 1.0,
                "cosine": lambda: (1 + math.cos(math.pi * progress)) / 2,
                "linear": lambda: 1 - progress,
                "poly": lambda: (1 - progress) ** self.config["poly_power"],
            }
            factor = (
                self.config["min_lr_ratio"]
                + (1 - self.config["min_lr_ratio"])
                * shape[self.config["lr_scheduler"]]()
            )
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * factor

    def step(self, scaler):
        previous_scale = scaler.get_scale()
        scaler.step(self.optimizer)
        scaler.update()
        if scaler.get_scale() >= previous_scale:
            self.completed_steps += 1
            if self.total_steps is not None:
                self._apply_lr()

    def state_dict(self):
        if self.total_steps is None:
            raise RuntimeError("Configure LR schedule before saving training controls")
        return {
            "version": 1,
            "config": self.config,
            "trainable_names": self.trainable_names,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "updates_per_epoch": self.updates_per_epoch,
        }
