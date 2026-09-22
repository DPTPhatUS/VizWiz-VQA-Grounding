"""Strict weight loading with experiment identity checks for training checkpoints."""
def load_model_weights(model, checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        saved = checkpoint.get("experiment_config")
        if saved is not None and saved != model.experiment_config:
            raise ValueError(
                f"Checkpoint configuration {saved} does not match model configuration "
                f"{model.experiment_config}. Use the matching experiment branch/options."
            )
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(state, strict=True)
