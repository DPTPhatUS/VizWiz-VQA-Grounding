"""Batch transfer and plain PyTorch weight loading."""
import torch


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def read_checkpoint(path):
    """This experiment uses full checkpoints with embedded architecture settings."""
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model_weights(model, saved):
    """Strictly load tensors and require the same experiment configuration."""
    if saved['experiment_config'] != model.experiment_config:
        raise ValueError("Checkpoint configuration does not match model configuration")
    model.load_state_dict(saved['model_state_dict'], strict=True)


def model_from_checkpoint(saved):
    from models.model import GroundingModel
    config = saved['experiment_config']
    model = GroundingModel(tiny=config['tiny'], connection_mode=config['connection_mode'])
    load_model_weights(model, saved)
    return model
