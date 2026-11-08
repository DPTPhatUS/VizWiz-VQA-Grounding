"""Selected-model checkpoint loading, reconstruction, and saving."""
import os
from pathlib import Path
import torch


def validate_checkpoint(saved):
    if not isinstance(saved, dict):
        raise ValueError('Expected a saved model dictionary')
    for key in ('model_state_dict', 'experiment_config', 'run_config'):
        if not isinstance(saved.get(key), dict):
            raise ValueError(f'Checkpoint requires a {key} dictionary')
    return saved


def read_checkpoint(path):
    return validate_checkpoint(torch.load(path, map_location='cpu', weights_only=True))


def model_options(saved, explicit_model=None):
    config = validate_checkpoint(saved)['experiment_config']
    kind = config.get('model')
    if kind not in ('cm', 'ftc', 'ftc-capacity'):
        raise ValueError(f'Unsupported model in checkpoint: {kind}')
    if explicit_model is not None and kind != explicit_model:
        raise ValueError('Explicit model conflicts with checkpoint configuration')
    required = ['tiny'] + (['connection_mode', 'context_enabled', 'membership_enabled'] if kind == 'cm' else [])
    if any(key not in config for key in required):
        raise ValueError('Incomplete model configuration in checkpoint')
    return dict(model=kind, tiny=config['tiny'],
                **({'connection_mode':config['connection_mode'], 'context':config['context_enabled'],
                    'membership':config['membership_enabled'],
                    'local_refinement':config.get('local_refinement', 'membership')} if kind == 'cm' else {}))


def load_model_weights(model, saved, assign=False):
    config = validate_checkpoint(saved)['experiment_config']
    if config != model.experiment_config:
        raise ValueError('Checkpoint configuration does not match model configuration')
    model.load_state_dict(saved['model_state_dict'], strict=True, assign=assign)


def model_from_checkpoint(saved):
    from models.factory import build_model
    options = model_options(saved)
    # Construct without a second copy of pretrained CLIP weights.
    with torch.device('meta'):
        restored = build_model(**options, pretrained=False)
    load_model_weights(restored, saved, assign=True)
    materialize_position_buffers(restored)
    return restored


def materialize_position_buffers(model):
    """Reconstruct CLIP position IDs, which are nonpersistent and absent from state_dict."""
    device = next(model.parameters()).device
    for module in model.modules():
        for name, buffer in module._buffers.items():
            if buffer is not None and buffer.is_meta:
                if name != 'position_ids' or not hasattr(module, 'position_embedding'):
                    raise ValueError(f'Unmaterialized nonpersistent buffer: {type(module).__name__}.{name}')
                module._buffers[name] = torch.arange(buffer.shape[-1], dtype=buffer.dtype,
                                                     device=device).expand(buffer.shape)


def save_checkpoint(path, saved):
    validate_checkpoint(saved)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(saved, temporary)
    os.replace(temporary, path)


def checkpoint_info(path, saved):
    """Plain experiment name and selected epoch for saved results."""
    return {'run':saved['run_config'].get('run_id', Path(path).parent.name),
            'epoch':saved.get('epoch')}
