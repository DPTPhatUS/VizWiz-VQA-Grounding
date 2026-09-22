import importlib.util
import unittest
import torch
from torch import nn

class CheckpointTests(unittest.TestCase):
    def test_rejects_different_experiment_before_loading(self):
        self.assertIsNotNone(importlib.util.find_spec('models.checkpoint'), 'checkpoint identity helper missing')
        from models.checkpoint import load_model_weights
        model = nn.Linear(2, 1)
        model.experiment_config = {'architecture': 'one'}
        with self.assertRaisesRegex(ValueError, 'configuration'):
            load_model_weights(model, {'model_state_dict': model.state_dict(), 'experiment_config': {'architecture': 'two'}})
        load_model_weights(model, {'model_state_dict': model.state_dict(), 'experiment_config': model.experiment_config})

    def test_raw_state_dict_still_loads_strictly(self):
        self.assertIsNotNone(importlib.util.find_spec('models.checkpoint'))
        from models.checkpoint import load_model_weights
        model = nn.Linear(2, 1)
        model.experiment_config = {'architecture': 'one'}
        load_model_weights(model, model.state_dict())
        with self.assertRaises(RuntimeError):
            load_model_weights(model, {})
