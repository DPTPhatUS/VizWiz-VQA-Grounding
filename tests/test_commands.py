"""Exercise public commands from the source checkout through real Python subprocesses."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from tests.support import fixture_data
from runtime import PROJECT_ROOT
from checkpoints import save_checkpoint
from models.factory import build_model


def run_command(*arguments):
    result = subprocess.run([sys.executable, *arguments], cwd=PROJECT_ROOT,
                            env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                            capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(f'Command failed: {arguments}\n{result.stdout}\n{result.stderr}')
    return result.stdout


class CommandTests(unittest.TestCase):
    def test_public_script_and_module_commands_resolve_and_preserve_options(self):
        scripts = {'train.py':'train', 'eval.py':'eval', 'IoU.py':'eval',
                   'predict_save.py':'predict_save', 'visualize_predictions.py':'visualize_predictions',
                   'preflight.py':'preflight', 'bake_exif.py':'bake_exif'}
        commands = [(script,'--help') for script in scripts]
        commands += [('-m',name,'--help') for name in sorted(set(scripts.values()))]
        commands += [('-m',f'experiments.{name}','--help')
                     for name in ('launch','summarize','hub','export')]
        commands += [('-m',f'diagnostics.{name}','--help')
                     for name in ('thresholds','questions','roles','benchmark')]
        with ThreadPoolExecutor(max_workers=4) as pool:
            reports = dict(zip(commands,pool.map(lambda command:run_command(*command),commands)))
        for script,name in scripts.items():
            flags = lambda text:set(re.findall(r'--[a-z][a-z0-9-]*',text))
            self.assertEqual(flags(reports[(script,'--help')]),
                             flags(reports[('-m',name,'--help')]))

    def test_manifest_launch_resume_and_summary_through_subprocesses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=fixture_data(root/'data')
            manifest={'name':'structure-smoke','common':{'tiny':True,'num_epochs':2,'batch_size':2,
                'lr':1e-5,'num_workers':0,'val_every':1,'checkpoint_policy':'rolling'},
                'runs':[{'id':'context_only_s42','model':'cm','connection_mode':'separate',
                         'disable_membership':True,'seed':42}]}
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            args=('-m','experiments.launch','--manifest',str(path),
                  '--data-root',str(data),'--output-root',str(root/'runs'),
                  '--device','cpu','--processes','1','--no-hub')
            run_command(*args)
            folder=root/'runs/context_only_s42'
            for name in ('latest.pt','best.pt','run.json','history.jsonl','manifest_run.json','test/metrics.json'):
                self.assertTrue((folder/name).is_file(),name)
            history=(folder/'history.jsonl').read_bytes()
            run_command(*args,'--resume')
            self.assertEqual((folder/'history.jsonl').read_bytes(),history)
            run_command('-m','experiments.summarize','--manifest',str(path),
                        '--root',str(root/'runs'),'--output-dir',str(root/'summary'))
            result=json.loads((root/'summary/summary.json').read_text())
            self.assertEqual(len(result['runs']),1)
            self.assertEqual(result['missing_runs'],[])

    def test_root_and_module_prediction_produce_the_same_native_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=fixture_data(root/'data')
            torch.manual_seed(42);model=build_model('ftc',tiny=True)
            checkpoint=root/'model.pt'
            save_checkpoint(checkpoint,{'format_version':1,'model_state_dict':model.state_dict(),
                'experiment_config':model.experiment_config,'run_config':{'image_size':32}})
            common=('--checkpoint',str(checkpoint),'--image',str(data/'val/0.jpg'),
                    '--question','What color?','--device','cpu')
            run_command('predict_save.py',*common,'--output',str(root/'script.png'))
            run_command('-m','predict_save',*common,'--output',str(root/'module.png'))
            with Image.open(root/'script.png') as script,Image.open(root/'module.png') as module:
                self.assertEqual(script.size,(19,13))
                np.testing.assert_array_equal(np.asarray(script),np.asarray(module))


if __name__ == '__main__':
    unittest.main()
