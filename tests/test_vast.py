"""Standalone notebook bootstrap and sequential training/retry behavior, offline."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import torch

import eval as evaluation
import train
from experiments import launch, vast
from runtime import PROJECT_ROOT
from tests.support import fixture_data
from tests.test_hub import MemoryHub


class VastWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fixed_zip_layout_reuse_and_retry_incomplete_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = fixture_data(root/'original')
            archive = root/'vizwiz.zip'
            with ZipFile(archive, 'w') as zipped:
                for path in original.rglob('*'):
                    if path.is_file():
                        zipped.write(path, str(Path('vizwiz')/path.relative_to(original)))
            workspace = root/'work'
            with patch.object(vast, 'hf_hub_download', return_value=str(archive)) as download:
                with patch.object(ZipFile, 'extractall', side_effect=OSError('interrupted extraction')):
                    with self.assertRaisesRegex(OSError, 'interrupted'):
                        vast.prepare_dataset(workspace)
                self.assertFalse((workspace/'data/.extracted').exists())
                self.assertTrue(archive.exists())
                data = vast.prepare_dataset(workspace)
                self.assertFalse(archive.exists())
                self.assertEqual(data, workspace/'data/vizwiz')
                self.assertEqual((data/'train_grounding.json').read_bytes(),
                                 (original/'train_grounding.json').read_bytes())
                self.assertEqual(download.call_count, 2)
                self.assertEqual(vast.prepare_dataset(workspace), data)
                self.assertEqual(download.call_count, 2)
                # Also clear an archive retained by an earlier notebook version.
                retained = workspace/'downloads/vizwiz.zip'
                retained.parent.mkdir(parents=True)
                retained.write_bytes(b'already extracted')
                self.assertEqual(vast.prepare_dataset(workspace), data)
                self.assertFalse(retained.exists())
                self.assertEqual(download.call_count, 2)

    def test_study_retries_upload_resumes_and_restores_on_new_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = fixture_data(root/'data')
            workspace = root/'work'
            partial = workspace/'outputs/ftc_s42'
            partial.mkdir(parents=True)
            (partial/'run.json').write_text('partial first attempt')
            manifest = {'name':'tiny-study', 'common':{'num_epochs':1, 'tiny':True,
                        'batch_size':4, 'num_workers':0, 'lr':1e-5, 'val_every':1,
                        'checkpoint_policy':'rolling'},
                        'runs':[{'id':'ftc_s42', 'model':'ftc', 'seed':42}]}
            path = root/'manifest.json'
            path.write_text(json.dumps(manifest))
            hub = MemoryHub(root)
            hub.file_exists = lambda repo, name, **kw: name in hub.commits[hub.head]
            calls = []
            def execute(command, **kwargs):
                calls.append(command[1])
                {'train.py':train.train_main, 'eval.py':evaluation.eval_main}[command[1]](command[2:])
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(vast, 'prepare_dataset', return_value=data), \
                    patch.object(vast, 'HfApi', return_value=hub), \
                    patch('experiments.hub._api', return_value=hub), \
                    patch('huggingface_hub.hf_hub_download', side_effect=hub.download), \
                    patch('experiments.hub.time.sleep'), \
                    patch.object(launch, 'subprocess', SimpleNamespace(run=execute)):
                hub.failures = 3
                with self.assertRaisesRegex(OSError, 'interrupted connection'):
                    vast.run_study(workspace, 'test/study', 'study', manifest=path, device='cpu')
                self.assertEqual(calls, ['train.py', 'eval.py', 'eval.py'])
                self.assertTrue((partial/'latest.pt').is_file())
                self.assertEqual(len(list((workspace/'interrupted').iterdir())), 1)
                history = (partial/'history.jsonl').read_bytes()
                calls.clear()
                report = vast.run_study(workspace, 'test/study', 'study', manifest=path, device='cpu')
                self.assertEqual(calls, [])
                self.assertEqual(hub.calls, 4)
                self.assertEqual((partial/'history.jsonl').read_bytes(), history)
                self.assertTrue((report/'runs.csv').is_file())
                self.assertTrue((report/'records.zip').is_file())
                # Running the notebook on a replacement instance restores completed
                # backups, avoiding a fresh run at an occupied Hub path.
                restored = root/'replacement'
                vast.run_study(restored, 'test/study', 'study', manifest=path, device='cpu')
                self.assertEqual(calls, [])
                self.assertEqual(hub.calls, 4)
                self.assertEqual((restored/'outputs/ftc_s42/latest.pt').read_bytes(),
                                 (partial/'latest.pt').read_bytes())
                self.assertEqual((restored/'outputs/ftc_s42/val/logits/0.npz').read_bytes(),
                                 (partial/'val/logits/0.npz').read_bytes())

    def test_uploaded_notebook_bootstraps_and_reuses_source_without_prompts(self):
        notebook = json.loads((PROJECT_ROOT/'vast.ipynb').read_text())
        cells = [cell for cell in notebook['cells'] if cell['cell_type']=='code']
        self.assertEqual(len(cells), 2)
        with tempfile.TemporaryDirectory() as tmp:
            state = {}
            exec(''.join(cells[0]['source']), state)
            state.update(WORK_DIR=tmp, HF_TOKEN='offline-placeholder', RUN_IDS=['ftc_s42'])
            calls = []
            class Process:
                def __init__(self, command, **kwargs):
                    calls.append((command, kwargs))
                    if command[0]=='git':
                        Path(command[-1]).mkdir()
                def wait(self):
                    return 0
            with patch('subprocess.Popen', Process):
                exec(''.join(cells[1]['source']), state)
                first = list(calls)
                exec(''.join(cells[1]['source']), state)
            self.assertEqual(sum(command[0]=='git' for command, _ in calls), 1)
            self.assertEqual(first[0][0][-1], str(Path(tmp)/'source'))
            command, settings = first[-1]
            self.assertEqual(command[0], str(Path(tmp)/'source/.venv/bin/python'))
            self.assertIn('experiments.vast', command)
            self.assertEqual(command[-2:], ['--runs', 'ftc_s42'])
            self.assertEqual(settings['env']['HF_TOKEN'], 'offline-placeholder')
            self.assertNotIn('offline-placeholder', ' '.join(command))
            self.assertTrue(all(not cell['outputs'] and cell['execution_count'] is None for cell in cells))

    def test_interrupt_stops_the_foreground_process_group(self):
        notebook = json.loads((PROJECT_ROOT/'vast.ipynb').read_text())
        cells = [cell for cell in notebook['cells'] if cell['cell_type']=='code']
        with tempfile.TemporaryDirectory() as tmp:
            state = {}
            exec(''.join(cells[0]['source']), state)
            state['WORK_DIR'] = tmp
            process = SimpleNamespace(pid=99999, wait=lambda:0)
            with patch('subprocess.Popen', return_value=process):
                exec(''.join(cells[1]['source']), state)
            from unittest.mock import Mock
            process.wait = Mock(side_effect=[KeyboardInterrupt(), 0])
            with patch('subprocess.Popen', return_value=process), patch('os.killpg') as stop:
                with self.assertRaises(KeyboardInterrupt):
                    state['run'](['anything'])
            stop.assert_called_once_with(process.pid, signal.SIGTERM)


if __name__ == '__main__':
    unittest.main()
