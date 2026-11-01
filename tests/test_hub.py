"""Fresh campaigns upload only their completed research artifacts."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from huggingface_hub.utils import filter_repo_objects
import torch

import eval as evaluation
import train
from checkpoints import read_checkpoint
from experiments import launch
from tests.support import fixture_data


class MemoryHub:
    def __init__(self):
        self.files = {}
        self.calls = []
        self.failures = 0

    def create_repo(self, **kwargs):
        assert kwargs['private'] is False

    def upload_folder(self, **kwargs):
        self.calls.append(kwargs['path_in_repo'])
        if self.failures:
            self.failures -= 1
            raise OSError('simulated upload failure')
        root = Path(kwargs['folder_path'])
        files = [str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()]
        for name in filter_repo_objects(files,allow_patterns=kwargs['allow_patterns']):
            self.files[f'{kwargs["path_in_repo"]}/{name}'] = (root/name).read_bytes()


class HubWorkflow(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = fixture_data(self.root/'data')
        self.hub = MemoryHub()
        mock = patch('experiments.hub.HfApi',return_value=self.hub)
        mock.start();self.addCleanup(mock.stop)
        self.manifest = self.root/'manifest.json'
        self.manifest.write_text(json.dumps({'name':'tiny-study',
            'hub':{'repo_id':'test/results','prefix':'study','private':False},
            'common':{'tiny':True,'num_epochs':2,'batch_size':2,'lr':1e-5,'num_workers':0,'val_every':1},
            'runs':[{'id':'ftc_s42','model':'ftc','seed':42}]}))
        self.args = ['--manifest',str(self.manifest),'--data-root',str(self.data),
                     '--output-root',str(self.root/'outputs'),'--device','cpu']
        self.events = []

    def execute(self, command, **kwargs):
        if command[1] == 'train.py':
            self.events.append('train')
            train.train_main(command[2:])
        else:
            self.events.append(command[command.index('--dataset')+1])
            evaluation.eval_main(command[2:])

    def test_fresh_campaigns_keep_best_and_upload_results_and_summary(self):
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(launch,'subprocess',SimpleNamespace(run=self.execute)):
            first = launch.main(self.args)
            second = launch.main(self.args)
        self.assertNotEqual(first,second)
        self.assertEqual(self.events,['train','val','test']*2)
        self.assertEqual(len(self.hub.calls),4)
        for campaign in (first,second):
            folder = campaign/'ftc_s42'
            self.assertEqual([p.name for p in folder.glob('*.pt')],['best.pt'])
            self.assertTrue((campaign/'summary/runs.csv').exists())
            self.assertFalse((campaign/'records.zip').exists())
            uploaded = f'study/{campaign.name}/ftc_s42/'
            names = {name[len(uploaded):] for name in self.hub.files if name.startswith(uploaded)}
            self.assertEqual(names,{'best.pt','run.json','history.jsonl', *[
                value for split in ('val','test') for value in (
                    f'{split}/metrics.json',f'{split}/logits/index.json',
                    f'{split}/logits/0.npz',f'{split}/logits/1.npz',f'{split}/logits/2.npz')]})
            self.assertIn(f'study/{campaign.name}/summary/summary.json',self.hub.files)
            saved = read_checkpoint(folder/'best.pt')
            self.assertNotIn('optimizer_state_dict',saved)
            self.assertNotIn('format_version',saved)
        a=read_checkpoint(first/'ftc_s42/best.pt')['model_state_dict']
        b=read_checkpoint(second/'ftc_s42/best.pt')['model_state_dict']
        for name in a:
            torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)

    def test_failure_stops_and_next_campaign_trains_from_scratch(self):
        self.hub.failures = 1
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(launch,'subprocess',SimpleNamespace(run=self.execute)):
            with self.assertRaisesRegex(OSError,'upload failure'):
                launch.main(self.args)
            self.assertEqual(len(self.hub.calls),1)
            completed = launch.main(self.args)
        self.assertEqual(self.events,['train','val','test']*2)
        self.assertEqual(len(list((self.root/'outputs').iterdir())),2)
        self.assertTrue((completed/'ftc_s42/best.pt').exists())
        self.assertFalse(list((self.root/'outputs').rglob('backup.json')))
        self.assertFalse(list((self.root/'outputs').rglob('hub_status.json')))


if __name__ == '__main__':
    unittest.main()
