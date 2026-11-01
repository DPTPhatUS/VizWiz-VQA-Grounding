"""Offline Hub transactions, interruption recovery, and cross-machine resume."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from huggingface_hub.errors import EntryNotFoundError
import torch

from tests.support import fixture_data
import eval as evaluation
from experiments.hub import backup_plan, check_destination, download_run, hub_path, upload_run
from experiments.launch import commands
from experiments import launch
from checkpoints import read_checkpoint
import train as train
import preflight as preflight


class MemoryHub:
    """A commit-addressed fake implementing the Hub operations this workflow uses."""
    def __init__(self, root):
        self.root = root
        self.head = '0'*40
        self.commits = {self.head:{}}
        self.private = False
        self.calls = 0
        self.failures = 0
        self.corrupt = None

    def create_repo(self, **kwargs):
        self.private = kwargs['private']

    def auth_check(self, repo_id, **kwargs):
        assert kwargs['write'] is True

    def repo_info(self, repo_id, revision=None, **kwargs):
        return SimpleNamespace(sha=self.head if revision in (None,'main') else revision, private=self.private)

    def list_repo_files(self, repo_id, revision=None, **kwargs):
        return list(self.commits[revision or self.head])

    def create_commit(self, **kwargs):
        assert kwargs['parent_commit'] == self.head
        self.calls += 1
        if self.calls <= self.failures:
            raise OSError('simulated interrupted connection')
        files = dict(self.commits[self.head])
        for op in kwargs['operations']:
            value = op.path_or_fileobj
            files[op.path_in_repo] = value if isinstance(value,bytes) else Path(value).read_bytes()
        self.head = f'{len(self.commits):040x}'
        self.commits[self.head] = files
        return SimpleNamespace(oid=self.head)

    def download(self, repo_id, filename, revision=None, local_dir=None, **kwargs):
        revision = self.head if revision in (None,'main') else revision
        if filename not in self.commits[revision]:
            raise EntryNotFoundError('missing test file')
        destination = Path(local_dir or self.root/'cache'/revision)/filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = self.commits[revision][filename]
        destination.write_bytes(b'corrupt' if filename == self.corrupt else data)
        return str(destination)


class HubWorkflow(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = fixture_data(self.root/'data')
        self.hub = MemoryHub(self.root)
        for target, replacement in [('experiments.hub._api',lambda:self.hub),
                                    ('huggingface_hub.hf_hub_download',self.hub.download),
                                    ('experiments.hub.time.sleep',lambda _:None)]:
            mock = patch(target,replacement)
            mock.start()
            self.addCleanup(mock.stop)
        self.repo = 'test/checkpoints'
        self.path = 'study/ftc_s42'
        self.out = self.root/'run'
        self.args = ['--tiny','--model','ftc','--data-root',str(self.data),'--device','cpu',
                     '--num-workers','0','--batch-size','2','--num-epochs','2','--val-every','1',
                     '--checkpoint-policy','rolling',
                     '--output-dir',str(self.out)]

    def train(self, args=None, scores=(.8,.7)):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(train,'validation_iou',side_effect=scores):
            train.train_main(args or self.args)

    def test_final_backup_metrics_and_verified_cross_machine_resume(self):
        self.train()
        self.assertEqual(self.hub.calls,0)
        (self.out/'private_photo.png').write_bytes(b'not an experiment record')
        with contextlib.redirect_stdout(io.StringIO()):
            evaluation.eval_main(['--checkpoint',str(self.out/'best.pt'),'--data-root',str(self.data),
                '--dataset','test','--device','cpu','--num-workers','0','--output-dir',str(self.out/'test')])
        final = upload_run(self.out,self.repo,self.path)
        self.assertEqual(self.hub.calls,1)
        current = json.loads(self.hub.commits[final['commit']][self.path+'/backup.json'])
        self.assertIn('test/metrics.json',current['files'])
        self.assertFalse(any(name.endswith('.png') for name in current['files']))
        restored = self.root/'another-machine'
        result = download_run(self.repo,self.path,restored)
        self.assertEqual(result['commit'],final['commit'])
        self.assertEqual(backup_plan(restored,self.path),current)
        for key,value in read_checkpoint(self.out/'latest.pt')['model_state_dict'].items():
            torch.testing.assert_close(value,read_checkpoint(restored/'latest.pt')['model_state_dict'][key],rtol=0,atol=0)
        resumed = list(self.args)
        resumed[resumed.index('--output-dir')+1] = str(restored)
        resumed[resumed.index('--num-epochs')+1] = '3'
        self.train(resumed+['--resume-checkpoint',str(restored/'latest.pt')],scores=(.6,))
        upload_run(restored,self.repo,self.path)
        self.assertEqual(read_checkpoint(restored/'latest.pt')['epoch'],3)
        self.assertEqual(read_checkpoint(restored/'best.pt')['epoch'],1)

    def test_interrupted_upload_preserves_finished_run_and_can_retry(self):
        self.train()
        self.hub.failures = 3
        with self.assertRaisesRegex(OSError,'interrupted'):
            upload_run(self.out,self.repo,self.path)
        self.assertEqual(self.hub.calls,3)
        self.assertEqual(read_checkpoint(self.out/'latest.pt')['epoch'],2)
        self.assertEqual(read_checkpoint(self.out/'best.pt')['epoch'],1)
        self.assertIn('error',json.loads((self.out/'hub_status.json').read_text()))
        upload_run(self.out,self.repo,self.path)
        self.assertEqual(self.hub.calls,4)
        self.assertEqual(read_checkpoint(self.out/'latest.pt')['epoch'],2)

    def test_corrupt_restore_and_local_overwrite_are_rejected(self):
        self.train()
        upload_run(self.out,self.repo,self.path)
        destination = self.root/'new-machine'
        self.hub.corrupt = self.path+'/best.pt'
        with self.assertRaisesRegex(ValueError,'Corrupt'):
            download_run(self.repo,self.path,destination)
        self.assertFalse(destination.exists())
        self.hub.corrupt = None
        download_run(self.repo,self.path,destination)
        with self.assertRaisesRegex(ValueError,'overwrite'):
            download_run(self.repo,self.path,destination)

    def test_remote_identity_dry_run_and_allowlist(self):
        self.train()
        upload_run(self.out,self.repo,self.path)
        with self.assertRaisesRegex(ValueError,'another run'):
            check_destination(self.repo,self.path)
        plan = upload_run(self.out,self.repo,self.path,dry_run=True)
        self.assertEqual(self.hub.calls,1)
        self.assertEqual(plan['epoch'],2)
        (self.out/'run.json').unlink()
        with self.assertRaises(FileNotFoundError):
            backup_plan(self.out,self.path)
        for value in ('','/root','../run','study/../run','study//run','study\\run'):
            with self.assertRaises(ValueError):hub_path(value)

    def test_manual_midrun_snapshot_and_historical_restore(self):
        first=list(self.args)
        first[first.index('--num-epochs')+1]='1'
        self.train(first,scores=(.8,))
        snapshot=upload_run(self.out,self.repo,self.path)['commit']
        self.train(self.args+['--resume-checkpoint',str(self.out/'latest.pt')],scores=(.7,))
        upload_run(self.out,self.repo,self.path)
        historical=self.root/'epoch-one'
        self.assertEqual(download_run(self.repo,self.path,historical,revision=snapshot)['epoch'],1)
        with self.assertRaisesRegex(ValueError,'newer'):
            upload_run(historical,self.repo,self.path)

    def test_launcher_uploads_once_after_training_and_test_and_resume_skips_training(self):
        manifest=json.loads(Path('configs/high_priority.json').read_text())
        cmds=commands(manifest,manifest['runs'][0],self.data,self.out,'cuda',2)
        self.assertTrue(all('--hf-repo' not in cmd for cmd in cmds))
        manifest['runs']=manifest['runs'][:1]
        manifest['common'].update(num_epochs=2,tiny=True,num_workers=0)
        manifest['hub']={'repo_id':self.repo,'prefix':'study','private':False}
        path=self.root/'manifest.json';path.write_text(json.dumps(manifest))
        calls=[]
        def execute(command,**kwargs):
            self.assertEqual(self.hub.calls,0)
            calls.append(command[1])
            {'preflight.py':preflight.main,'train.py':train.train_main,
             'eval.py':evaluation.eval_main}[command[1]](command[2:])
        argv=['--manifest',str(path),'--data-root',str(self.data),'--device','cpu',
              '--output-root',str(self.root/'campaign')]
        with contextlib.redirect_stdout(io.StringIO()),patch.object(launch,'subprocess',SimpleNamespace(run=execute)):
            launch.main(argv)
            self.assertEqual(calls,['preflight.py','train.py','eval.py'])
            self.assertEqual(self.hub.calls,1)
            calls.clear()
            launch.main(argv+['--resume'])
            self.assertEqual(calls,[])
            self.assertEqual(self.hub.calls,1)
        index=json.loads(self.hub.commits[self.hub.head]['study/ftc_s42/backup.json'])
        self.assertIn('test/metrics.json',index['files'])
        self.assertIn('manifest_run.json',index['files'])


if __name__ == '__main__':
    unittest.main()
