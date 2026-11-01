"""Offline dataset, capacity-probe, worker, and Run All behavior for the Vast notebook."""
import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile,ZipInfo

import torch

from experiments.vast import (dataset_root,extract_dataset,prepare_dataset,memory_probe,
                             selected_runs,worker,hardware_report,GIB)
from tests.support import fixture_data
from runtime import PROJECT_ROOT


def archive_fixture(root,wrapped=True):
    data=fixture_data(root/'original')
    archive=root/'vizwiz.zip'
    with ZipFile(archive,'w') as zipped:
        for path in data.rglob('*'):
            if path.is_file():
                zipped.write(path,str((Path('vizwiz') if wrapped else Path())/path.relative_to(data)))
    return archive


class VastWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_extract_detects_nested_data_and_reuses_pinned_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=archive_fixture(root)
            source={'repo_id':'owner/data','filename':'vizwiz.zip','requested_revision':'main','commit':'sha-one'}
            result=extract_dataset(archive,root/'extracted',source)
            self.assertEqual(dataset_root(root/'extracted'),root/'extracted/vizwiz')
            self.assertEqual(result['archive_sha256'],extract_dataset(archive,root/'extracted',source)['archive_sha256'])
            self.assertTrue(extract_dataset(archive,root/'extracted',source)['reused'])
            with self.assertRaisesRegex(ValueError,'source changed'):
                extract_dataset(archive,root/'extracted',dict(source,commit='sha-two'))
            with patch('huggingface_hub.HfApi',side_effect=AssertionError('must not follow a moving main branch')):
                reused=prepare_dataset('owner/data','vizwiz.zip','main',root/'downloads',root/'extracted')
            self.assertEqual(reused['source']['commit'],'sha-one')

    def test_unsafe_zip_incomplete_data_and_existing_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('../escape','/absolute','C:/escape','folder\\escape','symlink'):
                archive=root/'bad.zip'
                with ZipFile(archive,'w') as zipped:
                    info=ZipInfo(name)
                    if name=='symlink':info.external_attr=(stat.S_IFLNK|0o777)<<16
                    zipped.writestr(info,'unexpected')
                with self.assertRaisesRegex(ValueError,'Unsafe'):
                    extract_dataset(archive,root/'destination',{})
                self.assertFalse((root/'destination').exists())
            with ZipFile(root/'incomplete.zip','w') as zipped:zipped.writestr('train_grounding.json','{}')
            with self.assertRaisesRegex(ValueError,'complete dataset'):
                extract_dataset(root/'incomplete.zip',root/'destination',{})
            self.assertFalse((root/'destination').exists())
            (root/'destination').mkdir();(root/'destination/keep').write_text('keep')
            with self.assertRaisesRegex(ValueError,'existing data'):
                extract_dataset(root/'incomplete.zip',root/'destination',{})
            self.assertEqual((root/'destination/keep').read_text(),'keep')

    def test_hardware_driver_failure_stops_before_gpu_initialization(self):
        result=SimpleNamespace(stdout='550.90.01\n')
        with patch('experiments.vast.subprocess.run',return_value=result),patch('torch.cuda.is_available') as cuda:
            with self.assertRaisesRegex(ValueError,'580'):
                hardware_report('/tmp')
            cuda.assert_not_called()

    def test_tiny_memory_probe_completes_adam_steps_at_matched_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=memory_probe(fixture_data(Path(tmp)/'data'),tiny=True,device='cpu')
            self.assertEqual(result['global_batch'],4)
            self.assertEqual(result['measurements'][0]['microbatch'],4)
            self.assertEqual(result['measurements'][0]['steps'],2)
            self.assertTrue(torch.isfinite(torch.tensor(result['measurements'][0]['loss'])))

    def test_worker_runs_resumes_completed_run_and_rejects_changed_world_size(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);data=fixture_data(root/'data')
            manifest={'name':'vast-smoke','common':{'tiny':True,'num_epochs':2,'batch_size':4,'lr':1e-5,
                'num_workers':0,'val_every':1,'checkpoint_policy':'rolling'},
                'runs':[{'id':'cm_shared_s42','model':'cm','connection_mode':'shared','seed':42}]}
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            config={'manifest':str(path),'data_root':str(data),'output_root':str(root/'runs'),
                    'session_id':'first','session_dir':str(root/'first'),'processes':1,
                    'runs':[],'device':'cpu','upload':False,'auto_restore':False}
            config_path=root/'config.json';config_path.write_text(json.dumps(config));worker(config_path)
            status=json.loads((root/'first/status.json').read_text())
            self.assertEqual(status['state'],'complete')
            self.assertTrue(Path(status['records_archive']).is_file())
            history=(root/'runs/cm_shared_s42/history.jsonl').read_bytes()
            config.update(session_id='second',session_dir=str(root/'second'))
            config_path.write_text(json.dumps(config));worker(config_path)
            self.assertEqual((root/'runs/cm_shared_s42/history.jsonl').read_bytes(),history)
            config.update(processes=2,session_id='wrong-world',session_dir=str(root/'wrong-world'))
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'process count'):worker(config_path)
            self.assertEqual(json.loads((root/'wrong-world/status.json').read_text())['state'],'failed')
            with self.assertRaisesRegex(ValueError,'Unknown'):selected_runs(manifest,['missing'])

    def test_notebook_run_all_and_active_job_reattachment_without_real_cloud_calls(self):
        notebook=json.loads((PROJECT_ROOT/'vast.ipynb').read_text())
        self.assertEqual(notebook['nbformat'],4)
        code_cells={cell['id']:''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code'}
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])
                compile(''.join(cell['source']),cell['id'],'exec')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'workspace';work.mkdir()
            source=root/'source';source.mkdir()
            for name in ('train.py','pyproject.toml','uv.lock','configs/high_priority.json','experiments/vast.py','experiments/hub.py'):
                destination=source/name;destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(PROJECT_ROOT/name,destination)
            data=fixture_data(root/'data')
            tiny={'name':'notebook-smoke','common':{'tiny':True,'num_epochs':1,'batch_size':4,
                'lr':1e-5,'num_workers':0,'val_every':1,'checkpoint_policy':'rolling'},
                'runs':[{'id':'ftc_s42','model':'ftc','seed':42}]}
            tiny_path=root/'tiny.json';tiny_path.write_text(json.dumps(tiny))
            calls=[];spawned=[]
            def fake_run(command,**kwargs):
                command=[str(part) for part in command];calls.append(command)
                if 'experiments.vast' in command:
                    report=Path(command[command.index('--report')+1]);report.parent.mkdir(parents=True,exist_ok=True)
                    report.write_text(json.dumps({'data_root':str(data)} if 'data' in command else {'ok':True}))
                return subprocess.CompletedProcess(command,0)
            def fake_spawn(command,**kwargs):
                self.assertTrue(kwargs['start_new_session'])
                path=Path(command[-1]);config=json.loads(path.read_text())
                self.assertNotIn('HF_TOKEN',json.dumps(config))
                config.update(manifest=str(tiny_path),device='cpu',upload=False,auto_restore=False)
                path.write_text(json.dumps(config))
                worker(path)
                spawned.append((command,path))
                return SimpleNamespace(pid=424242)
            proxy=SimpleNamespace(run=fake_run,Popen=fake_spawn,DEVNULL=subprocess.DEVNULL,STDOUT=subprocess.STDOUT)
            display=SimpleNamespace(FileLink=lambda path:path,display=lambda value:None)
            original_import=builtins.__import__
            def custom_import(name,globals=None,locals=None,fromlist=(),level=0):
                if name=='subprocess':return proxy
                if name=='IPython.display':return display
                return original_import(name,globals,locals,fromlist,level)
            def namespace():
                scope={'__builtins__':dict(vars(builtins),__import__=custom_import)}
                exec(code_cells['settings'],scope)
                scope.update(WORK_DIR=str(work),REPO_DIR=str(source),RUN_IDS=['ftc_s42'])
                return scope
            with patch.dict(os.environ,{'HF_TOKEN':'offline-test-placeholder'}),\
                 patch('shutil.which',return_value='/bin/true'),\
                 patch('shutil.disk_usage',return_value=SimpleNamespace(free=500*GIB)),\
                 contextlib.redirect_stdout(io.StringIO()):
                scope=namespace()
                for name,code in code_cells.items():
                    if name!='settings':exec(compile(code,name,'exec'),scope)
                self.assertEqual(scope['FINAL_STATUS']['state'],'complete')
                self.assertEqual(len(spawned),1)
                self.assertTrue(any('auth' in command for command in calls))
                self.assertFalse(any('offline-test-placeholder' in str(command) for command in calls))
                config_path=spawned[0][1]
                before=len(calls)
                original_read=Path.read_bytes
                def proc_bytes(path):
                    if str(path)=='/proc/424242/cmdline':
                        return b'python\0-m\0experiments.vast\0worker\0--config\0'+os.fsencode(config_path)+b'\0'
                    return original_read(path)
                with patch.object(Path,'read_bytes',proc_bytes):
                    attached=namespace()
                    for name,code in code_cells.items():
                        if name!='settings':exec(compile(code,name,'exec'),attached)
                self.assertTrue(attached['LIVE_JOB'])
                self.assertEqual(len(spawned),1)
                self.assertEqual(len(calls),before)


if __name__=='__main__':
    unittest.main()
