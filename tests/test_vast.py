"""Standalone fresh setup, dataset download/extraction, and foreground interruption."""
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

from experiments import vast
from runtime import PROJECT_ROOT
from tests.support import fixture_data


class VastWorkflow(unittest.TestCase):
    def test_fresh_dataset_download_extract_and_zip_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=fixture_data(root/'original');work=root/'work'
            def download(*args, **kwargs):
                self.assertTrue(kwargs['force_download'])
                archive=Path(kwargs['local_dir'])/'vizwiz.zip'
                archive.parent.mkdir(parents=True)
                with ZipFile(archive,'w') as zipped:
                    for path in original.rglob('*'):
                        if path.is_file():
                            zipped.write(path,str(Path('vizwiz')/path.relative_to(original)))
                return str(archive)
            with patch.object(vast,'hf_hub_download',side_effect=download) as fetch:
                with patch.object(ZipFile,'extractall',side_effect=OSError('extraction failed')):
                    with self.assertRaisesRegex(OSError,'extraction failed'):
                        vast.prepare_dataset(work)
                self.assertTrue((work/'downloads/vizwiz.zip').exists())
                shutil.rmtree(work)
                data=vast.prepare_dataset(work)
                self.assertEqual((data/'val_grounding.json').read_bytes(),
                                 (original/'val_grounding.json').read_bytes())
                self.assertFalse((work/'downloads/vizwiz.zip').exists())
                self.assertFalse((work/'data/.extracted').exists())
                self.assertEqual(fetch.call_count,2)

    def test_notebook_always_clones_and_installs_into_a_fresh_workspace(self):
        notebook=json.loads((PROJECT_ROOT/'vast.ipynb').read_text())
        cells=[cell for cell in notebook['cells'] if cell['cell_type']=='code']
        self.assertEqual(len(cells),2)
        with tempfile.TemporaryDirectory() as tmp:
            state={};exec(''.join(cells[0]['source']),state)
            work=Path(tmp)/'work'
            state.update(WORK_DIR=str(work),HF_TOKEN='offline-placeholder',RUN_IDS=['ftc_s42'])
            calls=[]
            class Process:
                def __init__(self, command, **kwargs):
                    calls.append((command,kwargs))
                    if command[0]=='git':
                        Path(command[-1]).mkdir()
                def wait(self):
                    return 0
            with patch('subprocess.Popen',Process):
                exec(''.join(cells[1]['source']),state)
                with self.assertRaises(FileExistsError):
                    exec(''.join(cells[1]['source']),state)
                shutil.rmtree(work)
                exec(''.join(cells[1]['source']),state)
            self.assertEqual(sum(command[0]=='git' for command,_ in calls),2)
            command,settings=calls[-1]
            self.assertIn('experiments.vast',command)
            self.assertEqual(command[-2:],['--runs','ftc_s42'])
            self.assertEqual(settings['env']['HF_TOKEN'],'offline-placeholder')
            self.assertEqual(settings['env']['UV_PYTHON_INSTALL_DIR'],str(work/'python'))
            self.assertNotIn('offline-placeholder',' '.join(command))
            self.assertTrue(all(not cell['outputs'] for cell in cells))

    def test_interrupt_stops_the_foreground_process_group(self):
        notebook=json.loads((PROJECT_ROOT/'vast.ipynb').read_text())
        cells=[cell for cell in notebook['cells'] if cell['cell_type']=='code']
        with tempfile.TemporaryDirectory() as tmp:
            state={};exec(''.join(cells[0]['source']),state)
            state['WORK_DIR']=str(Path(tmp)/'work')
            process=SimpleNamespace(pid=99999,wait=lambda:0)
            with patch('subprocess.Popen',return_value=process):
                exec(''.join(cells[1]['source']),state)
            process.wait=Mock(side_effect=[KeyboardInterrupt(),0])
            with patch('subprocess.Popen',return_value=process),patch('os.killpg') as stop:
                with self.assertRaises(KeyboardInterrupt):
                    state['run'](['anything'])
            stop.assert_called_once_with(process.pid,signal.SIGTERM)


if __name__ == '__main__':
    unittest.main()
