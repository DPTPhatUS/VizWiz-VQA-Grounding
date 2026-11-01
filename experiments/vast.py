"""Single-GPU, sequential training for the standalone Vast notebook."""
import argparse
import json
from pathlib import Path
import time
from zipfile import ZipFile

from huggingface_hub import HfApi, hf_hub_download

from experiments import export, launch, summarize
from experiments.hub import DEFAULT_REPO, download_run
from runtime import DEFAULT_MANIFEST


def prepare_dataset(workspace):
    """The supplied ZIP contains vizwiz/ directly; reuse a completed extraction."""
    workspace = Path(workspace)
    data = workspace/'data'
    archive = workspace/'downloads/vizwiz.zip'
    ready = data/'.extracted'
    if not ready.exists():
        archive = Path(hf_hub_download('Last-Bullet/VizWiz-VQA-Grounding', 'vizwiz.zip',
                                      repo_type='dataset', local_dir=workspace/'downloads'))
        data.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive) as zipped:
            zipped.extractall(data)
        ready.touch()
    archive.unlink(missing_ok=True)
    return data/'vizwiz'


def run_study(workspace, hf_repo, hf_prefix, run_ids=None, *, manifest=DEFAULT_MANIFEST, device='cuda'):
    workspace = Path(workspace)
    data = prepare_dataset(workspace)
    output = workspace/'outputs'
    study = json.loads(Path(manifest).read_text())
    runs = study['runs']
    if run_ids:
        by_id = {run['id']: run for run in runs}
        runs = [by_id[run_id] for run_id in run_ids]
    api = HfApi(token=True)
    api.create_repo(repo_id=hf_repo, repo_type='model', private=False, exist_ok=True)
    for run in runs:
        folder = output/run['id']
        # A failure before the first saved epoch has nothing to resume. Retain its
        # partial records outside the results, then let this run start again.
        if folder.exists() and not (folder/'latest.pt').exists():
            interrupted = workspace/'interrupted'/f'{run["id"]}-{time.time_ns()}'
            interrupted.parent.mkdir(parents=True, exist_ok=True)
            folder.rename(interrupted)
        remote = f'{hf_prefix}/{run["id"]}'
        if not folder.exists() and api.file_exists(hf_repo, f'{remote}/backup.json', repo_type='model'):
            print(f'Restoring {run["id"]} from Hugging Face', flush=True)
            download_run(hf_repo, remote, folder, api=api)
        args = ['--manifest', str(manifest), '--data-root', str(data), '--output-root', str(output),
                '--processes', '1', '--device', device, '--runs', run['id'], '--skip-preflight',
                '--hf-repo', hf_repo, '--hf-prefix', hf_prefix]
        if (folder/'latest.pt').exists():
            args.append('--resume')
        print(f'\nRunning {run["id"]}', flush=True)
        launch.main(args)
    report = workspace/'reports'/str(time.time_ns())
    summarize.main(['--manifest', str(manifest), '--root', str(output), '--output-dir', str(report)])
    export.main(['--root', str(output), '--output', str(report/'records.zip')])
    print(f'\nFinished. Results: {report}', flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', default='/workspace/cm-study')
    parser.add_argument('--hf-repo', default=DEFAULT_REPO)
    parser.add_argument('--hf-prefix', default='matched-30ep-v1')
    parser.add_argument('--runs', nargs='+')
    args = parser.parse_args(argv)
    run_study(args.work_dir, args.hf_repo, args.hf_prefix, args.runs)


if __name__ == '__main__':
    main()
