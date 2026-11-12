"""Download the dataset and launch a fresh single-GPU study."""
import argparse
from pathlib import Path
from zipfile import ZipFile

from huggingface_hub import hf_hub_download

from experiments import launch
from experiments.hub import DEFAULT_REPO
from runtime import DEFAULT_MANIFEST


def prepare_dataset(workspace):
    workspace = Path(workspace)
    archive = Path(hf_hub_download('Last-Bullet/VizWiz-VQA-Grounding','vizwiz.zip',
                   repo_type='dataset',local_dir=workspace/'downloads',force_download=True))
    with ZipFile(archive) as zipped:
        zipped.extractall(workspace/'data')
    archive.unlink()
    return workspace/'data/vizwiz'


def run_study(workspace, hf_repo, hf_prefix, run_ids=None, *, manifest=DEFAULT_MANIFEST,
              device='cuda', save_masks=False):
    workspace = Path(workspace)
    data = prepare_dataset(workspace)
    args = ['--manifest',str(manifest),'--data-root',str(data),
            '--output-root',str(workspace/'outputs'),'--processes','1','--device',device,
            '--hf-repo',hf_repo,'--hf-prefix',hf_prefix]
    if run_ids:
        args += ['--runs',*run_ids]
    if save_masks:
        args += ['--save-masks']
    return launch.main(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',default='/workspace/cm-study')
    parser.add_argument('--hf-repo',default=DEFAULT_REPO)
    parser.add_argument('--hf-prefix',default='matched-30ep-v1')
    parser.add_argument('--runs',nargs='+')
    parser.add_argument('--manifest',default=str(DEFAULT_MANIFEST))
    parser.add_argument('--save-masks',action='store_true')
    args = parser.parse_args(argv)
    run_study(args.work_dir,args.hf_repo,args.hf_prefix,args.runs,
              manifest=args.manifest,save_masks=args.save_masks)


if __name__ == '__main__':
    main()
