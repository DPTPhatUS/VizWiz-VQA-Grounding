"""Upload completed research results to Hugging Face."""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO = 'Last-Bullet/vizwiz-cm-aciids-2027'
RUN_FILES = ['best.pt','run.json','history.jsonl', *[
    pattern for split in ('val','test') for pattern in (
        f'{split}/metrics.json',f'{split}/logits.npz')]]


def upload_run(run_dir, repo_id, path):
    api = HfApi(token=True)
    api.create_repo(repo_id=repo_id,repo_type='model',private=False,exist_ok=True)
    api.upload_folder(repo_id=repo_id,repo_type='model',folder_path=str(run_dir),
                      path_in_repo=path,allow_patterns=RUN_FILES)
    print(f'Uploaded {Path(run_dir).name} to {repo_id}/{path}',flush=True)


def upload_summary(directory, repo_id, path):
    api = HfApi(token=True)
    api.upload_folder(repo_id=repo_id,repo_type='model',folder_path=str(directory),
                      path_in_repo=path,allow_patterns=['summary.json','runs.csv'])
    print(f'Uploaded study summary to {repo_id}/{path}',flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--repo-id',default=DEFAULT_REPO)
    parser.add_argument('--path-in-repo',required=True)
    args = parser.parse_args(argv)
    upload_run(args.run_dir,args.repo_id,args.path_in_repo)


if __name__ == '__main__':
    main()
