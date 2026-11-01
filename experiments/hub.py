"""Atomic checkpoint backups and hash-verified restoration through Hugging Face."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import time

from checkpoints import checkpoint_metadata
from artifacts import sha256_file, write_json

DEFAULT_REPO = 'Last-Bullet/vizwiz-cm-aciids-2027'
RECORDS = ('run.json', 'history.jsonl', 'manifest_run.json', 'test/metrics.json',
           'latest.pt.sha256.json', 'best.pt.sha256.json')
CHECKPOINTS = ('latest.pt', 'best.pt')


def hub_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '\\' in value or any(p in ('', '.', '..') for p in value.split('/')):
        raise ValueError('Hub paths must be relative names without empty, dot, or parent components')
    return path.as_posix()


def _api():
    from huggingface_hub import HfApi
    # Tokens come from HF_TOKEN or `hf auth login`, never command-line arguments or run records.
    return HfApi(token=True)


def ensure_repository(repo_id, private=False, api=None):
    api = api or _api()
    api.create_repo(repo_id=repo_id, repo_type='model', private=private, exist_ok=True)
    api.auth_check(repo_id, repo_type='model', write=True)
    info = api.repo_info(repo_id, repo_type='model')
    if bool(info.private) != private:
        raise ValueError('Repository visibility differs from the requested public/private setting')
    return api, info.sha


def remote_index(api, repo_id, path, revision):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    try:
        downloaded = hf_hub_download(repo_id, f'{path}/backup.json', repo_type='model',
                                     revision=revision, token=True)
    except EntryNotFoundError:
        existing = api.list_repo_files(repo_id, repo_type='model', revision=revision)
        if any(name.startswith(path + '/') for name in existing):
            raise ValueError(f'Remote path already contains files without a backup index: {path}')
        return None
    index = json.loads(Path(downloaded).read_text())
    if index.get('backup_version') != 1 or index.get('path') != path:
        raise ValueError('Unrecognized Hub backup index')
    return index


def check_destination(repo_id, path, run_uuid=None, private=False, api=None):
    path = hub_path(path)
    api, revision = ensure_repository(repo_id, private, api)
    previous = remote_index(api, repo_id, path, revision)
    if previous is not None and previous['run_uuid'] != run_uuid:
        raise ValueError(f'Hub destination belongs to another run: {path}; restore it or choose a new prefix')
    return api, previous, revision


def backup_plan(run_dir, path):
    """Allow only the two checkpoints and experiment records; never upload dataset images."""
    root = Path(run_dir)
    path = hub_path(path)
    record = json.loads((root/'run.json').read_text())
    latest, best = (checkpoint_metadata(root/name) for name in CHECKPOINTS)
    run_uuid = record['run_config']['run_uuid']
    for saved in (latest, best):
        if saved['run_config']['run_uuid'] != run_uuid or saved['experiment_config'] != record['experiment']:
            raise ValueError('Run/checkpoint identity mismatch')
    if latest['best_epoch'] != best['epoch'] or best['epoch'] > latest['epoch']:
        raise ValueError('latest.pt and best.pt do not describe the same selection history')
    files = {}
    for name in (*CHECKPOINTS, *RECORDS):
        source = root/name
        if source.is_symlink():
            raise ValueError(f'Backup file cannot be a symlink: {name}')
        if source.is_file():
            files[name] = {'sha256':sha256_file(source), 'bytes':source.stat().st_size}
    for name in CHECKPOINTS:
        sidecar = root/(name+'.sha256.json')
        if sidecar.exists() and json.loads(sidecar.read_text())['sha256'] != files[name]['sha256']:
            raise ValueError(f'Checkpoint hash sidecar mismatch: {name}')
    metrics = root/'test/metrics.json'
    if metrics.exists() and json.loads(metrics.read_text())['summary']['checkpoint_sha256'] != files['best.pt']['sha256']:
        raise ValueError('Test metrics do not match the selected checkpoint')
    return {'backup_version':1, 'path':path, 'run_uuid':run_uuid,
            'epoch':latest['epoch'], 'best_epoch':best['epoch'],
            'best_validation_miou':latest['best_validation_miou'],
            'experiment_config':latest['experiment_config'], 'files':files}


def upload_run(run_dir, repo_id, path, private=False, dry_run=False, api=None):
    from huggingface_hub import CommitOperationAdd
    root = Path(run_dir)
    index = backup_plan(root, path)
    if dry_run:
        return {'repo_id':repo_id, **index}
    # One commit publishes latest + best + their index together. The launcher calls
    # this after training/evaluation exit, so checkpoint files are no longer changing.
    for attempt in range(3):
        api, previous, parent = check_destination(repo_id, index['path'], index['run_uuid'], private, api)
        if previous is not None:
            if previous['epoch'] > index['epoch']:
                raise ValueError('Remote backup is newer than local progress; restore before resuming')
            if previous['epoch'] == index['epoch'] and any(
                previous['files'][name]['sha256'] != index['files'][name]['sha256'] for name in CHECKPOINTS):
                raise ValueError('Different checkpoint contents for the same run and epoch')
            if previous == index:
                result = {'repo_id':repo_id, 'path':index['path'], 'epoch':index['epoch'],
                          'commit':parent, 'unchanged':True}
                write_json(root/'hub_status.json', result)
                return result
        operations = [CommitOperationAdd(path_in_repo=f'{index["path"]}/{name}', path_or_fileobj=str(root/name))
                      for name in index['files']]
        operations.append(CommitOperationAdd(path_in_repo=f'{index["path"]}/backup.json',
                                            path_or_fileobj=json.dumps(index, indent=2).encode()))
        try:
            commit = api.create_commit(repo_id=repo_id, repo_type='model', operations=operations,
                                       parent_commit=parent,
                                       commit_message=f'{index["path"]}: epoch {index["epoch"]}, best {index["best_epoch"]}')
            break
        except Exception as exc:
            if attempt == 2:
                write_json(root/'hub_status.json', {'repo_id':repo_id, 'path':index['path'],
                           'epoch':index['epoch'], 'error':type(exc).__name__})
                raise
            time.sleep(2 ** attempt)
    result = {'repo_id':repo_id, 'path':index['path'], 'epoch':index['epoch'],
              'commit':commit.oid, 'time_utc':datetime.now(timezone.utc).isoformat()}
    write_json(root/'hub_status.json', result)
    return result


def download_run(repo_id, path, run_dir, revision='main', api=None):
    """Pin one commit and verify every file before making the local run visible."""
    from huggingface_hub import hf_hub_download
    root = Path(run_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f'Refusing to overwrite local run: {root}')
    path = hub_path(path)
    api = api or _api()
    commit = api.repo_info(repo_id, repo_type='model', revision=revision).sha
    index = remote_index(api, repo_id, path, commit)
    if index is None:
        raise ValueError(f'No backup at {path}')
    if not set(CHECKPOINTS).issubset(index['files']):
        raise ValueError('Backup is missing resumable/selected checkpoints')
    allowed = set(CHECKPOINTS + RECORDS)
    if not set(index['files']).issubset(allowed):
        raise ValueError('Backup contains unexpected file paths')
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.hub-restore-', dir=root.parent) as tmp:
        staging = Path(tmp)
        for name, expected in index['files'].items():
            downloaded = Path(hf_hub_download(repo_id, f'{path}/{name}', repo_type='model', revision=commit,
                                              local_dir=staging, token=True))
            if downloaded.stat().st_size != expected['bytes'] or sha256_file(downloaded) != expected['sha256']:
                raise ValueError(f'Corrupt download: {name}')
        restored = staging/path
        if backup_plan(restored, path) != index:
            raise ValueError('Restored checkpoint/index mismatch')
        write_json(restored/'hub_status.json', {'repo_id':repo_id, 'path':path, 'commit':commit,
                                               'epoch':index['epoch'], 'restored':True})
        if root.exists():
            root.rmdir()
        os.replace(restored, root)
    return {'run_dir':str(root), 'commit':commit, 'epoch':index['epoch']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('upload', 'download'):
        command = sub.add_parser(action)
        command.add_argument('--repo-id', default=DEFAULT_REPO)
        command.add_argument('--path-in-repo', required=True)
        command.add_argument('--run-dir', required=True)
        if action == 'upload':
            command.add_argument('--private', action='store_true')
            command.add_argument('--dry-run', action='store_true')
        else:
            command.add_argument('--revision', default='main', help='Branch or immutable commit SHA')
    args = parser.parse_args(argv)
    if args.action == 'upload':
        result = upload_run(args.run_dir, args.repo_id, args.path_in_repo, args.private, args.dry_run)
    else:
        result = download_run(args.repo_id, args.path_in_repo, args.run_dir, args.revision)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
