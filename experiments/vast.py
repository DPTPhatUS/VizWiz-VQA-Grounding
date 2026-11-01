"""Dataset preparation, GPU checks, and resumable jobs used by vast.ipynb."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
from zipfile import ZipFile

from artifacts import sha256_file, write_json

GIB = 2**30
SPLITS = ('train', 'val', 'test')


def dataset_root(directory):
    """Find one complete dataset even when the archive wraps it in a top folder."""
    candidates = []
    for annotation in Path(directory).rglob('train_grounding.json'):
        root = annotation.parent
        if all((root/f'{split}_grounding.json').is_file() and (root/split).is_dir()
               and (root/'binary_masks_png'/split).is_dir() for split in SPLITS):
            candidates.append(root)
    if len(candidates) != 1:
        raise ValueError(f'Expected one complete dataset root, found {len(candidates)} in {directory}')
    return candidates[0]


def extract_dataset(archive, destination, source):
    """Validate ZIP paths, extract on the destination filesystem, then publish atomically."""
    destination = Path(destination)
    marker = destination/'.dataset-source.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['source'] != source:
            raise ValueError('Dataset source changed; choose a new extraction directory')
        root = dataset_root(destination)
        return {'data_root':str(root), **saved, 'reused':True}
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f'Refusing to extract over existing data: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = Path(archive)
    with ZipFile(archive) as zipped:
        for info in zipped.infolist():
            path = PurePosixPath(info.filename)
            if (path.is_absolute() or '..' in path.parts or '\\' in info.filename
                    or (path.parts and ':' in path.parts[0])
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError(f'Unsafe ZIP member: {info.filename}')
        needed = sum(info.file_size for info in zipped.infolist())
        if shutil.disk_usage(destination.parent).free < needed + 2*GIB:
            raise ValueError('Insufficient disk space for dataset extraction plus 2 GiB reserve')
        with tempfile.TemporaryDirectory(prefix='.vizwiz-extract-', dir=destination.parent) as temporary:
            staging = Path(temporary)
            zipped.extractall(staging)
            root = dataset_root(staging)
            record = {'source':source, 'archive_sha256':sha256_file(archive),
                      'uncompressed_bytes':needed, 'relative_data_root':str(root.relative_to(staging))}
            write_json(staging/'.dataset-source.json', record)
            if destination.exists():
                destination.rmdir()
            os.replace(staging, destination)
    return {'data_root':str(destination/record['relative_data_root']), **record, 'reused':False}


def prepare_dataset(repo_id, filename, revision, download_dir, extract_dir):
    from huggingface_hub import HfApi, hf_hub_download
    marker = Path(extract_dir)/'.dataset-source.json'
    if marker.exists():
        previous = json.loads(marker.read_text())
        source = previous['source']
        if (source['repo_id'],source['filename'],source['requested_revision']) != (repo_id,filename,revision):
            raise ValueError('Dataset settings changed; choose a new extraction directory')
        # Reuse the pinned extraction, rather than silently following a moving main branch.
        return {'data_root':str(dataset_root(extract_dir)), **previous, 'reused':True}
    commit = HfApi().dataset_info(repo_id, revision=revision).sha
    archive = hf_hub_download(repo_id,filename,repo_type='dataset',revision=commit,local_dir=download_dir)
    source = {'repo_id':repo_id, 'filename':filename, 'requested_revision':revision, 'commit':commit}
    return extract_dataset(archive,extract_dir,source)


def check_access(repo_id,dataset_repo,filename,revision='main'):
    from huggingface_hub import HfApi,get_hf_file_metadata,hf_hub_url
    from experiments.hub import ensure_repository
    ensure_repository(repo_id)
    commit = HfApi().dataset_info(dataset_repo,revision=revision).sha
    metadata = get_hf_file_metadata(hf_hub_url(dataset_repo,filename,repo_type='dataset',revision=commit),token=True)
    return {'checkpoint_repository':repo_id,'public':True,'write_access':True,
            'dataset_repository':dataset_repo,'dataset_commit':commit,'archive_bytes':metadata.size}


def available_ram():
    """Respect container memory limits when they are below host MemAvailable."""
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value = line.split(':',1)
        memory[key] = int(value.split()[0])*1024
    available = memory['MemAvailable']
    for limit_file,used_file in (
        ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
        ('/sys/fs/cgroup/memory.high','/sys/fs/cgroup/memory.current'),
        ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes'),
    ):
        if Path(limit_file).exists() and Path(used_file).exists():
            limit = Path(limit_file).read_text().strip()
            if limit.isdigit():
                available = min(available,max(0,int(limit)-int(Path(used_file).read_text())))
    return available


def hardware_report(workspace, processes=1, min_vram_gib=40, min_ram_gib=24, min_disk_gib=130):
    import torch
    if processes < 1 or 4 % processes:
        raise ValueError('Process count must divide global batch 4')
    driver = subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
                            check=True,capture_output=True,text=True).stdout.splitlines()
    if not driver or min(int(value.split('.')[0]) for value in driver) < 580:
        raise ValueError('The locked CUDA 13 build requires NVIDIA driver 580 or newer')
    if not torch.cuda.is_available() or torch.cuda.device_count() < processes:
        raise ValueError(f'Need {processes} working CUDA device(s); check the instance GPU/driver')
    gpus = []
    for index in range(processes):
        free,total = torch.cuda.mem_get_info(index)
        if free/GIB < min_vram_gib:
            raise ValueError(f'GPU {index} has {free/GIB:.1f} GiB free; configured guard is {min_vram_gib} GiB')
        gpus.append({'index':index,'name':torch.cuda.get_device_name(index),
                     'total_gib':total/GIB,'free_gib':free/GIB,'driver':driver[min(index,len(driver)-1)]})
    ram,disk = available_ram()/GIB,shutil.disk_usage(workspace).free/GIB
    if ram < min_ram_gib or disk < min_disk_gib:
        raise ValueError(f'Available RAM/disk = {ram:.1f}/{disk:.1f} GiB; need {min_ram_gib}/{min_disk_gib}')
    return {'gpus':gpus,'available_ram_gib':ram,'free_disk_gib':disk,
            'torch':torch.__version__,'torch_cuda':torch.version.cuda,'processes':processes}


def memory_probe(data_root,processes=1,tiny=False,device='cuda'):
    """Exercise the full CM-Separate recipe twice, including allocated Adam state."""
    import torch
    from dataset import VizWizGroundingDataset,collate_samples
    from losses import SupervisedObjective
    from models.factory import build_model
    from train import seed_everything
    from runtime import to_device
    if processes < 1 or 4 % processes:
        raise ValueError('Process count must divide global batch 4')
    if device != 'cuda' and not tiny:
        raise ValueError('CPU probes are reserved for tiny offline checks')
    torch.set_num_threads(2)
    dataset = VizWizGroundingDataset(data_root,'train',32 if tiny else 336)
    reports = []
    for index in range(processes if device == 'cuda' else 1):
        target = torch.device('cuda',index) if device == 'cuda' else torch.device('cpu')
        seed_everything(42)
        model = build_model('cm',tiny=tiny,connection_mode='separate').to(target).train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params,lr=1e-5)
        scaler = torch.amp.GradScaler('cuda',enabled=device=='cuda')
        batch = to_device(collate_samples([dataset[i%len(dataset)] for i in range(4//processes)]),target)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats(target)
        successful,attempts = 0,0
        while successful < 2 and attempts < 12:
            attempts += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(target.type,enabled=device=='cuda'):
                loss,_ = SupervisedObjective(.1)(model,batch)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss in full-batch memory probe')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params,1.)
            scale = scaler.get_scale()
            scaler.step(optimizer);scaler.update()
            successful += scaler.get_scale() >= scale
        if successful != 2 or not optimizer.state:
            raise ValueError('Memory probe did not complete two optimizer steps; inspect AMP gradients')
        if device == 'cuda':
            torch.cuda.synchronize(target)
        reports.append({'device':str(target),'microbatch':4//processes,'steps':successful,'attempts':attempts,'loss':float(loss.detach()),
            'peak_allocated_gib':torch.cuda.max_memory_allocated(target)/GIB if device=='cuda' else None,
            'peak_reserved_gib':torch.cuda.max_memory_reserved(target)/GIB if device=='cuda' else None})
        del batch,model,params,optimizer,scaler,loss
        if device == 'cuda':
            torch.cuda.empty_cache()
    return {'global_batch':4,'probe_model':'CM-Separate','tiny':tiny,'measurements':reports}


def selected_runs(manifest,run_ids):
    available = {run['id'] for run in manifest['runs']}
    if run_ids and not set(run_ids) <= available:
        raise ValueError(f'Unknown run IDs: {sorted(set(run_ids)-available)}')
    if run_ids and len(set(run_ids)) != len(run_ids):
        raise ValueError('Duplicate run IDs')
    return [run for run in manifest['runs'] if not run_ids or run['id'] in run_ids]


def execute_study(config):
    """Use the canonical launcher for each fresh/resumed run, then summarize and export."""
    from checkpoints import checkpoint_metadata
    from experiments import launch,summarize,export
    from experiments.hub import ensure_repository,remote_index,download_run
    manifest = json.loads(Path(config['manifest']).read_text())
    runs = selected_runs(manifest,config.get('runs'))
    output = Path(config['output_root'])
    session = Path(config['session_dir'])
    session.mkdir(parents=True,exist_ok=True)
    for run in runs:
        folder = output/run['id']
        has_files = folder.exists() and any(folder.iterdir())
        if has_files and not (folder/'latest.pt').is_file():
            raise ValueError(f'{folder} contains files but no resumable checkpoint; inspect before retrying')
        if not has_files and config.get('auto_restore',True) and config.get('upload',True):
            api,commit = ensure_repository(config['hf_repo'])
            path = f'{config["hf_prefix"]}/{run["id"]}'
            if remote_index(api,config['hf_repo'],path,commit) is not None:
                print(f'Restoring {run["id"]} from Hugging Face',flush=True)
                download_run(config['hf_repo'],path,folder,api=api)
        resume = (folder/'latest.pt').is_file()
        previous = checkpoint_metadata(folder/'latest.pt') if resume else None
        if resume and previous['run_config']['world_size'] != config['processes']:
            raise ValueError(f'{run["id"]}: resume must keep the original process count')
        # Allow space for the resumable checkpoint, best weights, and atomic-save temporary file.
        output.mkdir(parents=True,exist_ok=True)
        will_train = not resume or previous['epoch'] < manifest['common']['num_epochs']
        if will_train and config.get('device','cuda') == 'cuda' and shutil.disk_usage(output).free < 14*GIB:
            raise ValueError('Need at least 14 GiB free before the next run; retain/export existing artifacts')
        args = ['--manifest',config['manifest'],'--data-root',config['data_root'],
                '--output-root',str(output),'--processes',str(config['processes']),
                '--device',config.get('device','cuda'),'--runs',run['id']]
        if config.get('upload',True):
            args += ['--hf-repo',config['hf_repo'],'--hf-prefix',config['hf_prefix']]
        else:
            args += ['--no-hub']
        if resume:
            args += ['--resume']
        print(f'\n{"Resuming" if resume else "Starting"} {run["id"]}',flush=True)
        launch.main(args)
    summarize.main(['--manifest',config['manifest'],'--root',str(output),
                    '--output-dir',str(session/'summary')])
    export.main(['--root',str(output),'--output',str(session/'records.zip')])
    return {'completed_run_ids':[run['id'] for run in runs], 'summary':str(session/'summary/summary.json'),
            'records_archive':str(session/'records.zip')}


def worker(config_path):
    config = json.loads(Path(config_path).read_text())
    status = Path(config['session_dir'])/'status.json'
    record = {'session_id':config['session_id'],'pid':os.getpid(),
              'started_utc':datetime.now(timezone.utc).isoformat(),'state':'running'}
    write_json(status,record)
    try:
        result = execute_study(config)
    except BaseException as exc:
        write_json(status,{**record,'state':'failed','error_type':type(exc).__name__,'error':str(exc)})
        raise
    write_json(status,{**record,'state':'complete',**result,'finished_utc':datetime.now(timezone.utc).isoformat()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    auth = commands.add_parser('auth')
    auth.add_argument('--repo-id',required=True);auth.add_argument('--dataset-repo',required=True)
    auth.add_argument('--filename',default='vizwiz.zip');auth.add_argument('--revision',default='main')
    auth.add_argument('--report',required=True)
    data = commands.add_parser('data')
    data.add_argument('--repo-id',required=True);data.add_argument('--filename',default='vizwiz.zip')
    data.add_argument('--revision',default='main');data.add_argument('--download-dir',required=True)
    data.add_argument('--extract-dir',required=True);data.add_argument('--report',required=True)
    check = commands.add_parser('hardware')
    check.add_argument('--workspace',required=True);check.add_argument('--processes',type=int,default=1)
    check.add_argument('--min-vram-gib',type=float,default=40);check.add_argument('--min-ram-gib',type=float,default=24)
    check.add_argument('--min-disk-gib',type=float,default=130);check.add_argument('--report',required=True)
    probe = commands.add_parser('probe')
    probe.add_argument('--data-root',required=True);probe.add_argument('--processes',type=int,default=1)
    probe.add_argument('--tiny',action='store_true');probe.add_argument('--device',default='cuda')
    probe.add_argument('--report',required=True)
    job = commands.add_parser('worker');job.add_argument('--config',required=True)
    args = parser.parse_args(argv)
    if args.command == 'worker':
        worker(args.config);return
    if args.command == 'auth':
        result = check_access(args.repo_id,args.dataset_repo,args.filename,args.revision)
    elif args.command == 'data':
        result = prepare_dataset(args.repo_id,args.filename,args.revision,args.download_dir,args.extract_dir)
    elif args.command == 'hardware':
        result = hardware_report(args.workspace,args.processes,args.min_vram_gib,args.min_ram_gib,args.min_disk_gib)
    else:
        result = memory_probe(args.data_root,args.processes,args.tiny,args.device)
    write_json(args.report,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
