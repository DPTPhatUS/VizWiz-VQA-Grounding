"""Launch selected matched runs using the same commands on Linux and Kaggle."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from artifacts import write_json
from runtime import DEFAULT_MANIFEST, PROJECT_ROOT


def run_arguments(common,run,data_root,out,device):
    settings=dict(common,**{k:v for k,v in run.items() if k!='id'})
    settings.update(data_root=str(data_root),output_dir=str(out),device=device)
    args=[]
    for key,value in settings.items():
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:args.append(flag)
        elif value is not None:args.extend([flag,str(value)])
    return args


def commands(manifest,run,data_root,out,device,processes,resume=False):
    flags=run_arguments(manifest['common'],run,data_root,out,device)
    if resume:flags+=['--resume-checkpoint',str(out/'latest.pt')]
    train=([sys.executable,'train.py'] if processes==1 else
           [sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={processes}','train.py'])
    return [[sys.executable,'preflight.py',*flags], [*train,*flags],
            [sys.executable,'eval.py','--checkpoint',str(out/'best.pt'),'--data-root',str(data_root),
             '--dataset','test','--device',device,'--num-workers',str(manifest['common'].get('num_workers',2)),
             '--output-dir',str(out/'test')]]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default=str(DEFAULT_MANIFEST))
    parser.add_argument('--runs',nargs='+',help='Run IDs; omitted means all runs sequentially')
    parser.add_argument('--data-root',required=True);parser.add_argument('--output-root',default='outputs/revision')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--processes',type=int,default=1)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--no-hub',action='store_true',help='Disable the manifest\'s Hugging Face backups')
    parser.add_argument('--hf-repo',help='Override the manifest\'s Hub repository')
    parser.add_argument('--hf-prefix',help='Override the study path inside the Hub repository')
    args=parser.parse_args(argv)
    manifest=json.loads(Path(args.manifest).read_text());ids=[r['id'] for r in manifest['runs']]
    hub = dict(manifest.get('hub',{})) if not args.no_hub else None
    if args.no_hub and (args.hf_repo or args.hf_prefix):
        raise ValueError('Cannot combine --no-hub with Hub destination options')
    if (args.hf_repo or args.hf_prefix) and not hub:
        from experiments.hub import DEFAULT_REPO
        hub = {'repo_id':DEFAULT_REPO,'prefix':manifest['name'],'private':False}
    if args.hf_repo:
        hub['repo_id'] = args.hf_repo
    if args.hf_prefix:
        hub['prefix'] = args.hf_prefix
    if len(set(ids))!=len(ids) or any(Path(i).name!=i or i in ('.','..') for i in ids):
        raise ValueError('Run IDs must be unique simple directory names')
    if args.processes<1 or manifest['common']['batch_size']%args.processes:
        raise ValueError('Process count must divide the global batch')
    if args.runs and not set(args.runs)<=set(ids):raise ValueError('Unknown run IDs')
    data_root=Path(args.data_root).resolve();output_root=Path(args.output_root).resolve()
    for run in manifest['runs']:
        if args.runs and run['id'] not in args.runs:continue
        out=output_root/run['id']
        if not args.dry_run:
            if args.resume and not (out/'latest.pt').is_file():raise ValueError(f'Missing resumable checkpoint: {out}')
            if not args.resume and out.exists() and any(out.iterdir()):raise ValueError(f'Refusing to reuse {out}')
        steps=commands(manifest,run,data_root,out,args.device,args.processes,args.resume)
        destination = None
        if hub:
            from experiments.hub import hub_path
            destination = {'repo_id':hub['repo_id'], 'path':hub_path(f'{hub["prefix"]}/{run["id"]}'),
                           'private':hub.get('private',False), 'when':'after-training-and-test'}
        print(json.dumps({'run':run['id'],'commands':steps,'huggingface':destination}),flush=True)
        if args.dry_run:continue
        complete=False
        if args.resume:
            from checkpoints import checkpoint_metadata
            saved=checkpoint_metadata(out/'latest.pt')
            desired=dict(manifest['common'],**{k:v for k,v in run.items() if k!='id'})
            for key,value in desired.items():
                if saved['run_config'].get(key)!=value:
                    raise ValueError(f'Resume manifest conflicts on {key}')
            complete=saved['epoch']==manifest['common']['num_epochs']
        if hub:
            from experiments.hub import check_destination
            _, previous, _ = check_destination(hub['repo_id'],destination['path'],
                saved['run_config']['run_uuid'] if args.resume else None,hub.get('private',False))
            if previous is not None and previous['epoch'] > saved['epoch']:
                raise ValueError('Hub backup is newer than local progress; download before resuming')
        if not complete:
            for command in steps[:2]:subprocess.run(command,cwd=PROJECT_ROOT,check=True)
        if not (out/'test/metrics.json').exists():subprocess.run(steps[2],cwd=PROJECT_ROOT,check=True)
        else:
            from artifacts import sha256_file
            previous=json.loads((out/'test/metrics.json').read_text())
            if previous['summary'].get('checkpoint_sha256')!=sha256_file(out/'best.pt'):
                raise ValueError('Existing test evaluation is for a different checkpoint')
        write_json(out/'manifest_run.json',{'cohort':manifest['name'],'run':run,'common':manifest['common']})
        if hub:
            from experiments.hub import upload_run
            result=upload_run(out,hub['repo_id'],f'{hub["prefix"]}/{run["id"]}',hub.get('private',False))
            print(json.dumps({'huggingface_final':result}),flush=True)


if __name__=='__main__':main()
