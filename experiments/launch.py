"""Train a fresh campaign, evaluate selected models, and upload research results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from artifacts import write_json
from runtime import DEFAULT_MANIFEST, PROJECT_ROOT


def run_arguments(common, run, data_root, out, device):
    settings = dict(common, **{key:value for key,value in run.items() if key != 'id'})
    settings.update(data_root=str(data_root), output_dir=str(out), device=device)
    args = []
    for key,value in settings.items():
        flag = '--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:
                args.append(flag)
        elif value is not None:
            args.extend([flag,str(value)])
    return args


def commands(manifest, run, data_root, out, device, processes, save_masks=False):
    flags = run_arguments(manifest['common'],run,data_root,out,device)
    train = ([sys.executable,'train.py'] if processes == 1 else
             [sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={processes}','train.py'])
    evaluations = [[sys.executable,'eval.py','--checkpoint',str(out/'best.pt'),
        '--data-root',str(data_root),'--dataset',split,'--device',device,
        '--batch-size',str(manifest['common']['batch_size']),
        '--num-workers',str(manifest['common'].get('num_workers',2)),
        '--output-dir',str(out/split)] for split in ('val','test')]
    if save_masks:
        for command in evaluations:
            command.append('--save-masks')
    return [[*train,*flags], *evaluations]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default=str(DEFAULT_MANIFEST))
    parser.add_argument('--runs',nargs='+',help='Run IDs; omitted means all models sequentially')
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--output-root',default='outputs')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--processes',type=int,default=1)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--save-masks',action='store_true')
    parser.add_argument('--no-hub',action='store_true')
    parser.add_argument('--hf-repo')
    parser.add_argument('--hf-prefix')
    args = parser.parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text())
    by_id = {run['id']:run for run in manifest['runs']}
    runs = [by_id[name] for name in args.runs] if args.runs else manifest['runs']
    campaign = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(args.output_root).resolve()/campaign
    data_root = Path(args.data_root).resolve()
    hub = None if args.no_hub else manifest.get('hub')
    if not args.no_hub and (args.hf_repo or args.hf_prefix):
        from experiments.hub import DEFAULT_REPO
        hub = {'repo_id':args.hf_repo or (hub or {}).get('repo_id',DEFAULT_REPO),
               'prefix':args.hf_prefix or (hub or {}).get('prefix',manifest['name'])}
    if not args.dry_run:
        output.mkdir(parents=True)
    print(f'Campaign: {output}',flush=True)
    for run in runs:
        folder = output/run['id']
        steps = commands(manifest,run,data_root,folder,args.device,args.processes,args.save_masks)
        print(json.dumps({'run':run['id'],'commands':steps}),flush=True)
        if args.dry_run:
            continue
        subprocess.run(steps[0],cwd=PROJECT_ROOT,check=True)
        record = json.loads((folder/'run.json').read_text())
        record['study'] = manifest['name']
        write_json(folder/'run.json',record)
        for command in steps[1:]:
            subprocess.run(command,cwd=PROJECT_ROOT,check=True)
        if hub:
            from experiments.hub import upload_run
            upload_run(folder,hub['repo_id'],f'{hub["prefix"]}/{campaign}/{run["id"]}')
    if not args.dry_run:
        from experiments import summarize
        summarize.main(['--manifest',args.manifest,'--root',str(output),
                        '--output-dir',str(output/'summary')])
        if hub:
            from experiments.hub import upload_summary
            upload_summary(output/'summary',hub['repo_id'],f'{hub["prefix"]}/{campaign}/summary')
    return output


if __name__ == '__main__':
    main()
