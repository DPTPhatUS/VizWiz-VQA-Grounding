"""Summarize selected-checkpoint test metrics and paired training-seed differences."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from artifacts import write_json, new_output
from runtime import DEFAULT_MANIFEST

METRICS=('mean_iou','acc50','macro_precision','macro_recall','severe_count')


def summary_stats(rows):
    return {key:{'mean':statistics.mean(r[key] for r in rows),
                 'sample_std':statistics.stdev(r[key] for r in rows) if len(rows)>1 else None}
            for key in METRICS}


def summarize(root,manifest):
    rows=[];missing=[];settings=None
    for run in manifest['runs']:
        # Accept one campaign or its parent containing separately launched runs.
        direct = Path(root)/run['id']/'test/metrics.json'
        candidates = ([direct] if direct.exists() else []) + list(Path(root).glob(f'*/{run["id"]}/test/metrics.json'))
        if not candidates:missing.append(run['id']);continue
        if len(candidates) != 1:
            raise ValueError(f'Multiple results for {run["id"]}; use a root with one result per run')
        path=candidates[0];folder=path.parent.parent
        result=json.loads(path.read_text())['summary'];record=json.loads((folder/'run.json').read_text())
        if record.get('study')!=manifest['name']:
            raise ValueError(f'Study identity mismatch: {run["id"]}')
        resolved=record['run_config'];common=dict(manifest['common'],**{k:v for k,v in run.items() if k!='id'})
        for key,value in common.items():
            if resolved.get(key)!=value:raise ValueError(f'Run setting mismatch: {run["id"]}: {key}')
        world=resolved['world_size']
        if settings is not None and settings!=world:raise ValueError('Mixed DDP world sizes in matched study')
        settings=world
        if result.get('dataset')!='test' or result.get('metric_resolution')!='original':raise ValueError('Expected native-grid test evaluation')
        expected={'run':resolved['run_id'],'epoch':record['selection']['epoch']}
        if result['checkpoint']!=expected:
            raise ValueError('Test metrics do not match best checkpoint')
        label=run['id'].rsplit('_s',1)[0]
        rows.append({'run_id':run['id'],'configuration':label,'seed':run['seed'],
                     **{key:result[key] for key in METRICS},'source_commit':record['provenance']['git_commit']})
    groups={name:summary_stats([r for r in rows if r['configuration']==name]) for name in sorted({r['configuration'] for r in rows})}
    ftc={r['seed']:r for r in rows if r['configuration']=='ftc'}
    cm={r['seed']:r for r in rows if r['configuration']=='cm_separate'}
    paired=[{'seed':seed,**{key:cm[seed][key]-ftc[seed][key] for key in METRICS}} for seed in sorted(ftc.keys()&cm.keys())]
    controls = {}
    for label in ('cm_affinity', 'cm_two_role', 'context_only', 'membership_only', 'cm_noaux', 'ftc_capacity'):
        control = {r['seed']:r for r in rows if r['configuration']==label}
        pairs = [{'seed':seed, **{key:cm[seed][key]-control[seed][key] for key in METRICS}}
                 for seed in sorted(cm.keys()&control.keys())]
        if pairs:
            controls[label] = {'pairs':pairs, 'num_pairs':len(pairs), 'summary':summary_stats(pairs)}
    return {'cohort':manifest['name'],'runs':rows,'missing_runs':missing,'by_configuration':groups,
            'paired_cm_minus_ftc':paired,'paired_summary':summary_stats(paired) if paired else {},
            'paired_cm_minus_controls':controls,
            'uncertainty':'sample standard deviation across training seeds; not an image-level confidence interval'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--manifest',default=str(DEFAULT_MANIFEST))
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args(argv)
    result=summarize(args.root,json.loads(Path(args.manifest).read_text()))
    out=new_output(args.output_dir);write_json(out/'summary.json',result)
    with (out/'runs.csv').open('w',newline='') as stream:
        keys=['run_id','configuration','seed',*METRICS,'source_commit']
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(result['runs'])
    print(json.dumps({'completed':len(result['runs']),'missing':result['missing_runs']}))


if __name__=='__main__':main()
