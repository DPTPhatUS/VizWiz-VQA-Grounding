"""Validation threshold calibration and fixed-operating-point test evaluation."""
import argparse
import json
import math
from pathlib import Path
import torch
from metrics import native_logits, overlap_counts, summarize_counts
from utils import to_device, write_json
from revision.common import inference_arguments, load_inference, new_output, plot_lines

THRESHOLDS = [i/100 for i in range(10,91,5)]


def select_threshold(report):
    if report['split'] != 'val' or report.get('limit', 0):
        raise ValueError('Selection requires the complete validation split')
    rows = report['thresholds']
    return min(rows, key=lambda r: (-r['metrics']['mean_iou'], round(abs(r['threshold']-.5),12), r['threshold']))['threshold']


@torch.inference_mode()
def sweep(model, loader, device, thresholds):
    records = {f'{p:.2f}':{} for p in thresholds}
    for batch in loader:
        batch = to_device(batch,device)
        for name,logits,truth in native_logits(model(batch)['logits'],batch):
            for p in thresholds:
                records[f'{p:.2f}'][name] = overlap_counts(logits > math.log(p/(1-p)),truth)
    return records


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('validation','test'))
    inference_arguments(parser)
    parser.add_argument('--selection',help='Frozen validation selection JSON, required for test')
    args=parser.parse_args(argv)
    model,_,loader,device,metadata=load_inference(args,'val' if args.action=='validation' else 'test')
    selected=None
    if args.action=='test':
        if not args.selection:
            raise ValueError('Test requires --selection from validation')
        selection=json.loads(Path(args.selection).read_text())
        if selection.get('split')!='val' or selection.get('limit') or selection.get('checkpoint_sha256')!=metadata['checkpoint_sha256']:
            raise ValueError('Selection must be from this checkpoint on complete validation')
        selected=selection['threshold']
        if selected not in THRESHOLDS:
            raise ValueError('Invalid selected probability threshold')
    elif args.selection:
        raise ValueError('--selection is an input for test only')
    out=new_output(args.output_dir)
    thresholds=THRESHOLDS if selected is None else sorted(set([.5,selected]))
    counts=sweep(model,loader,device,thresholds)
    rows=[{'threshold':p,'metrics':summarize_counts(counts[f'{p:.2f}'])} for p in thresholds]
    report={**metadata,'thresholds':rows,'counts':counts}
    if args.action=='validation' and not args.limit:
        selected=select_threshold(report)
        write_json(out/'selection.json',{'split':'val','limit':0,'threshold':selected,
            'checkpoint_sha256':metadata['checkpoint_sha256'],'objective':'maximum validation mIoU',
            'tie_break':'closest to 0.5, then lower threshold'})
    write_json(out/'results.json',report)
    if args.action=='validation':
        plot_lines(out/'curves.pdf',thresholds,{key:[r['metrics'][key] for r in rows]
            for key in ('mean_iou','acc50','macro_precision','macro_recall')},'Foreground probability threshold')
        plot_lines(out/'area_curves.pdf',thresholds,{key:[r['metrics'][key] for r in rows]
            for key in ('mean_area_ratio','mean_relative_area_error')},'Foreground probability threshold')
    print(json.dumps({'selected_threshold':selected,'metrics':rows}))


if __name__=='__main__':
    main()
