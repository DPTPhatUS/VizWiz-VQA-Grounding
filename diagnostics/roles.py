"""Validation-only role quality and mask-area diagnostics by native target size."""
import argparse
import json
import torch
from losses import membership_targets, membership_loss
from metrics import native_logits, overlap_counts, sample_metrics
from runtime import to_device
from artifacts import write_json, new_output
from diagnostics.common import inference_arguments, load_inference, size_bin


def role_counts(output,mask):
    targets=membership_targets(output,mask)
    valid=output['edge_valid'].expand(targets.shape[0],-1,-1)
    truth=targets.argmax(-1)[valid]
    probability=output['edge_logits'].float().softmax(-1)[valid]
    prediction=probability.argmax(-1)
    confusion=torch.bincount(truth*3+prediction,minlength=9).reshape(3,3)
    return {'confusion':confusion.cpu().tolist(),
            'predicted_counts':torch.bincount(prediction,minlength=3).cpu().tolist(),
            'probability_sum':probability.sum(0).cpu().tolist(),'edges':int(valid.sum()),
            'balanced_soft_loss':float(membership_loss(output,mask).detach())}


def aggregate(rows):
    if not rows:return {'num_samples':0}
    confusion=torch.tensor([r['roles']['confusion'] for r in rows],dtype=torch.int64).sum(0)
    support=confusion.sum(1);present=support>0
    recalls=confusion.diagonal().double()/support.clamp_min(1)
    edges=sum(r['roles']['edges'] for r in rows)
    area=[r['mask']['relative_area_error'] for r in rows if r['mask']['relative_area_error'] is not None]
    return {'num_samples':len(rows),'edges':edges,'confusion':confusion.tolist(),
            'balanced_accuracy':float(recalls[present].mean()) if present.any() else None,
            'role_recall':[float(recalls[i]) if present[i] else None for i in range(3)],
            'predicted_frequencies':[sum(r['roles']['predicted_counts'][i] for r in rows)/edges for i in range(3)],
            'mean_role_probability':[sum(r['roles']['probability_sum'][i] for r in rows)/edges for i in range(3)],
            'mean_balanced_soft_loss':sum(r['roles']['balanced_soft_loss'] for r in rows)/len(rows),
            'mean_relative_area_error':sum(area)/len(area) if area else None,
            'mean_iou':sum(r['mask']['iou'] for r in rows)/len(rows)}


@torch.inference_mode()
def analyze(model,loader,device):
    records={}
    for batch in loader:
        batch=to_device(batch,device);output=model(batch)
        if 'edge_logits' not in output:
            raise ValueError('Role diagnostics require CM with membership enabled')
        for i,(name,logits,truth) in enumerate(native_logits(output['logits'],batch)):
            single={key:output[key][i:i+1] if key in ('membership_logits','edge_logits') else output[key]
                    for key in ('membership_logits','edge_logits','edge_indices','edge_valid')}
            area=float(truth.float().mean())
            records[name]={'size_bin':size_bin(area),'ground_truth_area_fraction':area,
                'roles':role_counts(single,batch['mask'][i:i+1]),
                'mask':sample_metrics(overlap_counts(logits>0,truth))}
    return {'records':records,'all':aggregate(list(records.values())),
            'by_size':{key:aggregate([r for r in records.values() if r['size_bin']==key])
                       for key in ('tiny','small','medium','large','dominant')}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);inference_arguments(parser)
    args=parser.parse_args(argv)
    model,_,loader,device,metadata=load_inference(args,'val')
    if model.experiment_config['model']!='cm' or not model.experiment_config['membership_enabled']:
        raise ValueError('Role diagnostics require CM with membership enabled')
    out=new_output(args.output_dir)
    result=analyze(model,loader,device)
    write_json(out/'results.json',{**metadata,'roles':['include','exclude','neutral'],
        'hard_label_rule':'argmax of fractional targets; ties choose include, then exclude, then neutral',
        'balanced_accuracy_rule':'macro recall over supported target roles; pooled valid edges',**result})
    print(json.dumps(result['all']))


if __name__=='__main__':main()
