"""Question interventions: mask sensitivity, not counterfactual grounding accuracy."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import torch
from dataset import collate_samples
from metrics import native_logits, overlap_counts, sample_metrics
from utils import to_device, write_json
from revision.common import inference_arguments, load_inference, new_output

CONDITIONS=('original','repeat','shuffled','generic','empty')


def normalized(text):
    return ' '.join(text.casefold().split())


def donor_mapping(dataset,seed):
    names=sorted(dataset.names)
    if len(names)<2:
        raise ValueError('Shuffling requires at least two images')
    donors=list(names);rng=random.Random(seed)
    for i in range(len(donors)-1,0,-1):
        j=rng.randrange(i);donors[i],donors[j]=donors[j],donors[i]
    questions={name:str(dataset.records[name]['question']) for name in names}
    fingerprint=hashlib.sha256(json.dumps(questions,sort_keys=True).encode()).hexdigest()
    return {'seed':seed,'question_sha256':fingerprint,'donors':dict(zip(names,donors))}


def get_donors(dataset,path,seed):
    expected=donor_mapping(dataset,seed)
    path=Path(path)
    if path.exists():
        saved=json.loads(path.read_text())
        if saved!=expected:
            raise ValueError('Donor mapping does not match this split and seed')
        return saved
    write_json(path,expected)
    return expected


def condition_texts(dataset,names,mapping,condition):
    original=[str(dataset.records[n]['question']) for n in names]
    if condition=='shuffled':
        original=[str(dataset.records[mapping['donors'][n]]['question']) for n in names]
    if condition=='generic':return ['Q: What is this?']*len(names)
    if condition=='empty':return ['Q:']*len(names)
    return [f'Q: {q}' for q in original]


def intervention(model,batch,texts,features=None):
    changed=dict(batch,text=texts)
    return model(changed) if features is None else model.decode(features,changed['text'])


@torch.inference_mode()
def analyze(model,loader,dataset,mapping,device,cache=True):
    records={};checked=False
    for batch in loader:
        batch=to_device(batch,device);names=batch['filename']
        features=model.encode_image(batch['image']) if cache else None
        outputs={condition:intervention(model,batch,condition_texts(dataset,names,mapping,condition),features)
                 for condition in CONDITIONS}
        if cache and not checked:
            torch.testing.assert_close(outputs['original']['logits'],model(batch)['logits'],rtol=1e-5,atol=1e-6)
            checked=True
        native={c:list(native_logits(outputs[c]['logits'],batch)) for c in CONDITIONS}
        for i,name in enumerate(names):
            ref=native['original'][i][1]>0;truth=native['original'][i][2]
            donor=mapping['donors'][name]
            identical=normalized(str(dataset.records[name]['question']))==normalized(str(dataset.records[donor]['question']))
            row={'donor':donor,'identical_question':identical,'conditions':{}}
            for c in CONDITIONS:
                pred=native[c][i][1]>0
                sm=sample_metrics(overlap_counts(pred,truth));agreement=sample_metrics(overlap_counts(pred,ref))['iou']
                row['conditions'][c]={'agreement_iou':agreement,'changed_fraction':float((pred^ref).float().mean()),
                    'area_fraction_change':float(pred.float().mean()-ref.float().mean()),
                    'gt_iou_descriptive':sm['iou'], 'gt_iou_change_descriptive':sm['iou']-sample_metrics(overlap_counts(ref,truth))['iou']}
            records[name]=row
    summaries={}
    for c in CONDITIONS:
        rows=[r['conditions'][c] for r in records.values() if c!='shuffled' or not r['identical_question']]
        summaries[c]={'num_samples':len(rows),**{key:sum(r[key] for r in rows)/len(rows) if rows else None
            for key in ('agreement_iou','changed_fraction','area_fraction_change','gt_iou_descriptive','gt_iou_change_descriptive')}}
    return {'records':records,'summary':summaries,'excluded_identical_shuffled':sum(r['identical_question'] for r in records.values())}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);inference_arguments(parser)
    parser.add_argument('--donors',required=True,help='Shared mapping file, created once and reused across models')
    parser.add_argument('--seed',type=int,default=2027)
    parser.add_argument('--no-cache-images',action='store_true')
    args=parser.parse_args(argv)
    model,dataset,loader,device,metadata=load_inference(args,'val')
    mapping=get_donors(dataset,args.donors,args.seed)
    out=new_output(args.output_dir)
    result=analyze(model,loader,dataset,mapping,device,not args.no_cache_images)
    write_json(out/'results.json',{**metadata,'donor_mapping':mapping,
        'interpretation':'Question sensitivity; GT IoU under altered questions is descriptive only',**result})
    print(json.dumps(result['summary']))


if __name__=='__main__':main()
