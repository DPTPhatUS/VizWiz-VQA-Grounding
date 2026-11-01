"""Model-forward and cached-input decoder timings, with synchronized CUDA measurements."""
import argparse
import contextlib
import json
import statistics
import time
import torch
from models.model import build_model
from utils import environment, parameter_counts, write_json


@torch.inference_mode()
def time_forward(call,device,precision,warmup=10,repeats=50):
    def autocast():return torch.autocast(device.type,dtype=torch.float16) if precision=='fp16' else contextlib.nullcontext()
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    with autocast():
        for _ in range(warmup):call()
        sync()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        times=[]
        for _ in range(repeats):
            sync();start=time.perf_counter();call();sync()
            times.append((time.perf_counter()-start)*1000)
    return {'mean_ms':statistics.mean(times),'median_ms':statistics.median(times),
            'batch1_images_per_second':1000/statistics.mean(times),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None}


@torch.inference_mode()
def decoder_inputs(model,batch):
    features=model.encode_image(batch['image']);text=model.text_encoder(batch['text'])
    if hasattr(model,'prepare_decoder'):
        return model.prepare_decoder(features,text)
    s1,s2,s3,vision=features
    tokens=model.text_proj(text.tokens)
    attended,_=model.cross_attn(vision.flatten(2).transpose(1,2),tokens,tokens,
                              key_padding_mask=~text.attention_mask,need_weights=False)
    return (attended.transpose(1,2).reshape_as(vision),s3,s2,s1,text)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=('ftc','cm-shared','cm-separate','ftc-capacity','unet'),
                        default=['ftc','cm-shared','cm-separate','ftc-capacity','unet'])
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--precision',choices=('fp32','fp16'),default='fp32')
    parser.add_argument('--warmup',type=int,default=10);parser.add_argument('--repeats',type=int,default=50)
    parser.add_argument('--tiny',action='store_true',help='Smoke timing only, excludes U-Net')
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    from pathlib import Path
    if Path(args.output).exists():raise ValueError('Refusing to overwrite benchmark output')
    if args.warmup<1 or args.repeats<1:raise ValueError('Warmup and repeats must be positive')
    device=torch.device(args.device)
    if args.precision=='fp16' and device.type!='cuda':raise ValueError('FP16 benchmark requires CUDA')
    if args.tiny and 'unet' in args.models:raise ValueError('Tiny smoke benchmark does not support U-Net')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    result={'environment':environment(),'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'precision':args.precision,'batch_size':1,'input_size':32 if args.tiny else 336,
            'warmup':args.warmup,'repeats':args.repeats,'tiny_smoke':args.tiny,'tf32':False,
            'timing_boundary':'GPU-resident input, model forward including text tokenization; excludes file IO and native resizing',
            'throughput_definition':'1000 / mean batch-1 latency in milliseconds','models':{}}
    for name in args.models:
        torch.manual_seed(42)
        if device.type=='cuda':torch.cuda.empty_cache()
        if name=='unet':
            from models.benchmark_unet import ReplicatedUNet
            model=ReplicatedUNet()
        else:
            model=build_model('cm' if name.startswith('cm-') else name,tiny=args.tiny,
                              connection_mode='shared' if name=='cm-shared' else 'separate')
        model.to(device).float().eval()
        size=32 if args.tiny else 336
        batch={'image':torch.rand(1,3,size,size,device=device),'text':['Q: What is the expiration date on this bottle?']}
        full=time_forward(lambda:model(batch),device,args.precision,args.warmup,args.repeats)
        cached=decoder_inputs(model,batch)
        if device.type=='cuda':torch.cuda.empty_cache()
        decoder=time_forward(lambda:model.decoder(*cached),device,args.precision,args.warmup,args.repeats)
        result['models'][name]={'parameters':parameter_counts(model),'experiment_config':model.experiment_config,
                                'forward':full,'decoder':decoder}
        del cached,model,batch
    write_json(args.output,result);print(json.dumps(result))


if __name__=='__main__':main()
