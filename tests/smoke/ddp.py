"""Run with torchrun --standalone --nproc_per_node=2 --module tests.smoke.ddp DATA_ROOT."""
import argparse
import json
import torch
import torch.distributed as dist
from dataset import VizWizGroundingDataset
from models.factory import build_model
from validation import validation_iou

parser=argparse.ArgumentParser();parser.add_argument('root')
parser.add_argument('--local-refinement',choices=('membership','symmetric-affinity','two-role'))
args=parser.parse_args()
torch.set_num_threads(2);dist.init_process_group('gloo')
try:
    rank,world=dist.get_rank(),dist.get_world_size()
    torch.manual_seed(42)
    model=build_model('cm' if args.local_refinement else 'ftc',tiny=True,
                      connection_mode='separate',local_refinement=args.local_refinement or 'membership')
    dataset=VizWizGroundingDataset(args.root,'val',32)
    loader_args=argparse.Namespace(batch_size=2,num_workers=0,device='cpu')
    score=validation_iou(model,dataset,loader_args,torch.device('cpu'),rank,world)
    shards=[list(range(r,len(dataset),world)) for r in range(world)]
    assert sorted(i for shard in shards for i in shard)==list(range(len(dataset)))
    if rank==0:
        single=validation_iou(model,dataset,loader_args,torch.device('cpu'))
        assert abs(score-single)<1e-12
        print(json.dumps({'ddp_validation_miou':score,'single_process_miou':single,'shards':shards}))
    dist.barrier()
finally:dist.destroy_process_group()
