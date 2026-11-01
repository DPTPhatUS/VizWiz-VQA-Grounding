"""Shared inference arguments, outputs, and provenance."""
import torch
from dataset import VizWizGroundingDataset, make_loader
from checkpoints import read_checkpoint, model_from_checkpoint, checkpoint_info
from runtime import environment, parameter_counts


def inference_arguments(parser, checkpoint_required=True):
    parser.add_argument('--checkpoint', required=checkpoint_required)
    parser.add_argument('--data-root', default='data/vizwiz')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--limit', type=int, default=0, help='Smoke tests only; never use for publication results')
    parser.add_argument('--output-dir', required=True)



def load_inference(args, split):
    if args.batch_size < 1 or args.num_workers < 0 or args.limit < 0:
        raise ValueError('Invalid loader or limit settings')
    saved = read_checkpoint(args.checkpoint)
    model = model_from_checkpoint(saved)
    size = saved['run_config']['image_size']
    info = checkpoint_info(args.checkpoint,saved)
    del saved
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = VizWizGroundingDataset(args.data_root, split, size)
    sampler = list(range(min(args.limit,len(dataset)))) if args.limit else None
    loader = make_loader(dataset,args,args.batch_size,sampler=sampler)
    metadata = {'checkpoint':info,
                'experiment_config':model.experiment_config, 'split':split, 'limit':args.limit,
                'parameters':parameter_counts(model), 'environment':environment()}
    return model, dataset, loader, device, metadata


def size_bin(area):
    for upper,name in [(0.01,'tiny'),(.05,'small'),(.25,'medium'),(.5,'large'),(float('inf'),'dominant')]:
        if area < upper:
            return name


def plot_lines(path, xs, series, xlabel):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5,3.5))
    for label,values in series.items():
        ax.plot(xs,values,marker='.',label=label)
    ax.set_xlabel(xlabel); ax.set_ylabel('Image-wise mean'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
