"""Predict one image/question mask using this branch's checkpoint configuration."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageOps
from models.model import build_model
from models.checkpoint import read_checkpoint, load_checkpoint


def prediction_parser(description=__doc__):
    parser=argparse.ArgumentParser(description=description)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--image',required=True)
    parser.add_argument('--question',required=True,help='Question text without a Q: prefix')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser


@torch.no_grad()
def predict_image(checkpoint,image_path,question,device):
    saved=read_checkpoint(checkpoint)
    config=SimpleNamespace(**saved['run_config'])
    model=build_model(config)
    load_checkpoint(checkpoint,model)
    model.to(device).eval()
    with Image.open(image_path) as source:
        image=ImageOps.exif_transpose(source).convert('RGB')
    question_text=f'Q: {question}'
    batch={'image':to_tensor(image.resize((config.image_size,config.image_size),Image.Resampling.BICUBIC)).unsqueeze(0).to(device),
           'text':[question_text],'question_text':[question_text],'filename':[Path(image_path).name]}
    if getattr(config,'needs_detail',False):
        batch['detail_image']=to_tensor(image.resize((config.detail_size,config.detail_size),Image.Resampling.BICUBIC)).unsqueeze(0).to(device)
    logits=model(batch)['logits'].float()
    mask=F.interpolate(logits,size=(image.height,image.width),mode='bilinear',align_corners=False)[0,0]>0
    return image,Image.fromarray(mask.cpu().numpy().astype('uint8')*255)


def main(argv=None):
    args=prediction_parser().parse_args(argv)
    _,mask=predict_image(args.checkpoint,args.image,args.question,args.device)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    mask.save(output)
    print(f'Saved binary mask to {output}')


if __name__=='__main__':
    main()
