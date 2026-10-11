"""Predict one image/question mask using this branch's checkpoint configuration."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageOps
from models.model import GroundingModel, TeacherModel
from utils import read_checkpoint, load_model_weights, is_tiny_state


def prediction_parser(description=__doc__):
    parser=argparse.ArgumentParser(description=description)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--image',required=True)
    parser.add_argument('--question',required=True,help='Question text without a Q: prefix')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--teacher",action="store_true",help="Use standalone teacher weights")
    return parser


@torch.no_grad()
def predict_image(checkpoint,image_path,question,device,teacher=False):
    saved=read_checkpoint(checkpoint)
    config=saved['run_config']
    state=saved['model_state_dict']
    is_teacher=(
        teacher
        or config.get("teacher_training",False)
        or (bool(state) and not any(k.startswith("coarse.") for k in state))
    )
    model_type=TeacherModel if is_teacher else GroundingModel
    model=model_type(tiny=config.get("tiny",is_tiny_state(state)))
    load_model_weights(model,saved)
    model.to(device).eval()
    with Image.open(image_path) as source:
        image=ImageOps.exif_transpose(source).convert('RGB')
    question_text=f'Q: {question}'
    batch={'image':to_tensor(image.resize((config.get("image_size",336),config.get("image_size",336)),Image.Resampling.BICUBIC)).unsqueeze(0).to(device),  # (1, 3, H, W)
           'text':[question_text],'question_text':[question_text],'filename':[Path(image_path).name]}
    logits=model(batch)['logits'].float()  # (1, 1, H_dec, W_dec)
    mask=F.interpolate(logits,size=(image.height,image.width),mode='bilinear',align_corners=False)[0,0]>0  # (H_orig, W_orig)
    return image,Image.fromarray(mask.cpu().numpy().astype('uint8')*255)


def main(argv=None):
    args=prediction_parser().parse_args(argv)
    _,mask=predict_image(args.checkpoint,args.image,args.question,args.device,teacher=args.teacher)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    mask.save(output)
    print(f'Saved binary mask to {output}')


if __name__=='__main__':
    main()
