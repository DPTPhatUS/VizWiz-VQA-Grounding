"""Predict one image/question mask using this branch's checkpoint configuration."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from dataset import load_image, image_inputs
from utils import load_inference_model, to_device


def prediction_parser(description=__doc__):
    parser=argparse.ArgumentParser(description=description)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--image',required=True)
    parser.add_argument('--question',required=True,help='Question text without a Q: prefix')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--stage",choices=("refiner","router"),help="For standalone raw weights; defaults to router")
    return parser


@torch.no_grad()
def predict_image(checkpoint,image_path,question,device,stage=None):
    model, config = load_inference_model(checkpoint, device, stage=stage)
    image = load_image(image_path)
    inputs = image_inputs(image, config.get("image_size", 336), config.get("detail_size", 672))
    batch = to_device({key: tensor.unsqueeze(0) for key, tensor in inputs.items()}, device)
    batch['question_text'] = [f'Q: {question}']
    logits = model(batch)['logits'].float()
    mask = F.interpolate(logits, size=(image.height, image.width),
                         mode='bilinear', align_corners=False)[0, 0] > 0
    return image, Image.fromarray(mask.cpu().numpy().astype('uint8') * 255)


def main(argv=None):
    args=prediction_parser().parse_args(argv)
    _,mask=predict_image(args.checkpoint,args.image,args.question,args.device,stage=args.stage)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    mask.save(output)
    print(f'Saved binary mask to {output}')


if __name__=='__main__':
    main()
