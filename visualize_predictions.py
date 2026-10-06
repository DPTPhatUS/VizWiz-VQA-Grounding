"""Save an original-resolution overlay from this branch's predicted evidence mask."""
from pathlib import Path
from PIL import Image
from predict_save import prediction_parser, predict_image


def main(argv=None):
    args=prediction_parser(__doc__).parse_args(argv)
    image,mask=predict_image(args.checkpoint,args.image,args.question,args.device)
    tinted=Image.blend(image,Image.new('RGB',image.size,(255,60,30)),.5)
    overlay=Image.composite(tinted,image,mask)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    overlay.save(output)
    print(f'Saved evidence overlay to {output}')


if __name__=='__main__':
    main()
