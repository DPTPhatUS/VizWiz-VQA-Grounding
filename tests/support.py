"""Small native-resolution splits for offline integration tests."""
import json
from pathlib import Path
import numpy as np
from PIL import Image


def fixture_data(root):
    root=Path(root)
    for split,count in [('train',4),('val',3),('test',3)]:
        (root/split).mkdir(parents=True)
        (root/'binary_masks_png'/split).mkdir(parents=True)
        records={}
        for i in range(count):
            name=f'{i}.jpg';height,width=13+i,19+i
            Image.new('RGB',(width,height),(60+30*i,80,120)).save(root/split/name)
            mask=np.zeros((height,width),dtype=np.uint8);mask[2:8,3:12]=255
            Image.fromarray(mask).save(root/'binary_masks_png'/split/f'{i}.png')
            records[name]={'question':['What color?','What number?','What color?','What object?'][i]}
        (root/f'{split}_grounding.json').write_text(json.dumps(records))
    return root
