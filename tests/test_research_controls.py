import json
import tempfile
import unittest
from pathlib import Path
import torch
from PIL import Image

from research.data import GroundingDataset, collate_samples
from research.losses import segmentation_loss, per_image_iou
from research.checkpoint import save_checkpoint, load_checkpoint
from research.model import ResearchGrounder

class ControlTests(unittest.TestCase):
    def test_question_only_original_geometry_and_collation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'val').mkdir(); (p/'binary_masks_png/val').mkdir(parents=True)
            Image.new('RGB',(17,11),'red').save(p/'val/x.jpg')
            Image.new('L',(17,11),255).save(p/'binary_masks_png/val/x.png')
            (p/'val_grounding.json').write_text(json.dumps({'x.jpg':{'question':'color?','most_common_answer':'red'}}))
            ds=GroundingDataset(p,'val',image_size=28,detail_size=56)
            s=ds[0]; self.assertEqual(s['text'],'Q: color?')
            self.assertEqual(s['original_mask'].shape,(1,11,17))
            b=collate_samples([s,s]); self.assertEqual(b['detail_image'].shape,(2,3,56,56))
            self.assertIsInstance(b['original_mask'],list)
            self.assertEqual(b['answer_available'].tolist(),[True,True])

    def test_loss_handles_empty_masks_and_backprop(self):
        z=torch.randn(2,1,8,8,requires_grad=True); y=torch.zeros_like(z)
        loss=segmentation_loss(z,y,1); loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(per_image_iou(torch.full_like(z,-2),y).tolist(),[1,1])

    def test_model_forward_gradients_and_padding_invariance(self):
        m=ResearchGrounder(tiny=True).eval()
        image=torch.rand(1,3,28,28)
        single=m({'image':image,'text':['Q: x']})['logits']
        double=m({'image':image.repeat(2,1,1,1),'text':['Q: x','Q: much longer question']})['logits'][:1]
        torch.testing.assert_close(single,double,atol=2e-5,rtol=2e-5)
        single.mean().backward()
        self.assertTrue(any(p.grad is not None for p in m.parameters()))

    def test_checkpoint_identity_rejects_wrong_configuration(self):
        m=ResearchGrounder(tiny=True); opt=torch.optim.Adam(m.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'state.pt'
            save_checkpoint(p,m,opt,None,1,.5,{'mode':'question'})
            saved=load_checkpoint(p,m)
            self.assertEqual(saved['epoch'],1)
            m.experiment_config['tiny']=False
            with self.assertRaisesRegex(ValueError,'configuration'):
                load_checkpoint(p,m)

if __name__=='__main__': unittest.main()
