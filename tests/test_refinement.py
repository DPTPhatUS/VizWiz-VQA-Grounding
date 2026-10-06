import importlib.util
import unittest
import torch
from research.model import ResearchGrounder


def implementation():
    if importlib.util.find_spec('research.refinement') is None:
        raise AssertionError('gain-guided refinement implementation is missing')
    from research import refinement
    return refinement


def batch():
    return {'image': torch.rand(2,3,28,28), 'detail_image': torch.rand(2,3,56,56),
            'mask': torch.ones(2,1,28,28), 'detail_mask': torch.ones(2,1,56,56),
            'text':['Q: what?', 'Q: where?'], 'question_text':['Q: what?', 'Q: where?']}


class RefinementTests(unittest.TestCase):
    def test_two_scale_windows_cover_nondivisible_canvas_and_scatter_aligns(self):
        r = implementation()
        boxes = r.candidate_windows(11,17)
        self.assertEqual(len(boxes),13)
        covered = torch.zeros(11,17)
        for y0,x0,y1,x1 in boxes:
            self.assertTrue(0 <= y0 < y1 <= 11 and 0 <= x0 < x1 <= 17)
            covered[y0:y1,x0:x1] += 1
        self.assertTrue((covered > 0).all())
        coarse = torch.zeros(1,1,4,6)
        result = r.blend_residuals(coarse, [torch.full((1,1,2,2),2.), torch.full((1,1,2,2),4.)],
                                   [(0,0,2,4),(0,2,2,6)])
        torch.testing.assert_close(result[0,0,0],torch.tensor([2.,2.,3.,3.,4.,4.]))
        self.assertTrue((result[:,:,2:] == 0).all())

    def test_gain_is_per_image_iou_difference_and_can_be_negative(self):
        r = implementation()
        truth = torch.tensor([[[[1.,0.]]], [[[1.,0.]]]])
        coarse = torch.tensor([[[[-1.,-1.]]], [[[1.,-1.]]]])
        refined = torch.tensor([[[[1.,-1.]]], [[[-1.,-1.]]]],requires_grad=True)
        gains = r.actual_gains(coarse, [refined], truth)
        torch.testing.assert_close(gains,torch.tensor([[1.],[-1.]]))
        self.assertFalse(gains.requires_grad)

    def test_selection_suppresses_duplicate_windows_and_skips_negative_gain(self):
        r = implementation()
        boxes = [(0,0,4,4),(0,0,4,4),(4,4,8,8)]
        scores = torch.tensor([[.9,.8,-.1],[.1,.2,.3]])
        selected = r.select_windows(scores,boxes,budget=4,skip_nonpositive=True)
        self.assertEqual(selected,[[0],[2,1]])
        self.assertEqual(r.select_windows(scores,boxes,1,False),[[0],[2]])

    def test_stages_freeze_modules_and_loss_gradients_follow_stage(self):
        r = implementation()
        for stage in ('refiner','router'):
            model = r.GainGuidedGrounder(ResearchGrounder(tiny=True),stage=stage,crop_size=16)
            model.train()
            self.assertFalse(model.coarse.training)
            self.assertEqual(model.refiner.training,stage=='refiner')
            output = model(batch(),compute_loss=True)
            output['loss'].backward()
            self.assertTrue(torch.isfinite(output['loss']))
            self.assertTrue(all(p.grad is None for p in model.coarse.parameters()))
            active = model.refiner if stage=='refiner' else model.router
            inactive = model.router if stage=='refiner' else model.refiner
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in active.parameters()))
            self.assertTrue(all(p.grad is None for p in inactive.parameters()))

    def test_random_routing_is_filename_seeded_and_batch_invariant(self):
        r = implementation()
        model = r.GainGuidedGrounder(ResearchGrounder(tiny=True),crop_size=16,policy='random').eval()
        inputs=batch(); inputs['filename']=['one.jpg','two.jpg']
        with torch.no_grad():
            first=model(inputs)
            inputs['image']=1-inputs['image']
            second=model(inputs)
            one={k:v[:1] for k,v in inputs.items()}
            single=model(one)
        torch.testing.assert_close(first['gain_scores'],second['gain_scores'])
        torch.testing.assert_close(second['gain_scores'][:1],single['gain_scores'])
        torch.testing.assert_close(second['logits'][:1],single['logits'],atol=1e-5,rtol=1e-5)

    def test_inference_uses_detail_pixels_but_not_annotations_and_obeys_budget(self):
        r = implementation()
        model = r.GainGuidedGrounder(ResearchGrounder(tiny=True),crop_size=16,policy='fixed',budget=2).eval()
        inputs = batch()
        with torch.no_grad():
            a = model(inputs)
            inputs['detail_image'] = 1-inputs['detail_image']
            b = model(inputs)
            self.assertFalse(torch.allclose(a['logits'],b['logits']))
            inputs.pop('mask'); inputs.pop('detail_mask'); inputs['text']=['A: secret','A: changed']
            c = model(inputs)
        torch.testing.assert_close(b['logits'],c['logits'])
        self.assertEqual(c['crop_counts'].tolist(),[2,2])
        self.assertEqual(c['logits'].shape,(2,1,56,56))
        for policy in ('gain','uncertainty','random','relevance'):
            model.policy=policy
            self.assertTrue((model(inputs)['crop_counts'] <= 2).all())

if __name__ == '__main__': unittest.main()
