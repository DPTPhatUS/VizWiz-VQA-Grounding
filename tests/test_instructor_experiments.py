"""Contracts for the instructor's essential/high-priority experiments."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from checkpoints import model_from_checkpoint, load_model_weights
from diagnostics.benchmark import count_flops
from experiments.summarize import summarize
from losses import membership_targets, membership_loss, SupervisedObjective
from metrics import boundary_band, boundary_counts, overlap_counts, summarize_counts
from models.connections import symmetric_logits
from models.factory import build_model
from models.mask_decoder import CompactDecoder
from tests.support import fixture_data
from tests.test_commands import run_command
from tests.test_refinement_controls import edge_output, inputs
from train import training_parser, configure_run


class InstructorModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_symmetric_three_role_initialization_capacity_and_reciprocal_probabilities(self):
        torch.manual_seed(42)
        original = build_model(tiny=True, connection_mode='separate').eval()
        rng = torch.get_rng_state()
        torch.manual_seed(42)
        model = build_model(tiny=True, connection_mode='separate', local_refinement='symmetric-membership').eval()
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        decoder = CompactDecoder(connection_mode='separate', local_refinement='symmetric-membership')
        self.assertEqual(sum(p.numel() for p in decoder.parameters()), 1120585)
        batch = inputs()
        left, output = original(batch), model(batch)
        torch.testing.assert_close(output['unary_logits'], left['unary_logits'], rtol=0, atol=0)
        expected = symmetric_logits(left['edge_logits'].float(), left['edge_indices'])
        torch.testing.assert_close(output['edge_logits'], expected, rtol=0, atol=0)
        probabilities = output['edge_logits'].softmax(-1)
        reverse = probabilities[:, output['edge_indices'], torch.arange(7, -1, -1)]
        valid = output['edge_valid']
        torch.testing.assert_close(probabilities[:,valid], reverse[:,valid], rtol=0, atol=0)

    def test_symmetric_targets_and_loss_preserve_valid_class_mass(self):
        for values, expected in [([0.,1.],[0.,.5,.5]), ([1.,1.],[1.,0.,0.]),
                                 ([0.,0.],[0.,0.,1.]), ([.25,.75],[.1875,.3125,.5])]:
            mask = torch.tensor(values).reshape(1,1,1,2)
            output = edge_output(mask)
            directed = membership_targets(output, mask)
            output['symmetric_targets'] = True
            target = membership_targets(output, mask)
            valid = output['edge_valid']
            torch.testing.assert_close(target[0,valid], torch.tensor([expected,expected]))
            torch.testing.assert_close(target.sum(-1), valid[None].float())
            torch.testing.assert_close(target.sum((1,2)), directed.sum((1,2)))
            loss = membership_loss(output, mask)
            self.assertAlmostEqual(loss.item(), math.log(3), places=6)
            loss.backward()
            self.assertTrue(torch.all(output['edge_logits'].grad[:,~valid] == 0))
        # Reciprocal averaging is also the constrained symmetric objective for
        # the existing directional targets, with the same valid class masses.
        mask = torch.rand(2,1,3,4)
        output = edge_output(mask)
        raw = torch.randn_like(output['edge_logits'], requires_grad=True)
        output['edge_logits'] = symmetric_logits(raw, output['edge_indices'])
        directed_loss = membership_loss(output, mask)
        output['symmetric_targets'] = True
        sym_loss = membership_loss(output, mask)
        torch.testing.assert_close(sym_loss, directed_loss)
        torch.testing.assert_close(torch.autograd.grad(sym_loss, raw, retain_graph=True)[0],
                                   torch.autograd.grad(directed_loss, raw)[0])

    def test_depths_preserve_weights_use_same_signed_update_and_roundtrip(self):
        batch = inputs()
        torch.manual_seed(42)
        original = build_model(tiny=True, connection_mode='separate')
        for mode in ('membership','symmetric-membership'):
            for steps in (0,1,2,3):
                torch.manual_seed(42)
                model = build_model(tiny=True, connection_mode='separate',
                                    local_refinement=mode, membership_steps=steps)
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
                output = model(batch)
                anchor = output['unary_logits'].float().flatten(1)
                expected = anchor
                weights = output['edge_logits'].float().softmax(-1) * output['edge_valid'][None,:,:,None]
                degree = output['edge_valid'].sum(-1).clamp_min(1)
                for _ in range(steps):
                    senders = expected.sigmoid()[:,output['edge_indices']]
                    expected = anchor + (weights[...,0]*senders).sum(-1)/degree - (weights[...,1]*senders).sum(-1)/degree
                torch.testing.assert_close(output['membership_logits'].flatten(1), expected, rtol=0, atol=0)
                loss,_ = SupervisedObjective(.1)(model,batch)
                loss.backward()
                for name,p in model.named_parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad,name)
                        self.assertTrue(torch.isfinite(p.grad).all(),name)
                    else:
                        self.assertIsNone(p.grad,name)
                self.assertEqual(model.decoder.connections.update_strength.requires_grad, steps > 0)
                saved={'model_state_dict':model.state_dict(),'experiment_config':model.experiment_config,'run_config':{}}
                restored=model_from_checkpoint(saved).eval()
                model.eval()
                torch.testing.assert_close(restored(batch)['logits'],model(batch)['logits'],rtol=0,atol=0)
                self.assertEqual(restored.decoder.connections.steps,steps)
                if steps != 2 or mode != 'membership':
                    with self.assertRaisesRegex(ValueError,'configuration'):
                        load_model_weights(original,saved)

    def test_cli_rejects_negative_and_ignored_depths(self):
        for flags in [[],['--membership-steps','0'],['--membership-steps','3','--local-refinement','symmetric-membership']]:
            args=training_parser().parse_args(flags);configure_run(args)
            self.assertIn(args.membership_steps,(0,2,3))
        for flags in [['--membership-steps','-1'],['--disable-membership','--membership-steps','0'],
                      ['--model','ftc','--membership-steps','2']]:
            args=training_parser().parse_args(flags)
            args._explicit={v[2:].replace('-','_') for v in flags if v.startswith('--')}
            with self.assertRaises(ValueError):configure_run(args)


class InstructorMeasurements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_boundary_erosion_matches_iterative_reference_and_image_edges(self):
        rng=np.random.default_rng(2027)
        for shape in [(1,1),(3,5),(40,60),(101,153)]:
            for mask in [np.zeros(shape,dtype=bool),np.ones(shape,dtype=bool),rng.random(shape)>.3]:
                eroded=mask.copy()
                for _ in range(max(1,round(.02*math.hypot(*shape)))):
                    padded=np.pad(eroded,1)
                    eroded=np.logical_and.reduce([padded[y:y+shape[0],x:x+shape[1]] for y in range(3) for x in range(3)])
                np.testing.assert_array_equal(boundary_band(mask),mask & ~eroded)
        full=torch.ones(5,5,dtype=torch.bool)
        self.assertEqual(int(boundary_band(full).sum()),16)
        counts=overlap_counts(full,full)|boundary_counts(full,full)
        self.assertEqual(summarize_counts({'full':counts})['boundary_miou'],1)
        empty=torch.zeros_like(full)
        counts=overlap_counts(empty,full)|boundary_counts(empty,full)
        self.assertEqual(summarize_counts({'empty':counts})['boundary_miou'],0)

    def test_flop_count_convention_and_attention_setting_restored(self):
        layer=torch.nn.Linear(3,4,bias=False)
        x=torch.zeros(2,3)
        before=torch.backends.mha.get_fastpath_enabled()
        result=count_flops(lambda:layer(x),torch.device('cpu'),'fp32')
        self.assertEqual(result['counted_flops'],2*2*3*4)
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(),before)
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            count_flops(lambda:(_ for _ in ()).throw(RuntimeError('deliberate')),torch.device('cpu'),'fp32')
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(),before)

    def test_benchmark_all_depths_and_symmetric_three_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'benchmark.json'
            run_command('-m','diagnostics.benchmark','--tiny','--device','cpu','--models',
                        'ftc','ftc-capacity','cm-separate','cm-symmetric-membership',
                        '--membership-steps','0','1','2','3','--warmup','1','--repeats','2','--output',str(path))
            report=json.loads(path.read_text())
            self.assertEqual(len(report['models']),10)
            self.assertIn('flop_exclusions',report)
            for key,result in report['models'].items():
                for mode in ('forward','decoder'):
                    self.assertGreater(result[mode]['mean_ms'],0)
                    self.assertGreater(result[mode]['counted_flops'],0)
                    self.assertIsNone(result[mode]['peak_allocated_bytes'])
                if key.startswith('cm-'):
                    self.assertEqual(result['experiment_config']['membership_steps'],int(key[-1]))

    def test_priority_manifest_and_training_through_launcher(self):
        source=json.loads(Path('configs/instructor_high_priority.json').read_text())
        self.assertEqual(len(source['runs']),15)
        self.assertEqual(len({r['id'] for r in source['runs']}),15)
        self.assertEqual(source['common']['num_epochs'],30)
        for prefix in ('cm_symmetric_membership','cm_steps0','cm_steps1','cm_steps3','cm_separate'):
            self.assertEqual({r['seed'] for r in source['runs'] if r['id'].startswith(prefix+'_s')},{42,123,456})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=fixture_data(root/'data')
            manifest={'name':'instructor-smoke','common':{'tiny':True,'num_epochs':1,'batch_size':2,
                'num_workers':0,'lr':1e-5,'val_every':1},
                'runs':[r for r in source['runs'] if r['seed']==42]}
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            run_command('-m','experiments.launch','--manifest',str(path),'--data-root',str(data),
                        '--output-root',str(root/'outputs'),'--device','cpu','--no-hub')
            result=summarize(root/'outputs',manifest)
            self.assertEqual(len(result['runs']),5)
            self.assertFalse(result['missing_boundary_runs'])
            for row in result['runs']:
                self.assertIn('val_mean_iou',row)
                self.assertIn('val_boundary_miou',row)
                self.assertTrue(0 <= row['boundary_miou'] <= 1)
            for label in ('cm_symmetric_membership','cm_steps0','cm_steps1','cm_steps3'):
                self.assertIn('boundary_miou',result['paired_cm_minus_controls'][label]['summary'])


if __name__=='__main__':
    unittest.main()
