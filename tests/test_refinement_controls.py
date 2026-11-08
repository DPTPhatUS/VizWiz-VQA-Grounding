"""Scientific contracts for symmetric affinity and conditional two-role CM."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from checkpoints import model_from_checkpoint, load_model_weights
from diagnostics.roles import role_counts
from experiments.summarize import summarize, METRICS
from losses import (SupervisedObjective, segmentation_loss, membership_targets,
                    membership_loss, affinity_target_masses, affinity_loss)
from models.connections import neighborhood, symmetric_logits, propagate_affinity
from models.factory import build_model
from models.mask_decoder import CompactDecoder
from tests.test_commands import run_command
from tests.support import fixture_data
from train import training_parser, configure_run

MODES = ('symmetric-affinity', 'two-role')


def inputs():
    generator = torch.Generator().manual_seed(17)
    return {'image':torch.randn(2, 3, 32, 32, generator=generator),
            'text':['Q: What color?', 'Q: What number?'],
            'mask':(torch.rand(2, 1, 32, 32, generator=generator) > .6).float()}


def edge_output(mask, roles=3):
    batch, _, height, width = mask.shape
    indices, valid = neighborhood(height, width, 'cpu')
    return {'membership_logits':torch.zeros_like(mask),
            'edge_logits':torch.zeros(batch, height*width, 8, roles, requires_grad=True),
            'affinity_logits':torch.zeros(batch, height*width, 8, requires_grad=True),
            'edge_indices':indices, 'edge_valid':valid}


class RefinementControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_common_initial_weights_rng_and_real_decoder_counts(self):
        torch.manual_seed(42)
        original = build_model(tiny=True, connection_mode='separate')
        state = original.state_dict()
        rng = torch.get_rng_state()
        for mode, count in [('symmetric-affinity',1120550), ('two-role',1120568)]:
            torch.manual_seed(42)
            control = build_model(tiny=True, connection_mode='separate', local_refinement=mode)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            for name, value in control.state_dict().items():
                expected = state[name]
                if value.shape != expected.shape:
                    self.assertIn(name, ('decoder.connections.membership_head.weight',
                                         'decoder.connections.membership_head.bias',
                                         'decoder.connections.update_strength'))
                    expected = expected[:value.shape[0]]
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            decoder = CompactDecoder(connection_mode='separate', local_refinement=mode)
            self.assertEqual(sum(p.numel() for p in decoder.parameters()), count)
            self.assertTrue(all(p.requires_grad for p in decoder.parameters()))
        self.assertEqual(sum(p.numel() for p in CompactDecoder(connection_mode='separate').parameters()),1120585)

    def test_symmetric_scores_reverse_edges_and_gradients(self):
        indices, valid = neighborhood(2, 3, 'cpu')
        raw = torch.arange(48.).reshape(1, 6, 8).requires_grad_()
        score = symmetric_logits(raw, indices)
        opposite = torch.arange(7, -1, -1)
        torch.testing.assert_close(score[:, valid], score[:, indices, opposite][:, valid], rtol=0, atol=0)
        # 0 -> 1 is slot 3 at receiver 1; its reverse is slot 4 at receiver 0.
        self.assertEqual(score[0, 0, 4].item(), (raw[0,0,4].item()+raw[0,1,3].item())/2)
        score[:, valid].sum().backward()
        self.assertTrue(torch.all(raw.grad[:, valid] > 0))
        self.assertTrue(torch.all(raw.grad[:, ~valid] == 0))

    def test_affinity_walk_preserves_signed_constant_and_isolated_values(self):
        for height, width in [(1,1), (2,3)]:
            indices, valid = neighborhood(height, width, 'cpu')
            affinity = torch.randn(1, height*width, 8)
            for value in (-2., 0., 3.):
                unary = torch.full((1,1,height,width), value)
                actual = propagate_affinity(unary, affinity, indices, valid, torch.tensor([0.]), 2)
                torch.testing.assert_close(actual, unary)
        indices, valid = neighborhood(2,3,'cpu')
        unary = torch.arange(-3.,3.).reshape(1,1,2,3)
        actual = propagate_affinity(unary, torch.zeros(1,6,8), indices, torch.zeros_like(valid), torch.tensor([0.]), 2)
        torch.testing.assert_close(actual, unary)

    def test_two_step_walk_has_expected_self_normalization_and_anchor(self):
        indices, valid = neighborhood(1,2,'cpu')
        unary = torch.tensor([[[[2.,-2.]]]])
        # Neighbor affinity .5, self affinity 1, alpha .5.
        once = propagate_affinity(unary, torch.zeros(1,2,8), indices, valid, torch.tensor([0.]), 1)
        twice = propagate_affinity(unary, torch.zeros(1,2,8), indices, valid, torch.tensor([0.]), 2)
        torch.testing.assert_close(once, torch.tensor([[[[4/3,-4/3]]]]))
        torch.testing.assert_close(twice, torch.tensor([[[[11/9,-11/9]]]]))
        self.assertGreater(twice[0,0,0,1].item(), -2.)
        self.assertLess(twice[0,0,0,1].item(), 0.)

    def test_affinity_targets_use_both_agreement_types_and_fractional_masks(self):
        for mask, expected in [([1.,1.],[1.,0.,0.]), ([0.,0.],[0.,1.,0.]),
                               ([0.,1.],[0.,0.,1.]), ([.25,.75],[.1875,.1875,.625])]:
            truth = torch.tensor(mask).reshape(1,1,1,2)
            output = edge_output(truth)
            targets = affinity_target_masses(output, truth)
            valid = output['edge_valid']
            torch.testing.assert_close(targets[0,valid], torch.tensor([expected,expected]))
            torch.testing.assert_close(targets.sum(-1), valid.float()[None])
            self.assertAlmostEqual(affinity_loss(output, truth).item(), math.log(2), places=6)
            affinity_loss(output, truth).backward()
            self.assertTrue(torch.all(output['affinity_logits'].grad[:,~valid] == 0))
        # Border placeholders must contribute neither loss nor gradient.
        output['affinity_logits'].data[:,~valid] = 100
        self.assertAlmostEqual(affinity_loss(output, truth).item(), math.log(2), places=6)

    def test_affinity_supervision_rewards_correct_agreements_and_boundaries(self):
        for truth in (torch.zeros(1,1,1,2), torch.ones(1,1,1,2), torch.tensor([[[[0.,1.]]]])):
            output = edge_output(truth)
            agreeing = truth[0,0,0,0] == truth[0,0,0,1]
            output['affinity_logits'] = torch.full((1,2,8), 20. if agreeing else -20.)
            self.assertLess(affinity_loss(output, truth).item(), 1e-7)
            output['affinity_logits'] = -output['affinity_logits']
            self.assertGreater(affinity_loss(output, truth).item(), 19.)

    def test_two_role_targets_ignore_background_sources_only_in_auxiliary_loss(self):
        truth = torch.tensor([[[[1.,0.]]]])
        output = edge_output(truth, roles=2)
        targets = membership_targets(output, truth)
        torch.testing.assert_close(targets[0,0,4], torch.tensor([0.,0.]))
        torch.testing.assert_close(targets[0,1,3], torch.tensor([0.,1.]))
        membership_loss(output, truth).backward()
        self.assertTrue(torch.all(output['edge_logits'].grad[0,0] == 0))
        self.assertGreater(output['edge_logits'].grad[0,1,3].abs().sum().item(), 0.)
        for value in (0., 1.):
            truth = torch.full((1,1,2,2), value)
            output = edge_output(truth, roles=2)
            loss = membership_loss(output, truth)
            self.assertAlmostEqual(loss.item(), math.log(2) if value else 0., places=6)
            loss.backward()
            self.assertTrue(torch.isfinite(output['edge_logits'].grad).all())
        truth = torch.tensor([[[[.25,.75]]]])
        output = edge_output(truth, roles=2)
        targets = membership_targets(output, truth)
        torch.testing.assert_close(targets[0,0,4], torch.tensor([.1875,.5625]))
        torch.testing.assert_close(targets[0,1,3], torch.tensor([.1875,.0625]))

    def test_two_role_keeps_all_valid_edges_in_forward_and_can_cancel(self):
        model = build_model(tiny=True, connection_mode='separate', local_refinement='two-role').eval()
        with torch.no_grad():
            graph = model.decoder.connections
            graph.membership_head.weight.zero_()
            graph.membership_head.bias.zero_()
            batch = inputs()
            output = model(batch)
            # u + m - m can round differently from u in floating-point arithmetic.
            torch.testing.assert_close(output['membership_logits'], output['unary_logits'], rtol=1e-6, atol=1e-7)
            graph.membership_head.bias[:] = torch.tensor([20.,-20.])
            positive = model(batch)
            self.assertTrue(torch.all(positive['membership_logits'] > positive['unary_logits']))
            batch['mask'].zero_()  # Ground truth is not a forward input or an edge gate.
            again = model(batch)
            torch.testing.assert_close(again['logits'], positive['logits'], rtol=0, atol=0)

    def test_controls_have_finite_connected_gradients_and_correct_loss_routing(self):
        for mode in MODES:
            model = build_model(tiny=True, connection_mode='separate', local_refinement=mode)
            batch = inputs()
            loss, parts = SupervisedObjective(.1)(model, batch)
            self.assertIn('affinity' if mode == MODES[0] else 'membership', parts)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                else:
                    self.assertIsNone(parameter.grad, name)
            self.assertGreater(model.decoder.connections.membership_head.weight.grad.abs().sum().item(),0)
            self.assertGreater(model.decoder.connections.update_strength.grad.abs().sum().item(),0)
            output = model(batch)
            with self.assertRaisesRegex(ValueError,'three-role'):
                role_counts(output, batch['mask'])

    def test_zero_auxiliary_weight_matches_segmentation_gradients(self):
        for mode in MODES:
            model = build_model(tiny=True, local_refinement=mode)
            batch = inputs()
            total, _ = SupervisedObjective(0)(model,batch)
            segmentation = segmentation_loss(model(batch)['logits'],batch['mask'])
            torch.testing.assert_close(total,segmentation,rtol=0,atol=0)
            parameters = tuple(p for p in model.parameters() if p.requires_grad)
            for actual, expected in zip(torch.autograd.grad(total,parameters), torch.autograd.grad(segmentation,parameters)):
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_checkpoint_roundtrips_and_rejects_wrong_architecture(self):
        batch = inputs()
        for mode in MODES:
            model = build_model(tiny=True, connection_mode='separate', local_refinement=mode).eval()
            saved = {'model_state_dict':model.state_dict(), 'experiment_config':model.experiment_config, 'run_config':{}}
            restored = model_from_checkpoint(saved).eval()
            self.assertEqual(restored.experiment_config['local_refinement'],mode)
            with torch.no_grad():
                torch.testing.assert_close(model(batch)['logits'],restored(batch)['logits'],rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'configuration'):
                load_model_weights(build_model(tiny=True,connection_mode='separate'),saved)
            incomplete = copy.deepcopy(saved)
            del incomplete['experiment_config']['local_refinement']
            with self.assertRaises((RuntimeError, ValueError)):
                model_from_checkpoint(incomplete)

    def test_cli_controls_and_rejection(self):
        for mode in MODES:
            args = training_parser().parse_args(['--local-refinement',mode])
            configure_run(args)
            self.assertEqual(args.edge_loss_weight,.1)
            self.assertEqual(args.local_refinement,mode)
        for argv in [['--model','ftc','--local-refinement','two-role'],
                     ['--disable-membership','--local-refinement','symmetric-affinity'],
                     ['--model','ftc-capacity','--local-refinement','membership']]:
            args = training_parser().parse_args(argv)
            args._explicit = {a[2:].replace('-','_') for a in argv if a.startswith('--')}
            with self.assertRaises(ValueError):
                configure_run(args)


class GradualCampaigns(unittest.TestCase):
    def test_manifest_priority_and_existing_runs_preserved(self):
        manifest = json.loads(Path('configs/high_priority.json').read_text())
        expected = [f'{name}_s{seed}' for name in ('cm_affinity','cm_two_role') for seed in (42,123,456)]
        self.assertEqual([r['id'] for r in manifest['runs'][:6]], expected)
        self.assertEqual(len(manifest['runs']),27)
        self.assertEqual(len({r['id'] for r in manifest['runs']}),27)
        for name in ('ftc','cm_shared','cm_separate','context_only','membership_only','cm_noaux','ftc_capacity'):
            self.assertEqual({r['seed'] for r in manifest['runs'] if r['id'].rsplit('_s',1)[0]==name},{42,123,456})

    def test_paired_summary_across_separate_campaigns_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [{'id':f'{name}_s{seed}','seed':seed} for name in ('cm_separate','cm_affinity','cm_two_role') for seed in (42,123,456)]
            manifest = {'name':'controls','common':{},'runs':runs}
            for i,run in enumerate(runs):
                folder = root/f'campaign-{i}'/run['id']
                (folder/'test').mkdir(parents=True)
                full = run['id'].startswith('cm_separate')
                metrics = {key:(.6 if full else .5) for key in METRICS}
                metrics.update(dataset='test',metric_resolution='original',checkpoint={'run':run['id'],'epoch':1})
                (folder/'test/metrics.json').write_text(json.dumps({'summary':metrics}))
                record = {'study':'controls','run_config':{'seed':run['seed'],'world_size':1,'run_id':run['id']},
                          'selection':{'epoch':1},'provenance':{'git_commit':'fixture'}}
                (folder/'run.json').write_text(json.dumps(record))
            result = summarize(root,manifest)
            self.assertFalse(result['missing_runs'])
            for label in ('cm_affinity','cm_two_role'):
                paired = result['paired_cm_minus_controls'][label]
                self.assertEqual(paired['num_pairs'],3)
                self.assertAlmostEqual(paired['summary']['mean_iou']['mean'],.1)
            duplicate = root/'duplicate'/runs[0]['id']/'test'
            duplicate.mkdir(parents=True)
            (duplicate/'metrics.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Multiple results'):
                summarize(root,manifest)

    def test_new_controls_train_reload_and_export_native_metrics_in_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = fixture_data(root/'data')
            manifest = {'name':'controls-smoke','common':{'tiny':True,'num_epochs':1,
                'batch_size':2,'lr':1e-5,'num_workers':0,'val_every':1},
                'runs':[{'id':name+'_s42','model':'cm','connection_mode':'separate',
                         'local_refinement':mode,'seed':42}
                        for name,mode in zip(('cm_affinity','cm_two_role'),MODES)]}
            path = root/'manifest.json'
            path.write_text(json.dumps(manifest))
            run_command('-m','experiments.launch','--manifest',str(path),'--data-root',str(data),
                        '--output-root',str(root/'runs'),'--device','cpu','--no-hub')
            campaign = next((root/'runs').iterdir())
            for run in manifest['runs']:
                folder = campaign/run['id']
                saved = torch.load(folder/'best.pt',weights_only=True)
                self.assertEqual(saved['experiment_config']['local_refinement'],run['local_refinement'])
                self.assertEqual(saved['run_config']['local_refinement'],run['local_refinement'])
                history = json.loads((folder/'history.jsonl').read_text())
                self.assertIn('affinity' if run['local_refinement']==MODES[0] else 'membership', history)
                for split in ('val','test'):
                    result = json.loads((folder/split/'metrics.json').read_text())
                    self.assertEqual(result['summary']['num_samples'],3)
                    self.assertEqual(result['summary']['metric_resolution'],'original')
                    self.assertEqual(result['summary']['checkpoint']['epoch'],1)
                    self.assertTrue((folder/split/'logits.npz').is_file())
            self.assertEqual(len(summarize(root/'runs',manifest)['runs']),2)


if __name__ == '__main__':
    unittest.main()
