"""Scientific behavior and offline experiment pipeline contracts."""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from dataset import VizWizGroundingDataset, collate_samples
from losses import SupervisedObjective, segmentation_loss, membership_targets
from models.cm import GroundingModel
from models.factory import build_model
from models.ftc import FTCDecoder
from metrics import metric_samples, native_logits, overlap_counts, summarize_counts
from checkpoints import model_from_checkpoint, read_checkpoint, load_model_weights, materialize_position_buffers
from diagnostics.questions import donor_mapping, get_donors, analyze as question_analysis
from diagnostics.roles import role_counts, aggregate
from diagnostics.thresholds import THRESHOLDS, select_threshold
from experiments.launch import commands
from tests.support import fixture_data
import train as train
import preflight as preflight

FIXTURES=Path(__file__).parent/'fixtures'


def batch():
    torch.manual_seed(99)
    return {'image':torch.randn(2,3,32,32),'text':['Q: Color?','Q: What number?'],
            'mask':(torch.rand(2,1,32,32)>.5).float()}


def state_equal(test,a,b):
    test.assertEqual(set(a),set(b))
    for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)


def checkpoint(model):
    return {'format_version':1, 'model_state_dict':model.state_dict(),
            'experiment_config':model.experiment_config, 'run_config':{}}


class RevisionModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_cm_reference_initialization_and_outputs(self):
        fixtures=torch.load(FIXTURES/'cm_reference.pt',weights_only=True)
        for mode,f in fixtures.items():
            torch.manual_seed(42);model=GroundingModel(tiny=True,connection_mode=mode).eval()
            state_equal(self,model.state_dict(),f['state'])
            with torch.no_grad():actual=model(f['batch'])
            for key in actual:torch.testing.assert_close(actual[key],f['output'][key],rtol=0,atol=0)
            state_equal(self,model_from_checkpoint(checkpoint(model)).state_dict(),f['state'])

    def test_exact_original_ftc(self):
        f=torch.load(FIXTURES/'ftc_reference.pt',weights_only=True)
        torch.manual_seed(42);model=build_model('ftc',tiny=True).eval()
        state_equal(self,model.state_dict(),f['state'])
        with torch.no_grad():torch.testing.assert_close(model(f['batch'])['logits'],f['output']['logits'],rtol=0,atol=0)
        state_equal(self,model_from_checkpoint(checkpoint(model)).state_dict(),f['state'])

    def test_current_checkpoint_format_and_strict_reconstruction(self):
        for kind in ('cm','ftc','ftc-capacity'):
            model=build_model(kind,tiny=True)
            saved=checkpoint(model)
            state_equal(self,model_from_checkpoint(saved).state_dict(),model.state_dict())
            saved.pop('format_version')
            with self.assertRaisesRegex(ValueError,'format_version'):model_from_checkpoint(saved)
            saved['format_version']=1
            del saved['experiment_config']['tiny']
            with self.assertRaisesRegex(ValueError,'Incomplete'):model_from_checkpoint(saved)

    def test_ablation_initialization_gradients_and_checkpoint_identity(self):
        for mode in ['shared','separate']:
            torch.manual_seed(42);base=GroundingModel(tiny=True,connection_mode=mode)
            for context,membership in [(True,False),(False,True),(False,False)]:
                torch.manual_seed(42);model=GroundingModel(tiny=True,connection_mode=mode,context=context,membership=membership)
                state_equal(self,model.state_dict(),base.state_dict())
                graph=model.decoder.connections
                if not context:graph.context_head.register_forward_hook(lambda *args:self.fail('Disabled context executed'))
                if not membership:graph.membership_head.register_forward_hook(lambda *args:self.fail('Disabled membership executed'))
                loss,_=SupervisedObjective(.1 if membership else 0)(model,batch());loss.backward()
                for name,p in model.named_parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
                    else:self.assertIsNone(p.grad,name)
                saved=checkpoint(model)
                state_equal(self,model_from_checkpoint(saved).state_dict(),model.state_dict())
                with self.assertRaisesRegex(ValueError,'configuration'):load_model_weights(base,saved)

    def test_noaux_loss_and_gradients(self):
        model=GroundingModel(tiny=True,connection_mode='separate');inputs=batch()
        total,_=SupervisedObjective(0)(model,inputs);seg=segmentation_loss(model(inputs)['logits'],inputs['mask'])
        torch.testing.assert_close(total,seg,rtol=0,atol=0)
        params=tuple(p for p in model.parameters() if p.requires_grad)
        for left,right in zip(torch.autograd.grad(total,params),torch.autograd.grad(seg,params)):
            torch.testing.assert_close(left,right,rtol=0,atol=0)

    def test_capacity_count_and_active_blocks(self):
        self.assertEqual(sum(p.numel() for p in FTCDecoder().parameters()),690753)
        self.assertEqual(sum(p.numel() for p in FTCDecoder(capacity=True).parameters()),1120845)
        torch.manual_seed(42);base=build_model('ftc',tiny=True)
        torch.manual_seed(42);capacity=build_model('ftc-capacity',tiny=True)
        for key,value in base.state_dict().items():torch.testing.assert_close(value,capacity.state_dict()[key],rtol=0,atol=0)
        loss,_=SupervisedObjective(0)(capacity,batch());loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in capacity.decoder.capacity.parameters()))

    def test_nonpersistent_clip_positions_materialize_after_meta_load(self):
        class Embeddings(nn.Module):
            def __init__(self):
                super().__init__()
                self.position_embedding=nn.Embedding(77,8)
                self.register_buffer('position_ids',torch.empty(1,77,device='meta',dtype=torch.int64),persistent=False)
        model=Embeddings();materialize_position_buffers(model)
        torch.testing.assert_close(model.position_ids,torch.arange(77).view(1,77))
        self.assertNotIn('position_ids',model.state_dict())

    def test_cli_rejects_conflicts(self):
        for argv in [['--model','ftc','--disable-context'],['--model','ftc','--edge-loss-weight','0'],
                     ['--disable-membership','--edge-loss-weight','.1']]:
            args=train.training_parser().parse_args(argv);args._explicit={a[2:].replace('-','_') for a in argv if a.startswith('--')}
            with self.assertRaises(ValueError):train.configure_run(args)


class RevisionMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_native_grid_and_probability_thresholds(self):
        logits=torch.tensor([[[[-2.,0.],[1.,3.]]]])
        b={'filename':['a.jpg'],'original_mask':[torch.ones(1,5,7)]}
        old=next(metric_samples(logits,b));_,resized,truth=next(native_logits(logits,b))
        torch.testing.assert_close(old[2],resized>math.log(.5/(1-.5)))
        for p in THRESHOLDS:torch.testing.assert_close(resized.sigmoid()>p,resized>math.log(p/(1-p)))
        self.assertAlmostEqual(summarize_counts({'a':overlap_counts(resized>0,truth)})['mean_iou'],old[1],places=6)
        empty=summarize_counts({'e':{'intersection':0,'predicted':0,'ground_truth':0,'pixels':3}})
        self.assertEqual(empty['mean_iou'],1);self.assertIsNone(empty['mean_area_ratio'])

    def test_validation_selection_ties_and_test_rejection(self):
        report={'split':'val','thresholds':[{'threshold':p,'metrics':{'mean_iou':.6}} for p in [.4,.6]]}
        self.assertEqual(select_threshold(report),.4)
        report['thresholds'].append({'threshold':.5,'metrics':{'mean_iou':.6}})
        self.assertEqual(select_threshold(report),.5)
        report['split']='test'
        with self.assertRaises(ValueError):select_threshold(report)
        report['split']='val';report['limit']=2
        with self.assertRaises(ValueError):select_threshold(report)

    def test_role_targets_fractional_empty_full_and_borders(self):
        model=GroundingModel(tiny=True);inputs=batch();output=model(inputs)
        for target in [torch.zeros_like(inputs['mask']),torch.ones_like(inputs['mask']),inputs['mask']]:
            targets=membership_targets(output,target)
            torch.testing.assert_close(targets.sum(-1),output['edge_valid'].float().expand(2,-1,-1))
            counts=role_counts(output,target);self.assertTrue(math.isfinite(counts['balanced_soft_loss']))
            self.assertEqual(counts['edges'],2*int(output['edge_valid'].sum()))
            summary=aggregate([{'roles':counts,'mask':{'relative_area_error':.1,'iou':.4}}])
            self.assertAlmostEqual(sum(summary['predicted_frequencies']),1);self.assertTrue(0<=summary['balanced_accuracy']<=1)

    def test_question_mapping_and_actual_text_intervention(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=fixture_data(Path(tmp)/'data');dataset=VizWizGroundingDataset(data,'val',32)
            mapping=get_donors(dataset,Path(tmp)/'donors.json',2027)
            self.assertEqual(mapping,donor_mapping(dataset,2027));self.assertEqual(set(mapping['donors']),set(mapping['donors'].values()))
            self.assertTrue(all(a!=b for a,b in mapping['donors'].items()))
            self.assertEqual(mapping,get_donors(dataset,Path(tmp)/'donors.json',2027))
            class Spy(nn.Module):
                def __init__(self):super().__init__();self.seen=[]
                def forward(self,b):
                    self.seen.append(list(b['text']))
                    return {'logits':torch.tensor([len(t)%2 for t in b['text']],dtype=torch.float32).view(-1,1,1,1).expand(-1,1,2,2)}
            model=Spy();loader=[collate_samples([dataset[i] for i in range(len(dataset))])]
            result=question_analysis(model,loader,dataset,mapping,torch.device('cpu'),cache=False)
            self.assertEqual(model.seen[3],['Q: What is this?']*3);self.assertEqual(model.seen[4],['Q:']*3)
            self.assertEqual(result['summary']['repeat']['changed_fraction'],0)
            self.assertEqual(result['summary']['shuffled']['num_samples'],3-result['excluded_identical_shuffled'])
            for kind in ['ftc','cm']:
                cached=question_analysis(build_model(kind,tiny=True).eval(),loader,dataset,mapping,torch.device('cpu'),cache=True)
                self.assertEqual(cached['summary']['repeat']['changed_fraction'],0)


class RevisionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_validation_logging_preserves_training(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);data=fixture_data(root/'data')
            common=['--tiny','--data-root',str(data),'--device','cpu','--num-workers','0',
                    '--batch-size','2','--num-epochs','2','--save-every','1']
            train.train_main(common+['--output-dir',str(root/'plain')])
            train.train_main(common+['--output-dir',str(root/'val'),'--val-every','1'])
            a=read_checkpoint(root/'plain/checkpoint_epoch2.pt');b=read_checkpoint(root/'val/checkpoint_epoch2.pt')
            state_equal(self,a['model_state_dict'],b['model_state_dict']);self.assertTrue((root/'val/best.pt').exists())

    def test_rolling_resume_preserves_best_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);data=fixture_data(root/'data')
            common=['--tiny','--model','ftc','--data-root',str(data),'--device','cpu','--num-workers','0',
                    '--batch-size','2','--val-every','1','--checkpoint-policy','rolling']
            with patch.object(train,'validation_iou',return_value=.9):train.train_main(common+['--num-epochs','1','--output-dir',str(root/'resume')])
            with patch.object(train,'validation_iou',return_value=.8):
                train.train_main(common+['--num-epochs','2','--output-dir',str(root/'resume'),'--resume-checkpoint',str(root/'resume/latest.pt')])
            with patch.object(train,'validation_iou',side_effect=[.9,.8]):train.train_main(common+['--num-epochs','2','--output-dir',str(root/'continuous')])
            a=read_checkpoint(root/'resume/latest.pt');b=read_checkpoint(root/'continuous/latest.pt')
            state_equal(self,a['model_state_dict'],b['model_state_dict']);self.assertEqual(a['best_epoch'],1)
            self.assertEqual(read_checkpoint(root/'resume/best.pt')['epoch'],1)
            with self.assertRaisesRegex(ValueError,'seed'):
                train.train_main(common+['--num-epochs','3','--seed','123','--output-dir',str(root/'resume'),
                                        '--resume-checkpoint',str(root/'resume/latest.pt')])

    def test_preflight_all_models_and_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            data=fixture_data(Path(tmp)/'data');self.assertEqual(preflight.validate_data(data),{'train':4,'val':3})
            for options in [[],['--disable-context'],['--disable-membership'],['--model','ftc'],['--model','ftc-capacity']]:
                result=preflight.main(['--tiny','--device','cpu','--data-root',str(data),*options]);self.assertTrue(math.isfinite(result['losses'][0]))
            manifest=json.loads(Path('configs/high_priority.json').read_text());self.assertEqual(len(manifest['runs']),13)
            from experiments.launch import main
            absent=Path(tmp)/'never-created';main(['--data-root',str(data),'--output-root',str(absent),'--runs','ftc_s42','--dry-run'])
            self.assertFalse(absent.exists());cmds=commands(manifest,manifest['runs'][0],data,absent,'cpu',2)
            self.assertIn('torch.distributed.run',cmds[1]);self.assertIn('--val-every',cmds[1])


if __name__=='__main__':unittest.main()
