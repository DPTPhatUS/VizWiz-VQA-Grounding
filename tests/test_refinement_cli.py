import json
import tempfile
import unittest
from pathlib import Path
import torch
from PIL import Image
from research.checkpoint import save_checkpoint,read_checkpoint,initialize_weights
from research.model import ResearchGrounder
from research.engine import train_main,eval_main,training_parser,check_arguments
from research import variant


class RefinementCLITests(unittest.TestCase):
    def test_requires_initialized_frozen_reference(self):
        self.assertEqual(getattr(variant,'EXPERIMENT',None),'gain-guided-refinement')
        args=training_parser().parse_args(['--tiny','--image-size','28'])
        with self.assertRaisesRegex(ValueError,'init|resume'):
            check_arguments(args)

    def test_two_stage_initialization_resume_and_all_question_only_eval_policies(self):
        self.assertEqual(getattr(variant,'EXPERIMENT',None),'gain-guided-refinement')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for split in ('train','val'):
                (root/split).mkdir(); (root/'binary_masks_png'/split).mkdir(parents=True)
                records={}
                for i,size in enumerate(((17,11),(11,19))):
                    name=f'{i}.jpg'
                    Image.new('RGB',size,'red').save(root/split/name)
                    Image.new('L',size,255 if i==0 else 0).save(root/'binary_masks_png'/split/f'{i}.png')
                    records[name]={'question':'where?', 'most_common_answer':'privileged secret'}
                (root/f'{split}_grounding.json').write_text(json.dumps(records))
            coarse=ResearchGrounder(tiny=True)
            save_checkpoint(root/'control.pt',coarse,None,None,1,.1,{'text_mode':'question'})
            common=['--tiny','--device','cpu','--image-size','28','--detail-size','56','--crop-size','16',
                    '--data-root',str(root),'--num-workers','0','--batch-size','2']
            train_main(common+['--stage','refiner','--init-checkpoint',str(root/'control.pt'),
                              '--num-epochs','1','--output-dir',str(root/'refiner')])
            refiner=read_checkpoint(root/'refiner/last.pt')
            for k,v in coarse.state_dict().items():
                torch.testing.assert_close(refiner['model_state_dict']['coarse.'+k],v)
            args=training_parser().parse_args(common+['--stage','router','--init-checkpoint',str(root/'refiner/last.pt')])
            check_arguments(args)
            router=variant.build_model(args)
            initialize_weights(root/'refiner/last.pt',router)
            for k,v in router.state_dict().items():
                torch.testing.assert_close(v,refiner['model_state_dict'][k])
            train_main(common+['--stage','router','--init-checkpoint',str(root/'refiner/last.pt'),
                              '--num-epochs','1','--output-dir',str(root/'router')])
            router_checkpoint=read_checkpoint(root/'router/last.pt')
            for k,v in refiner['model_state_dict'].items():
                if k.startswith(('refiner.','coarse.')):
                    torch.testing.assert_close(router_checkpoint['model_state_dict'][k],v)
            train_main(common+['--stage','router','--resume-checkpoint',str(root/'router/last.pt'),
                              '--num-epochs','2','--output-dir',str(root/'router')])
            self.assertEqual(read_checkpoint(root/'router/last.pt')['epoch'],2)
            with self.assertRaisesRegex(ValueError,'configuration'):
                train_main(common+['--stage','router','--budget','4','--resume-checkpoint',str(root/'router/last.pt'),
                                  '--num-epochs','3','--output-dir',str(root/'router')])
            for policy,budget in [('gain',1),('uncertainty',2),('random',4),('fixed',2),('relevance',2)]:
                target=root/policy
                eval_main(['--checkpoint',str(root/'router/best.pt'),'--data-root',str(root),
                           '--device','cpu','--num-workers','0','--output-dir',str(target),
                           '--policy',policy,'--budget',str(budget)])
                metrics=json.loads((target/'metrics.json').read_text())['summary']
                self.assertEqual(metrics['text_mode'],'question')
                self.assertEqual(metrics['inference']['policy'],policy)
                self.assertEqual(metrics['inference']['budget'],budget)
                with Image.open(target/'0.png') as prediction:
                    self.assertEqual(prediction.size,(17,11))

if __name__=='__main__': unittest.main()
