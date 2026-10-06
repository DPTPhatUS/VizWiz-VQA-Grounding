import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import torch
from research import variant

@unittest.skipUnless(getattr(variant, "EXPERIMENT", "controls") == "controls",
                     "variant branch has its own CLI smoke test")
class CLITests(unittest.TestCase):
    def run_cli(self, *args):
        env=dict(os.environ, OMP_NUM_THREADS="1", MPLCONFIGDIR="/tmp/vizwiz-mpl")
        result=subprocess.run([sys.executable,*args],capture_output=True,text=True,env=env)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return result

    def test_training_resume_and_original_resolution_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for split in ("train","val"):
                (root/split).mkdir(); (root/"binary_masks_png"/split).mkdir(parents=True)
                rows={}
                for i,size in enumerate([(17,11),(11,19)]):
                    name=f"{i}.jpg"; Image.new("RGB",size,"red").save(root/split/name)
                    Image.new("L",size,255).save(root/"binary_masks_png"/split/f"{i}.png")
                    rows[name]={"question":"what?","most_common_answer":"red"}
                (root/f"{split}_grounding.json").write_text(json.dumps(rows))
            out=root/"out"
            common=["--tiny","--image-size","28","--data-root",str(root),
                    "--num-workers","0","--batch-size","2","--output-dir",str(out)]
            self.run_cli("train_research.py",*common,"--num-epochs","1")
            self.assertTrue((out/"best.pt").exists())
            self.run_cli("train_research.py",*common,"--num-epochs","2",
                         "--resume-checkpoint",str(out/"last.pt"))
            ckpt=torch.load(out/"last.pt",weights_only=False)
            self.assertEqual(ckpt["epoch"],2)
            self.run_cli("eval_research.py","--checkpoint",str(out/"best.pt"),
                         "--data-root",str(root),"--num-workers","0",
                         "--output-dir",str(root/"pred"))
            with Image.open(root/"pred/0.png") as prediction:
                self.assertEqual(prediction.size,(17,11))
            results=json.loads((root/"pred/metrics.json").read_text())
            self.assertEqual(results["summary"]["num_samples"],2)
            self.assertEqual(results["summary"]["text_mode"],"question")

if __name__=="__main__": unittest.main()
