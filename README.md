# Gain-guided refinement

This branch trains a compact grounding model with a high-resolution crop refiner and an expected-IoU-gain router. Inputs and evaluation use only the question. The coarse model stays frozen; the refiner is also frozen while training the router.

Install the project dependencies, or keep your notebook's Torch/Torchvision installation and install `transformers pillow numpy`. Run from this branch with a VizWiz root containing `<split>_grounding.json`, `<split>/` images, and `binary_masks_png/<split>/` masks.

```bash
python train.py --data-root /path/to/vizwiz --stage refiner --init-checkpoint /path/to/compact-controls/model_final_epoch100.pt --output-dir outputs-refiner --num-epochs 30 --batch-size 1
python train.py --data-root /path/to/vizwiz --stage router --init-checkpoint outputs-refiner/model_final_epoch30.pt --output-dir outputs-router --num-epochs 30 --batch-size 1
python eval.py --data-root /path/to/vizwiz --checkpoint outputs-router/model_final_epoch30.pt --dataset test --output-dir results-router --budget 2
```

The refiner starts from a question-only compact controls checkpoint. The router starts from the refiner stage. Either stage can also initialize from its own saved weights. Images use a 336-pixel coarse input and a 672-pixel detail canvas; crops use 336 pixels. Refiner evaluation uses fixed coverage, and router evaluation selects positive predicted gains. `--budget` chooses up to 1, 2, or 4 crops. Separate evaluation and exported masks use original image dimensions.

```bash
python predict_save.py --checkpoint outputs-router/model_final_epoch30.pt --image example.jpg --question "Where is the label?" --output mask.png
python visualize_predictions.py --checkpoint outputs-router/model_final_epoch30.pt --image example.jpg --question "Where is the label?" --output overlay.png
```

Use `train.py --help` and `eval.py --help` for operational options. Real-data accuracy and GPU throughput require training; tiny models are used only by local offline tests.

## Training output and separate evaluation

Training reads only the training split: no validation dataset is loaded and no validation runs, including after the final epoch. This also applies to teacher training and both refinement stages.

At completion, training saves exactly one raw `state_dict`: `model_final_epoch<N>.pt`, matching the baseline format. It contains model tensors only, with no optimizer, scaler, epoch wrapper, or RNG state. There are no automatic `last.pt` or `best.pt` saves, and `--validate-every` is removed. `--save-every` defaults to **10** and saves resumable `checkpoint_epoch10.pt`, `checkpoint_epoch20.pt`, etc., containing model, optimizer, scaler, and completed epoch. Use `--save-every 0` to disable these periodic checkpoints while still saving the final raw model.

Keep the existing small `config.json` beside the weight file when copying outputs to Kaggle or another directory. The evaluation, prediction, teacher, and refinement loaders use it to identify the architecture and stage. `history.jsonl` contains training losses only. Run `eval.py` separately with `--dataset val` or `--dataset test`; the commands above show the final weight filename for their epoch count.

To train further from these weights, use `--init-checkpoint /path/to/model_final_epoch<N>.pt` and a fresh output directory. This starts a new optimizer and epoch counter; `--num-epochs` is the number of additional epochs. Distillation still requires its teacher. To resume a periodic checkpoint, use `--resume-checkpoint /path/to/checkpoint_epoch10.pt` and a fresh output directory; optimizer/scaler state and completed epoch are restored, and `--num-epochs` is the final total epoch count. Older supported full checkpoints remain loadable.
