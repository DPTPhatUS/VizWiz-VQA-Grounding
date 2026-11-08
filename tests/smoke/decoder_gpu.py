"""Full-resolution decoder smoke; --device cpu works without a CUDA driver."""
import argparse
import json
import time

import torch
from models.mask_decoder import CompactDecoder
from models.text_encoder import TextFeatures
from losses import segmentation_loss, membership_loss, affinity_loss

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
parser.add_argument('--models', nargs='+', choices=('shared','separate','symmetric-affinity','two-role'),
                    default=['shared','separate','symmetric-affinity','two-role'])
args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(2)
for name in args.models:
    mode = 'shared' if name == 'shared' else 'separate'
    refinement = name if name in ('symmetric-affinity','two-role') else 'membership'
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = CompactDecoder(connection_mode=mode, local_refinement=refinement).to(device)
    features = [torch.randn(1, 1024, 24, 24, device=device, requires_grad=True) for _ in range(4)]
    attention = torch.ones(1, 20, dtype=torch.bool, device=device)
    valid = attention.clone(); valid[:, [0, -1]] = False
    text = TextFeatures(torch.randn(1, 20, 768, device=device), attention, valid)
    mask = torch.zeros(1, 1, 336, 336, device=device); mask[:, :, 90:110, 200:240] = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    if device.type == 'cuda':torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.autocast(device.type, enabled=device.type=='cuda'):
        output = model(*features, text)
        relation_loss = affinity_loss if refinement == 'symmetric-affinity' else membership_loss
        loss = segmentation_loss(output['logits'], mask) + .1 * relation_loss(output, mask)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(f.grad is not None and torch.isfinite(f.grad).all() for f in features)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    scaler.step(optimizer); scaler.update()
    if device.type == 'cuda':torch.cuda.synchronize()
    assert output['logits'].shape == (1, 1, 384, 384)
    assert output['membership_logits'].shape == (1, 1, 48, 48)
    if refinement == 'symmetric-affinity':
        assert output['affinity_logits'].shape == (1, 2304, 8)
    else:
        assert output['edge_logits'].shape == (1, 2304, 8, 2 if refinement=='two-role' else 3)
    print(json.dumps(dict(mode=mode, local_refinement=refinement, device=str(device), loss=loss.item(), decoder_parameters=sum(p.numel() for p in model.parameters()),
                         connection_parameters=sum(p.numel() for p in model.connections.parameters()),
                         seconds=time.perf_counter()-start, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None)), flush=True)
    del model, features, text, mask, optimizer, output, loss, scaler
