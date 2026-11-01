"""Opt-in full-resolution decoder smoke test; does not train a CLIP backbone."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from models.mask_decoder import CompactDecoder
from models.text_encoder import TextFeatures
from losses import segmentation_loss, membership_loss

torch.set_num_threads(2)
for mode in ('shared', 'separate'):
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = CompactDecoder(connection_mode=mode).cuda()
    features = [torch.randn(1, 1024, 24, 24, device='cuda', requires_grad=True) for _ in range(4)]
    attention = torch.ones(1, 20, dtype=torch.bool, device='cuda')
    valid = attention.clone(); valid[:, [0, -1]] = False
    text = TextFeatures(torch.randn(1, 20, 768, device='cuda'), attention, valid)
    mask = torch.zeros(1, 1, 336, 336, device='cuda'); mask[:, :, 90:110, 200:240] = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    torch.cuda.synchronize(); start = time.perf_counter()
    with torch.autocast('cuda'):
        output = model(*features, text)
        loss = segmentation_loss(output['logits'], mask) + .1 * membership_loss(output, mask)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(f.grad is not None and torch.isfinite(f.grad).all() for f in features)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    scaler.step(optimizer); scaler.update()
    torch.cuda.synchronize()
    assert output['logits'].shape == (1, 1, 384, 384)
    assert output['membership_logits'].shape == (1, 1, 48, 48)
    assert output['edge_logits'].shape == (1, 2304, 8, 3)
    print(json.dumps(dict(mode=mode, loss=loss.item(), decoder_parameters=sum(p.numel() for p in model.parameters()),
                         connection_parameters=sum(p.numel() for p in model.connections.parameters()),
                         seconds=time.perf_counter()-start, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)), flush=True)
    del model, features, text, mask, optimizer, output, loss, scaler
