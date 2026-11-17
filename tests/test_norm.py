import torch
from gptlab.config import ModelConfig
from gptlab.model import GPT, RMSNorm


def test_rmsnorm_flag_applies_everywhere():
    cfg = ModelConfig(vocab_size=10, block_size=8, n_embd=32,
                      n_layer=2, n_head=2, norm="rmsnorm")
    norms = [m for m in GPT(cfg).modules()
             if isinstance(m, (RMSNorm, torch.nn.LayerNorm))]
    assert len(norms) == 2 * 2 + 1          # ln1 + ln2 per layer, plus ln_f
    assert all(isinstance(m, RMSNorm) for m in norms)