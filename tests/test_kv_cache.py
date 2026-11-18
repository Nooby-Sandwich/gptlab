import pytest
import torch

from gptlab.config import ModelConfig
from gptlab.model import GPT


def make(pos):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=20, block_size=16, n_embd=32, n_layer=2,
                      n_head=2, attn="sdpa", pos=pos)
    return GPT(cfg).eval()


@pytest.mark.parametrize("pos", ["learned", "rope"])
def test_cached_logits_match_full_forward(pos):
    model = make(pos)
    idx = torch.randint(0, 20, (2, 12))
    with torch.no_grad():
        full, _ = model(idx)
        logits, past = model.forward_cached(idx[:, :8])      # prefill 8 tokens
        steps = [logits]
        for t in range(8, 12):                               # then one at a time
            logits, past = model.forward_cached(idx[:, t : t + 1], past)
            steps.append(logits)
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-4)


def test_cached_greedy_generation_matches_uncached():
    model = make("rope")
    start = torch.zeros((1, 1), dtype=torch.long)
    a = model.generate(start, 14, top_k=1, use_cache=True)   # 15 tokens <= block_size
    b = model.generate(start, 14, top_k=1, use_cache=False)
    assert torch.equal(a, b)


def test_generation_past_block_size_runs():
    out = make("rope").generate(torch.zeros((1, 1), dtype=torch.long), 40)
    assert out.shape == (1, 41)