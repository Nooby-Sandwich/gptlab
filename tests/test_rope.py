import torch
from gptlab.config import ModelConfig
from gptlab.model import GPT, apply_rope, rope_cache


def test_rope_preserves_vector_length():
    cos, sin = rope_cache(64, 16)
    x = torch.randn(2, 4, 16, 64)
    assert torch.allclose(x.norm(dim=-1), apply_rope(x, cos, sin).norm(dim=-1), atol=1e-4)


def test_rope_scores_depend_only_on_relative_distance():
    T, hd = 16, 64
    cos, sin = rope_cache(hd, T)
    q = torch.randn(1, 1, 1, hd).expand(1, 1, T, hd)  # same vector at every position
    k = torch.randn(1, 1, 1, hd).expand(1, 1, T, hd)
    s = (apply_rope(q, cos, sin) @ apply_rope(k, cos, sin).transpose(-2, -1))[0, 0]
    assert torch.allclose(s[2, 0], s[5, 3], atol=1e-4)  # distance 2
    assert torch.allclose(s[2, 0], s[9, 7], atol=1e-4)  # distance 2


def test_gpt_rope_forward():
    cfg = ModelConfig(vocab_size=20, block_size=16, n_embd=32, n_layer=2,
                      n_head=2, attn="sdpa", pos="rope")
    idx = torch.randint(0, 20, (2, 16))
    logits, loss = GPT(cfg)(idx, idx)
    assert logits.shape == (2, 16, 20) and loss.item() > 0