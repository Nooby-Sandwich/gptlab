import torch
from gptlab.config import ModelConfig
from gptlab.model import FeedForward, SwiGLU


def test_swiglu_shape():
    cfg = ModelConfig(vocab_size=10, n_embd=384)
    x = torch.randn(2, 8, 384)
    assert SwiGLU(cfg)(x).shape == x.shape


def test_swiglu_param_count_matches_relu_mlp():
    cfg = ModelConfig(vocab_size=10, n_embd=384)
    a = sum(p.numel() for p in FeedForward(cfg).parameters())
    b = sum(p.numel() for p in SwiGLU(cfg).parameters())
    assert abs(a - b) / a < 0.01