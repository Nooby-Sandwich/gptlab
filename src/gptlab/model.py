"""Baseline GPT: the same architecture as the original train.py, cleaned up.

Fixes vs. the original:
  * MultiHeadAttention is defined once (the original defined it twice).
  * Attention scores are scaled by head_size**-0.5 (the original used n_embd).
  * No global hyperparameters: everything comes from ModelConfig.
  * generate() supports temperature and top-k.

Later upgrades (SDPA, RMSNorm, SwiGLU, RoPE, KV cache) will be added on top of
this file and compared against this baseline.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from gptlab.config import ModelConfig


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, cfg: ModelConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        # Not a weight: persistent=False keeps it out of the saved checkpoint.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ v  # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)

# add SwiGLU
class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(2 * 4 * cfg.n_embd /3) # 1024 when n_embed = 384
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)  # gate branch
        self.w2 = nn.Linear(cfg.n_embd, hidden, bias=False)  # value branch
        self.w3 = nn.Linear(hidden, cfg.n_embd, bias=False)  # back down
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

def make_mlp(cfg: ModelConfig):
    if cfg.mlp == "relu":
        return FeedForward(cfg)
    if cfg.mlp == "swiglu":
        return SwiGLU(cfg)
    raise ValueError(f"Unknown mlp: {cfg.mlp!r}")


# RoPE Cache 
def rope_cache(head_dim, max_len, base=10000.0):
    """Rotation angles for every (position, frequency pair)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)   # (max_len, head_dim/2)
    return angles.cos(), angles.sin()


# apply RoPE
def apply_rope(x, cos, sin):
    """Rotate each consecutive pair of dims of x (B, heads, T, head_dim) by its position's angle."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.type_as(x)




class FusedAttention(nn.Module):
    """All heads at once, using Pytorch's fused fused attention kernel."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        # add RoPE
        self.use_rope = cfg.pos == "rope"
        if self.use_rope:
            cos, sin = rope_cache(cfg.n_embd // cfg.n_head, cfg.block_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
                        


    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, heads, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        # rotate queries and keys only, never values
        if self.use_rope:
            q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
            k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])
                
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def make_norm(cfg: ModelConfig):
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.n_embd)
    if cfg.norm == "layernorm":
        return nn.LayerNorm(cfg.n_embd)
    raise ValueError(f"Unknown norm: {cfg.norm!r}")
    

class Block(nn.Module):
    """Transformer block: communication (attention) then computation (MLP)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        #self.sa = MultiHeadAttention(cfg)
        self.sa = FusedAttention(cfg) if cfg.attn == "sdpa" else MultiHeadAttention(cfg)
        #self.ffwd = FeedForward(cfg)
        self.ffwd = make_mlp(cfg)
        # self.ln1 = nn.LayerNorm(cfg.n_embd)
        # self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ln1 = make_norm(cfg)
        self.ln2 = make_norm(cfg)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    # def __init__(self, cfg: ModelConfig):
    #     super().__init__()
    #     self.cfg = cfg
    #     self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
    #     self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
    #     self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
    #     #self.ln_f = nn.LayerNorm(cfg.n_embd)
    #     self.ln_f = make_norm(cfg)
    #     self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.attn not in ("manual", "sdpa"):
            raise ValueError(f"Unknown attn: {cfg.attn!r}")
        if cfg.pos not in ("learned", "rope"):
            raise ValueError(f"Unknown pos: {cfg.pos!r}")
        if cfg.pos == "rope" and cfg.attn != "sdpa":
            raise ValueError("--pos rope needs --attn sdpa")
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if cfg.pos == "learned":
            self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        # x = self.token_embedding_table(idx) + pos_emb  # (B, T, C)

        x = self.token_embedding_table(idx)  # (B, T, C)

        if self.cfg.pos == "learned":
            pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
            x = x + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1), targets.view(B * T)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively extend idx (B, T). Call model.eval() first."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())