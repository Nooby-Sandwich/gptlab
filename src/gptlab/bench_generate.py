"""Compare generation speed with and without the KV cache.

Usage: uv run python -m gptlab.bench_generate --ckpt checkpoints/rope_reg.pt [--device cpu]
Run it with nothing else using the GPU.
"""
import argparse
import time

import torch

from gptlab.config import ModelConfig
from gptlab.model import GPT


def timed(model, ctx, n, use_cache, device):
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    model.generate(ctx, n, use_cache=use_cache)
    if device == "cuda":
        torch.cuda.synchronize()
    return n / (time.time() - t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/rope_reg.pt")
    p.add_argument("--tokens", type=int, default=200)   # +1 prompt token stays under block_size
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = GPT(ModelConfig(**ckpt["model_config"])).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ctx = torch.zeros((1, 1), dtype=torch.long, device=args.device)

    for flag in (False, True):  # warm-up: CUDA start-up, kernel selection
        timed(model, ctx, 20, flag, args.device)

    results = {False: [], True: []}
    for _ in range(args.rounds):  # alternate modes so thermal drift hits both equally
        for flag in (False, True):
            results[flag].append(timed(model, ctx, args.tokens, flag, args.device))

    slow, fast = max(results[False]), max(results[True])
    print(f"no cache: {slow:8.1f} tokens/s")
    print(f"cache:    {fast:8.1f} tokens/s   ({fast / slow:.1f}x)")


if __name__ == "__main__":
    main()