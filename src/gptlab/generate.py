"""Generate text from a trained checkpoint.

Usage:
  uv run python -m gptlab.generate --prompt "The struggle for" --temperature 0.8 --top_k 40
"""
import argparse

import torch

from gptlab.config import ModelConfig
from gptlab.model import GPT
from gptlab.tokenizer import CharTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/baseline.pt")
    p.add_argument("--prompt", default="")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = CharTokenizer(ckpt["chars"])

    if args.prompt:
        try:
            ids = tokenizer.encode(args.prompt)
        except KeyError as e:
            raise SystemExit(f"Prompt has a character not in the vocabulary: {e}")
    else:
        ids = [0]  # start from the first vocabulary symbol (a newline)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(
        idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()