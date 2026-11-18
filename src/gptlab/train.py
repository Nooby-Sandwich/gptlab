"""Train the GPT.   Usage:  uv run python -m gptlab.train [--n_layer 8 ...]

Every field of ModelConfig / TrainConfig is a command line flag.
Logs a CSV (step, train loss, val loss, tokens/sec) and saves the checkpoint
with the best validation loss, including the config and vocabulary so the
model can be reloaded without the original dataset.
"""
import argparse
import csv
import time
from dataclasses import asdict, fields
from pathlib import Path
import math
import torch

from gptlab.config import ModelConfig, TrainConfig
from gptlab.data import get_batch, load_data
from gptlab.model import GPT


def parse_args():
    p = argparse.ArgumentParser()
    for cls in (ModelConfig, TrainConfig):
        for f in fields(cls):
            if f.name == "vocab_size":
                continue
            p.add_argument(f"--{f.name}", type=f.type, default=f.default)
    return p.parse_args()


@torch.no_grad()
def estimate_loss(model, splits, tcfg, block_size, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(tcfg.eval_iters)
        for k in range(tcfg.eval_iters):
            x, y = get_batch(data, tcfg.batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def get_lr(step, tcfg):
    # 1. Linear warmup
    if step < tcfg.warmup_iters:
        return tcfg.learning_rate * (step + 1) / tcfg.warmup_iters
    # 2. If past max_iters, stop decaying (safety)
    if step > tcfg.max_iters:
        return tcfg.learning_rate * 0.1
    # 3. Cosine decay
    decay_ratio = (step - tcfg.warmup_iters) / (tcfg.max_iters - tcfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    min_lr = tcfg.learning_rate * 0.1
    return min_lr + coeff * (tcfg.learning_rate - min_lr)


def main():
    args = parse_args()
    tcfg = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    mcfg = ModelConfig(**{f.name: getattr(args, f.name) for f in fields(ModelConfig)
                          if f.name != "vocab_size"})

    torch.manual_seed(tcfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    #tokenizer, train_data, val_data = load_data(tcfg.data_path)
    tokenizer, train_data, val_data = load_data(tcfg.data_path, tcfg.val_path)
    mcfg.vocab_size = tokenizer.vocab_size
    print(f"Data: {len(train_data) + len(val_data):,} chars, vocab {mcfg.vocab_size}")

    model = GPT(mcfg).to(device)
    print(f"Parameters: {model.num_params() / 1e6:.2f}M")
    #optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.learning_rate)
    # new optimizer with added weight_decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = tcfg.learning_rate,
        weight_decay=tcfg.weight_decay
    )



    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{tcfg.run_name}.pt"
    log_file = open(out_dir / f"{tcfg.run_name}_log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["step", "train_loss", "val_loss", "tokens_per_sec"])

    splits = {"train": train_data, "val": val_data}
    tokens_per_step = tcfg.batch_size * mcfg.block_size
    best_val = float("inf")
    sync(device)
    t_last, it_last = time.time(), 0

    for it in range(tcfg.max_iters + 1):
        if it % tcfg.eval_interval == 0 or it == tcfg.max_iters:
            sync(device)
            now = time.time()
            tps = (it - it_last) * tokens_per_step / (now - t_last) if it > it_last else 0.0

            losses = estimate_loss(model, splits, tcfg, mcfg.block_size, device)
            print(f"step {it:5d}: train {losses['train']:.4f}  val {losses['val']:.4f}"
                  f"  {tps:,.0f} tok/s")
            log.writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{tps:.0f}"])
            log_file.flush()

            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model": model.state_dict(),
                    "model_config": asdict(mcfg),
                    "chars": tokenizer.chars,
                    "step": it,
                    "val_loss": best_val,
                }, ckpt_path)

            sync(device)
            t_last, it_last = time.time(), it  # exclude eval time from throughput

        if it == tcfg.max_iters:
            break

        # Update the learning rate 
        lr = get_lr(it, tcfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        x, y = get_batch(train_data, tcfg.batch_size, mcfg.block_size, device)
        #_, loss = model(x, y)
        
        # NEW: Wrap the forward pass in autocast for bf16
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    log_file.close()
    print(f"Best val loss {best_val:.4f}. Checkpoint: {ckpt_path}")

    # Load the best checkpoint, not the overfit final one
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    # Quick sample from the final model
    model.eval()
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(ctx, max_new_tokens=400)[0].tolist()))


if __name__ == "__main__":
    main()