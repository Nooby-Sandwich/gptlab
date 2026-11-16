"""Dataset loading and random batch sampling."""
from pathlib import Path

import torch

from gptlab.tokenizer import CharTokenizer

def load_data(path, train_frac=0.9):
    """Read the text file, build the tokenizer, split into train/val tensors."""
    text = Path(path).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_frac * len(data))
    return tokenizer, data[:n], data[n:]


def get_batch(data, batch_size, block_size, device):
    """Sample random windows. y is x shifted one position to the right."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)