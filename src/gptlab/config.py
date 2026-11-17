"""Configuration for gptlab.

Model hyperparameters live in ModelConfig (saved inside checkpoints so a model
can be rebuilt later). Training-only settings live in TrainConfig.
Every field becomes a command line flag in train.py, e.g. --n_layer 8.
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 0      # filled in from the tokenizer at runtime
    block_size: int = 256    # max context length
    n_embd: int = 384        # embedding dimension
    n_layer: int = 6         # number of transformer blocks
    n_head: int = 6          # attention heads per block
    dropout: float = 0.2
    # attn: str = "sdpa"       # attention 
    # norm: str = "rsmnorm"    # normalisation
    attn: str = "manual"
    norm: str = "layernorm"
    mlp: str = "relu"          # multi layer perceptron 

@dataclass
class TrainConfig:
    data_path: str = "data/input.txt"
    out_dir: str = "checkpoints"
    run_name: str = "baseline"
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    learning_rate: float = 3e-4
    seed: int = 1337
    weight_decay: float = 0.1  # Penalize large weights
    warmup_iters: int = 200    # Linear warmup steps