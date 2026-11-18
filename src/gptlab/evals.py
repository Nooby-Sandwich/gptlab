"""Run evaluations on a trained checkpoint."""
import math
import torch
import difflib
from pathlib import Path

from gptlab.config import ModelConfig
from gptlab.model import GPT
from gptlab.tokenizer import CharTokenizer

def load_checkpoint(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = ModelConfig(**checkpoint["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = CharTokenizer(checkpoint["chars"])
    return model, tokenizer, cfg

@torch.no_grad()
def calculate_bpc(model, val_data, block_size, device):
    """Calculates Bits Per Character (BPC)."""
    # Take a random batch from the val data
    ix = torch.randint(len(val_data) - block_size, (1,))
    x = val_data[ix[0]:ix[0]+block_size].unsqueeze(0).to(device)
    y = val_data[ix[0]+1:ix[0]+block_size+1].unsqueeze(0).to(device)
    
    _, loss = model(x, y)
    # Convert nats to bits
    bpc = loss.item() / math.log(2)
    return bpc

def check_memorization(model, tokenizer, train_text, device, num_samples=10):
    """Generates text and checks for exact substring matches in training data."""
    print(f"\n--- Memorization Check ({num_samples} samples) ---")
    total_memorized = 0
    
    for _ in range(num_samples):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        out = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
        generated_text = tokenizer.decode(out[0].tolist())
        
        # Check for 20-character chunks that match the training data exactly
        match = difflib.SequenceMatcher(None, train_text, generated_text)
        longest_match = match.find_longest_match(0, len(train_text), 0, len(generated_text))
        
        if longest_match.size > 20:
            total_memorized += 1
            print(f"  [Memorized {longest_match.size} chars]: {generated_text[:50]}...")
        else:
            print(f"  [Novel]: {generated_text[:50]}...")
            
    mem_rate = (total_memorized / num_samples) * 100
    print(f"Memorization Rate: {mem_rate:.1f}% of samples contained >20 exact chars from training data.")
    return mem_rate


def run_prompt_suite(model, tokenizer, device):
    """Tests how the model handles specific prompts."""
    prompts = [
        "The struggle for existence",
        "Natural selection",
        "It is interesting to contemplate",
        "The giraffe"
    ]
    
    print("\n--- Prompt Completion Eval ---")
    for p in prompts:
        ids = tokenizer.encode(p)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=100, temperature=0.7, top_k=40)
        generated = tokenizer.decode(out[0].tolist())
        print(f"\nPrompt: '{p}'")
        print(f"Output: {generated.replace(chr(10), ' ')}")

def main(ckpt_path="checkpoints/swiglu_reg.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {ckpt_path} on {device}...")
    
    model, tokenizer, cfg = load_checkpoint(ckpt_path, device)
    
    # Load training data for memorization check
    train_text = Path("data/input.txt").read_text(encoding="utf-8")
    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    
    # 1. BPC
    bpc = calculate_bpc(model, train_data, cfg.block_size, device)
    print(f"\n--- Metric: Bits Per Character (BPC) ---")
    print(f"BPC: {bpc:.4f} (Lower is better. Random guessing is ~6.6. Perfect is 0.0)")
    
    # 2. Memorization
    check_memorization(model, tokenizer, train_text[:100000], device, num_samples=10) # Only check against first 100k chars for speed
    
    # 3. Prompt Suite
    run_prompt_suite(model, tokenizer, device)

if __name__ == "__main__":
    main()
