"""Plot validation-loss curves for every run in results/*_log.csv."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ORDER = ["baseline", "sdpa", "rmsnorm", "swiglu", "swiglu_reg", "rope_reg"]
NEW_RECIPE = {"swiglu_reg", "rope_reg"}   # warmup + cosine LR, weight decay, bf16


def load(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["step"]) >= 1500]
    return [int(r["step"]) for r in rows], [float(r["val_loss"]) for r in rows]


paths = {p.stem.removesuffix("_log"): p for p in Path("results").glob("*_log.csv")}
# multi-book runs use a different validation set, so they get their own plot
paths = {n: p for n, p in paths.items() if not n.startswith("multi")}
names = [n for n in ORDER if n in paths] + sorted(n for n in paths if n not in ORDER)

fig, ax = plt.subplots(figsize=(9, 5))
for i, name in enumerate(names):
    steps, vals = load(paths[name])
    ax.plot(steps, vals, marker="o", color=f"C{i}",
            linestyle="-" if name in NEW_RECIPE else "--", label=name)

ax.set_ylim(0.995, 1.165)
ax.set_xlabel("training step")
ax.set_ylabel("validation loss")
ax.set_title("Validation loss by run (Darwin, Origin of Species)")
ax.grid(alpha=0.3)
ax.legend(title="dashed = old recipe\nsolid = new recipe",
          loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.savefig("results/val_loss.png", dpi=150, bbox_inches="tight")
print("Wrote results/val_loss.png")