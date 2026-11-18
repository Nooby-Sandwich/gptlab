"""Build a multi-book Darwin dataset with a per-book 90/10 split."""
import re
from pathlib import Path

BOOKS = {2009: "Origin of Species",
         1227: "Expression of the Emotions",
         2300: "Descent of Man",
         3704: "Voyage of the Beagle"
        }
RAW, OUT = Path("data/raw"), Path("data/multi")


def clean(book_id):
    raw = (RAW / f"{book_id}.txt").read_text(encoding="utf-8-sig")
    start = raw.index("*** START OF")
    start = raw.index("\n", start) + 1
    body = raw[start : raw.index("*** END OF")]
    # Cut glossary/index at the back (alphabetical lists would skew validation).
    cuts = [m.start() for m in re.finditer(r"^\s*(GLOSSARY|INDEX)\.?\s*$", body, re.M)
            if m.start() > 0.85 * len(body)]
    if cuts:
        body = body[: min(cuts)]
        print(f"{BOOKS[book_id]}: cut back matter, kept {len(body):,} chars")
    else:
        print(f"{BOOKS[book_id]}: WARNING no index found, check the tail below")
    print("   tail:", repr(body.strip()[-80:]))
    return body.strip()


train_parts, val_parts = [], []
for book_id in BOOKS:
    body = clean(book_id)
    n = int(0.9 * len(body))
    train_parts.append(body[:n])
    val_parts.append(body[n:])

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "train.txt").write_text("\n\n".join(train_parts) + "\n", encoding="utf-8")
(OUT / "val.txt").write_text("\n\n".join(val_parts) + "\n", encoding="utf-8")
print("train chars:", sum(map(len, train_parts)), " val chars:", sum(map(len, val_parts)))