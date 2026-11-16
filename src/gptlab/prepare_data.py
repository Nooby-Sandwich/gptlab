from pathlib import Path

raw = Path("data/darwin_raw.txt").read_text(encoding="utf-8-sig")

start = raw.index("*** START OF")
start = raw.index("\n", start) + 1   # skip the marker line itself
end = raw.index("*** END OF")
body = raw[start:end]

# Drop the glossary and alphabetical index at the back of the book.
cut = body.rindex("GLOSSARY")
print(f"Cutting at {cut / len(body):.3f} of the text")

Path("data/input.txt").write_text(body[:cut].strip() + "\n", encoding="utf-8")
print("Wrote data/input.txt")