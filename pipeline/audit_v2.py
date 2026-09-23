import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
p = ROOT / cfg["output_dataset"]
rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
N = len(rows)

def count(field):
    return sum(bool(r.get(field)) for r in rows)

print("products:", N)
print("category:", Counter(r.get("category") for r in rows))
print("front:", count("has_front_image"), "/", N)
print("back:", count("has_back_image"), "/", N)
print("texture:", count("has_texture_image"), "/", N)
print("cutout:", count("has_transparent_cutout"), "/", N)
print("manual review:", count("manual_review_required"), "/", N)

reason_counter = Counter()
for r in rows:
    reason_counter.update(r.get("manual_review_reasons") or [])
print("review reasons:", reason_counter)

ready = sum(
    bool(r.get("has_front_image"))
    and bool(r.get("has_back_image"))
    and bool(r.get("has_texture_image"))
    and bool(r.get("has_transparent_cutout"))
    for r in rows
)
print("fully image-ready:", ready, "/", N, f"({ready/N*100:.1f}%)" if N else "")
