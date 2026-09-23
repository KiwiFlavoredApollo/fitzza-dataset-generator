import json, csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "outputs" / "products_dataset_completed_v2_3.jsonl"

if not src.exists():
    raise SystemExit("Run v2.3 classifier first.")

rows = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
N = len(rows)

def n(field):
    return sum(bool(r.get(field)) for r in rows)

print("products:", N)
print("front:", n("has_front_image"), "/", N)
print("back:", n("has_back_image"), "/", N)
print("texture:", n("has_texture_image"), "/", N)
print("cutout:", n("has_transparent_cutout"), "/", N)
print("manual review:", n("manual_review_required"), "/", N)

reasons = Counter(x for r in rows for x in (r.get("manual_review_reasons") or []))
print("review reasons:", reasons)

csv_path = ROOT / "outputs" / "image_review_v2_3.csv"
fields = [
    "product_id","product_name","category",
    "front_url","front_prob","front_margin","front_ok",
    "back_url","back_prob","back_margin","back_ok",
    "texture_url","texture_prob","texture_margin","texture_ok",
    "cutout_path","cutout_ok","manual_review","review_reasons"
]
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        sel = r.get("selected_images") or {}
        fr, ba, te = sel.get("front") or {}, sel.get("back") or {}, sel.get("texture") or {}
        w.writerow({
            "product_id": r.get("product_id"),
            "product_name": r.get("product_name"),
            "category": r.get("category") or r.get("source_category"),
            "front_url": fr.get("image_url"),
            "front_prob": fr.get("role_probability"),
            "front_margin": fr.get("role_margin"),
            "front_ok": r.get("has_front_image"),
            "back_url": ba.get("image_url"),
            "back_prob": ba.get("role_probability"),
            "back_margin": ba.get("role_margin"),
            "back_ok": r.get("has_back_image"),
            "texture_url": te.get("image_url"),
            "texture_prob": te.get("role_probability"),
            "texture_margin": te.get("role_margin"),
            "texture_ok": r.get("has_texture_image"),
            "cutout_path": r.get("transparent_cutout_path"),
            "cutout_ok": r.get("has_transparent_cutout"),
            "manual_review": r.get("manual_review_required"),
            "review_reasons": ", ".join(r.get("manual_review_reasons") or []),
        })
print("csv:", csv_path)
