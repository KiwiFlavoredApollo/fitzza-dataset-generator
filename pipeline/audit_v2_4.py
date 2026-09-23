import json,csv
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
src=ROOT/"outputs"/"products_dataset_completed_v2_4.jsonl"
rows=[json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
N=len(rows)

def c(field): return sum(bool(r.get(field)) for r in rows)

print("products:",N)
print("front:",c("has_front_image"),"/",N)
print("back:",c("has_back_image"),"/",N)
print("texture:",c("has_texture_image"),"/",N)
print("cutout:",c("has_transparent_cutout"),"/",N)
print("manual review:",c("manual_review_required"),"/",N)
print("reasons:",Counter(x for r in rows for x in (r.get("manual_review_reasons") or [])))

csvp=ROOT/"outputs"/"image_review_v2_4.csv"
fields=["product_id","product_name","category",
"front_prob","front_confidence","front_ok","front_url",
"back_prob","back_confidence","back_ok","back_url",
"texture_prob","texture_confidence","texture_ok","texture_url",
"cutout_ok","cutout_path","manual_review","review_reasons"]
with csvp.open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for r in rows:
        s=r.get("selected_images") or {}
        fr=s.get("front") or {}; ba=s.get("back") or {}; te=s.get("texture") or {}
        w.writerow({
            "product_id":r.get("product_id"),"product_name":r.get("product_name"),
            "category":r.get("category") or r.get("source_category"),
            "front_prob":fr.get("role_probability"),"front_confidence":fr.get("confidence"),
            "front_ok":r.get("has_front_image"),"front_url":fr.get("image_url"),
            "back_prob":ba.get("role_probability"),"back_confidence":ba.get("confidence"),
            "back_ok":r.get("has_back_image"),"back_url":ba.get("image_url"),
            "texture_prob":te.get("role_probability"),"texture_confidence":te.get("confidence"),
            "texture_ok":r.get("has_texture_image"),"texture_url":te.get("image_url"),
            "cutout_ok":r.get("has_transparent_cutout"),"cutout_path":r.get("transparent_cutout_path"),
            "manual_review":r.get("manual_review_required"),
            "review_reasons":", ".join(r.get("manual_review_reasons") or [])
        })
print("csv:",csvp)
