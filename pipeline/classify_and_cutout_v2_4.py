from __future__ import annotations
import argparse, json, hashlib, traceback
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, SegformerImageProcessor, AutoModelForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]

BINARY_PROMPTS = {
    "front": (
        [
            "the front side of a fashion product facing the camera",
            "front view of the clothing or fashion item",
            "a product photo showing the front of the garment"
        ],
        [
            "the back or side of a fashion product",
            "rear view or side view of the clothing item",
            "a product photo not showing the front side"
        ]
    ),
    "back": (
        [
            "the back side of a fashion product facing the camera",
            "rear view of the clothing or fashion item",
            "a product photo showing the back of the garment"
        ],
        [
            "the front or side of a fashion product",
            "front view or side view of the clothing item",
            "a product photo not showing the back side"
        ]
    ),
    "texture": (
        [
            "a close-up macro image of fabric texture or material surface",
            "close-up textile detail showing weave grain wash or surface",
            "a detailed material texture photo"
        ],
        [
            "a full product view or model wearing the product",
            "a normal fashion product photo not focused on material texture",
            "a full-body or full-item fashion photo"
        ]
    ),
    "product_only": (
        [
            "a fashion product by itself with no person wearing it",
            "an isolated ecommerce product-only photo",
            "a single clothing item shoe or hat displayed alone"
        ],
        [
            "a person or model wearing the fashion product",
            "a lookbook photo with a human model",
            "fashion item worn by a person"
        ]
    ),
}

CATEGORY_TARGET_LABELS = {
    "top": {"upper-clothes", "shirt"},
    "outer": {"upper-clothes", "coat"},
    "bottom": {"pants", "skirt"},
    "shoes": {"left-shoe", "right-shoe"},
    "hat": {"hat"},
}

def load_config():
    return json.loads((ROOT/"config.json").read_text(encoding="utf-8"))

def device(cfg):
    if cfg.get("device") != "auto":
        return cfg["device"]
    return "cuda" if torch.cuda.is_available() else "cpu"

def local_path(cache, pid, url):
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in {".jpg",".jpeg",".png",".webp"}:
        suffix = ".jpg"
    h = hashlib.sha1(url.encode()).hexdigest()[:14]
    return cache / str(pid) / f"{h}{suffix}"

def candidates(p, cfg):
    imgs = p.get("images") or []
    thumb = [x for x in imgs if x.get("image_source_type") == "thumbnail"]
    gallery = [x for x in imgs if x.get("image_source_type") == "goods_gallery"]
    detail = [x for x in imgs if x.get("image_source_type") == "goods_contents"][:int(cfg["max_goods_contents_per_product"])]
    seq = thumb + gallery + detail
    seen, out = set(), []
    for x in seq:
        u = x.get("image_url")
        if u and u not in seen:
            seen.add(u); out.append(x)
        if len(out) >= int(cfg["max_images_per_product"]):
            break
    return out

def open_rgb(p):
    return ImageOps.exif_transpose(Image.open(p)).convert("RGB")

@torch.inference_mode()
def enc_text(model, processor, prompts, dev):
    inp = processor(text=prompts, return_tensors="pt", padding=True)
    out = model.text_model(
        input_ids=inp["input_ids"].to(dev),
        attention_mask=inp["attention_mask"].to(dev)
    )
    feat = model.text_projection(out.pooler_output)
    return feat / feat.norm(dim=-1, keepdim=True)

@torch.inference_mode()
def enc_img(model, processor, images, dev):
    inp = processor(images=images, return_tensors="pt")
    out = model.vision_model(pixel_values=inp["pixel_values"].to(dev))
    feat = model.visual_projection(out.pooler_output)
    return feat / feat.norm(dim=-1, keepdim=True)

def build_binary_vectors(model, processor, dev):
    vecs = {}
    for role, (pos, neg) in BINARY_PROMPTS.items():
        pv = enc_text(model, processor, pos, dev).mean(dim=0)
        nv = enc_text(model, processor, neg, dev).mean(dim=0)
        pv = pv / pv.norm()
        nv = nv / nv.norm()
        vecs[role] = (pv, nv)
    return vecs

@torch.inference_mode()
def binary_scores(images, model, processor, vecs, dev):
    f = enc_img(model, processor, images, dev)
    out = []
    for i in range(f.shape[0]):
        row = {}
        for role, (pv, nv) in vecs.items():
            pos = torch.dot(f[i], pv)
            neg = torch.dot(f[i], nv)
            prob = torch.sigmoid((pos-neg) * 14.0).item()
            row[role] = {
                "prob": float(prob),
                "pos_cos": float(pos.item()),
                "neg_cos": float(neg.item()),
                "margin": float((pos-neg).item())
            }
        out.append(row)
    return out

def confidence(prob, cfg):
    if prob >= float(cfg["selection_confidence_high"]):
        return "high"
    if prob >= float(cfg["selection_confidence_medium"]):
        return "medium"
    return "low"

def best_role(scored, role, cfg, excluded=None):
    excluded = set(excluded or [])
    pool = [x for x in scored if x["image_url"] not in excluded]
    if not pool:
        return None

    def rank(x):
        p = x["binary"][role]["prob"]
        bonus = 0.0
        if role in {"front","back"} and x["source_type"] in {"thumbnail","goods_gallery"}:
            bonus += 0.035
        if role == "texture" and x["source_type"] == "goods_contents":
            bonus += 0.08
        if role == "front" and x.get("is_thumbnail"):
            bonus += 0.05
        return p + bonus

    b = max(pool, key=rank)
    out = dict(b)
    out["role_probability"] = float(b["binary"][role]["prob"])
    out["role_margin"] = float(b["binary"][role]["margin"])
    out["selection_score"] = float(rank(b))
    out["confidence"] = confidence(out["role_probability"], cfg)
    threshold = float(cfg[f"{role}_binary_min_prob"])
    out["meets_threshold"] = out["role_probability"] >= threshold
    return out

def choose_front(scored, cfg):
    # Thumbnail is a strong prior for ecommerce front/representative view,
    # but CLIP must still support it. Otherwise fall back to best gallery candidate.
    thumbs = [x for x in scored if x.get("is_thumbnail")]
    if thumbs:
        t = thumbs[0]
        if t["binary"]["front"]["prob"] >= 0.48:
            out = dict(t)
            out["role_probability"] = float(t["binary"]["front"]["prob"])
            out["role_margin"] = float(t["binary"]["front"]["margin"])
            out["selection_score"] = out["role_probability"] + 0.05
            out["confidence"] = confidence(out["role_probability"], cfg)
            out["meets_threshold"] = out["role_probability"] >= float(cfg["front_binary_min_prob"])
            return out
    return best_role(scored, "front", cfg)

def id2label_map(model):
    return {int(k): str(v).lower() for k,v in model.config.id2label.items()}

@torch.inference_mode()
def cutout(image, category, processor, model, dev):
    inp = processor(images=image, return_tensors="pt").to(dev)
    out = model(**inp)
    logits = torch.nn.functional.interpolate(out.logits, size=(image.height,image.width), mode="bilinear", align_corners=False)
    pred = logits.argmax(dim=1)[0].cpu().numpy()
    labels = id2label_map(model)
    ids = [idx for idx,name in labels.items() if name in CATEGORY_TARGET_LABELS.get(category,set())]
    if not ids:
        return None, 0.0
    mask = np.isin(pred, ids).astype(np.uint8)*255
    coverage = float((mask>0).mean())
    if coverage < 0.015:
        return None, coverage
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask, mode="L"))
    return rgba, coverage

def ref(x):
    if not x: return None
    return {
        "image_url": x["image_url"],
        "local_path": x["local_path"],
        "source_type": x["source_type"],
        "role_probability": x["role_probability"],
        "role_margin": x["role_margin"],
        "selection_score": x["selection_score"],
        "confidence": x["confidence"],
        "meets_threshold": x["meets_threshold"],
        "binary": x["binary"],
    }

def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False)+"\n")

def done_ids(path):
    if not path.exists(): return set()
    s=set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try: s.add(str(json.loads(line)["product_id"]))
            except: pass
    return s

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fresh", action="store_true")
    args=ap.parse_args()

    cfg=load_config()
    dev=device(cfg)
    dataset=ROOT/cfg["dataset_path"]
    cache=ROOT/cfg["cache_dir"]
    out=ROOT/"outputs"/"products_dataset_completed_v2_4.jsonl"
    cutdir=ROOT/"outputs"/"cutouts_v2_4"
    err=ROOT/"outputs"/"classification_errors_v2_4.jsonl"
    cutdir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        for p in [out,err]:
            if p.exists(): p.unlink()

    rows=[json.loads(x) for x in dataset.read_text(encoding="utf-8").splitlines() if x.strip()]
    done=done_ids(out)

    print("device:",dev)
    print("loading models...")
    clip=CLIPModel.from_pretrained(cfg["classification_model"]).to(dev).eval()
    proc=CLIPProcessor.from_pretrained(cfg["classification_model"])
    vecs=build_binary_vectors(clip,proc,dev)
    segp=SegformerImageProcessor.from_pretrained(cfg["segmentation_model"])
    segm=AutoModelForSemanticSegmentation.from_pretrained(cfg["segmentation_model"]).to(dev).eval()
    print("models loaded")

    todo=[r for r in rows if str(r["product_id"]) not in done]
    if args.limit: todo=todo[:args.limit]
    print("to process:",len(todo))

    for p in tqdm(todo, desc="products"):
        pid=str(p["product_id"])
        try:
            av=[]
            for meta in candidates(p,cfg):
                lp=local_path(cache,pid,meta["image_url"])
                if not lp.exists(): continue
                try:
                    im=open_rgb(lp)
                    if im.width>=160 and im.height>=160:
                        av.append((meta,lp,im))
                except: pass
            if not av: raise RuntimeError("no cached images")

            scores=binary_scores([x[2] for x in av],clip,proc,vecs,dev)
            scored=[]
            for (meta,lp,_), sc in zip(av,scores):
                scored.append({
                    "image_url":meta["image_url"],
                    "local_path":str(lp.relative_to(ROOT)),
                    "source_type":meta.get("image_source_type"),
                    "is_thumbnail":bool(meta.get("is_thumbnail")),
                    "binary":sc,
                })

            front=choose_front(scored,cfg)
            back=best_role(scored,"back",cfg,[front["image_url"]] if front else None)
            texture=best_role(scored,"texture",cfg,[x["image_url"] for x in [front,back] if x])
            product_only=best_role(scored,"product_only",cfg)

            cut_src=product_only if product_only and product_only["meets_threshold"] else front
            cut_path=None; coverage=None
            if cut_src:
                try:
                    im=open_rgb(ROOT/cut_src["local_path"])
                    rgba,coverage=cutout(im,p.get("category") or p.get("source_category"),segp,segm,dev)
                    if rgba is not None:
                        cp=cutdir/f"{pid}.png"
                        rgba.save(cp)
                        cut_path=str(cp.relative_to(ROOT))
                except Exception as e:
                    append(err,{"product_id":pid,"stage":"cutout","error":repr(e)})

            reasons=[]
            if not front or not front["meets_threshold"]: reasons.append("front_review")
            if not back or not back["meets_threshold"]: reasons.append("back_review")
            if not texture or not texture["meets_threshold"]: reasons.append("texture_review")
            if not cut_path: reasons.append("cutout_failed")

            q=dict(p)
            q["image_classification_version"]="v2.4_binary_role_scoring"
            q["selected_images"]={"front":ref(front),"back":ref(back),"texture":ref(texture)}
            q["transparent_cutout_path"]=cut_path
            q["transparent_cutout_mask_coverage"]=coverage
            q["has_front_image"]=bool(front and front["meets_threshold"])
            q["has_back_image"]=bool(back and back["meets_threshold"])
            q["has_texture_image"]=bool(texture and texture["meets_threshold"])
            q["has_transparent_cutout"]=bool(cut_path)
            q["manual_review_required"]=bool(reasons)
            q["manual_review_reasons"]=reasons
            q["image_classification_candidates"]=scored
            append(out,q)

        except Exception as e:
            append(err,{"product_id":pid,"stage":"product","error":repr(e),"traceback":traceback.format_exc(limit=4)})

    print("DONE")
    print(out)

if __name__=="__main__":
    main()
