from __future__ import annotations
import argparse, json, hashlib, traceback
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import (
    CLIPModel, CLIPProcessor,
    SegformerImageProcessor, AutoModelForSemanticSegmentation
)

ROOT = Path(__file__).resolve().parents[1]

VIEW_PROMPTS = {
    "front": [
        "a fashion product seen directly from the front",
        "front view of clothing, shoes, or a hat for an online store",
        "the front side of the fashion item is facing the camera",
    ],
    "back": [
        "a fashion product seen directly from the back",
        "back view of clothing, shoes, or a hat for an online store",
        "the rear side of the fashion item is facing the camera",
    ],
    "side_angled": [
        "a side view or angled three-quarter view of a fashion product",
        "fashion item photographed from the side or an oblique angle",
    ],
}

SCENE_PROMPTS = {
    "product_only": [
        "a single isolated fashion product with no person wearing it",
        "an e-commerce product-only photo of clothing, shoes, or a hat",
        "a fashion item displayed by itself",
    ],
    "model_worn": [
        "a fashion model or person wearing the product",
        "lookbook photo of a person wearing clothing or accessories",
    ],
    "texture": [
        "a close-up macro photo of fabric texture or material surface",
        "close-up textile detail showing weave, denim wash, leather grain, or nylon surface",
        "detail shot of fabric or material texture",
    ],
    "other": [
        "a size chart, brand graphic, packaging, poster, or informational image",
        "an image that is not a useful fashion product photograph",
    ],
}

CATEGORY_TARGET_LABELS = {
    "top": {"upper-clothes", "shirt"},
    "outer": {"upper-clothes", "coat"},
    "bottom": {"pants", "skirt"},
    "shoes": {"left-shoe", "right-shoe"},
    "hat": {"hat"},
}

def load_config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

def get_device(cfg):
    if cfg.get("device") != "auto":
        return cfg["device"]
    return "cuda" if torch.cuda.is_available() else "cpu"

def local_image_path(cache_dir: Path, product_id: str, url: str):
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:14]
    return cache_dir / str(product_id) / f"{h}{suffix}"

def candidate_images(product, cfg):
    images = product.get("images") or []
    thumb = [x for x in images if x.get("image_source_type") == "thumbnail"]
    gallery = [x for x in images if x.get("image_source_type") == "goods_gallery"]
    details = [x for x in images if x.get("image_source_type") == "goods_contents"]
    details = details[: int(cfg["max_goods_contents_per_product"])]
    seq = thumb + gallery + details

    seen, out = set(), []
    for x in seq:
        u = x.get("image_url")
        if u and u not in seen:
            seen.add(u)
            out.append(x)
        if len(out) >= int(cfg["max_images_per_product"]):
            break
    return out

def open_rgb(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")

@torch.inference_mode()
def encode_text(model, processor, prompts, device):
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    out = model.text_model(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
    )
    feat = model.text_projection(out.pooler_output)
    return feat / feat.norm(dim=-1, keepdim=True)

@torch.inference_mode()
def encode_images(model, processor, images, device):
    inputs = processor(images=images, return_tensors="pt")
    out = model.vision_model(pixel_values=inputs["pixel_values"].to(device))
    feat = model.visual_projection(out.pooler_output)
    return feat / feat.norm(dim=-1, keepdim=True)

def build_group_vectors(model, processor, prompt_groups, device):
    labels = list(prompt_groups)
    vecs = []
    for label in labels:
        prompt_vecs = encode_text(model, processor, prompt_groups[label], device)
        v = prompt_vecs.mean(dim=0)
        v = v / v.norm()
        vecs.append(v)
    return labels, torch.stack(vecs)

@torch.inference_mode()
def classify_group(images, model, processor, labels, vectors, device, temperature=12.0):
    img_feat = encode_images(model, processor, images, device)
    logits = img_feat @ vectors.T
    probs = torch.softmax(logits * temperature, dim=-1).cpu().numpy()
    cosine = logits.cpu().numpy()
    return [
        {
            "prob": {labels[i]: float(p[i]) for i in range(len(labels))},
            "cosine": {labels[i]: float(c[i]) for i in range(len(labels))},
        }
        for p, c in zip(probs, cosine)
    ]

def view_confidence(view_prob, role, cfg):
    own = float(view_prob.get(role, 0.0))
    others = [float(v) for k, v in view_prob.items() if k != role]
    margin = own - max(others or [0.0])
    min_prob = float(cfg["view_front_min_prob"] if role == "front" else cfg["view_back_min_prob"])
    passed = own >= min_prob and margin >= float(cfg["view_min_margin"])
    return own, margin, passed

def choose_front_or_back(scored, role, cfg, excluded=None):
    excluded = set(excluded or [])
    pool = [x for x in scored if x["image_url"] not in excluded]
    if not pool:
        return None

    def rank(x):
        own, margin, _ = view_confidence(x["view_prob"], role, cfg)
        source_bonus = 0.05 if x["source_type"] in {"thumbnail", "goods_gallery"} else 0.0
        product_bonus = x["scene_prob"].get("product_only", 0.0) * 0.12
        return own + max(margin, -0.10) * 0.35 + product_bonus + source_bonus

    best = max(pool, key=rank)
    own, margin, passed = view_confidence(best["view_prob"], role, cfg)
    out = dict(best)
    out["selection_score"] = float(rank(best))
    out["role_probability"] = own
    out["role_margin"] = float(margin)
    out["meets_threshold"] = bool(passed)
    return out

def choose_texture(scored, cfg, excluded=None):
    excluded = set(excluded or [])
    pool = [x for x in scored if x["image_url"] not in excluded]
    if not pool:
        return None

    def rank(x):
        p = float(x["scene_prob"].get("texture", 0.0))
        return p + (0.08 if x["source_type"] == "goods_contents" else 0.0)

    best = max(pool, key=rank)
    out = dict(best)
    out["selection_score"] = float(rank(best))
    out["role_probability"] = float(best["scene_prob"].get("texture", 0.0))
    out["role_margin"] = float(
        best["scene_prob"].get("texture", 0.0)
        - max(v for k, v in best["scene_prob"].items() if k != "texture")
    )
    out["meets_threshold"] = out["role_probability"] >= float(cfg["texture_min_prob"])
    return out

def choose_product_only(scored, cfg):
    if not scored:
        return None
    best = max(
        scored,
        key=lambda x: float(x["scene_prob"].get("product_only", 0.0))
        + (0.05 if x["source_type"] in {"thumbnail", "goods_gallery"} else 0.0)
    )
    out = dict(best)
    out["selection_score"] = float(best["scene_prob"].get("product_only", 0.0))
    out["role_probability"] = float(best["scene_prob"].get("product_only", 0.0))
    out["role_margin"] = float(
        best["scene_prob"].get("product_only", 0.0)
        - max(v for k, v in best["scene_prob"].items() if k != "product_only")
    )
    out["meets_threshold"] = out["role_probability"] >= float(cfg["product_only_min_prob"])
    return out

def id2label_map(model):
    return {int(k): str(v).lower() for k, v in model.config.id2label.items()}

@torch.inference_mode()
def segment_cutout(image, category, processor, model, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model(**inputs)
    logits = torch.nn.functional.interpolate(
        out.logits, size=(image.height, image.width),
        mode="bilinear", align_corners=False
    )
    pred = logits.argmax(dim=1)[0].cpu().numpy()

    labels = id2label_map(model)
    wanted = CATEGORY_TARGET_LABELS.get(category, set())
    ids = [idx for idx, name in labels.items() if name in wanted]
    if not ids:
        return None, 0.0

    mask = np.isin(pred, ids).astype(np.uint8) * 255
    coverage = float((mask > 0).mean())
    if coverage < 0.015:
        return None, coverage

    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask, mode="L"))
    return rgba, coverage

def ref(x):
    if not x:
        return None
    return {
        "image_url": x["image_url"],
        "local_path": x["local_path"],
        "source_type": x["source_type"],
        "view_prob": x["view_prob"],
        "scene_prob": x["scene_prob"],
        "role_probability": x["role_probability"],
        "role_margin": x["role_margin"],
        "selection_score": x["selection_score"],
        "meets_threshold": x["meets_threshold"],
    }

def read_done(path):
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(str(json.loads(line)["product_id"]))
        except Exception:
            pass
    return done

def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    device = get_device(cfg)

    dataset = ROOT / cfg["dataset_path"]
    cache = ROOT / cfg["cache_dir"]
    output = ROOT / "outputs" / "products_dataset_completed_v2_3.jsonl"
    cutout_dir = ROOT / "outputs" / "cutouts_v2_3"
    error_log = ROOT / "outputs" / "classification_errors_v2_3.jsonl"

    output.parent.mkdir(parents=True, exist_ok=True)
    cutout_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        for p in [output, error_log]:
            if p.exists():
                p.unlink()

    rows = [json.loads(x) for x in dataset.read_text(encoding="utf-8").splitlines() if x.strip()]
    done = read_done(output)

    print("device:", device)
    print("dataset:", len(rows))
    print("already processed:", len(done))
    print("loading CLIP...")

    clip_model = CLIPModel.from_pretrained(cfg["classification_model"]).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(cfg["classification_model"])
    view_labels, view_vectors = build_group_vectors(clip_model, clip_processor, VIEW_PROMPTS, device)
    scene_labels, scene_vectors = build_group_vectors(clip_model, clip_processor, SCENE_PROMPTS, device)

    print("loading segmentation model...")
    seg_processor = SegformerImageProcessor.from_pretrained(cfg["segmentation_model"])
    seg_model = AutoModelForSemanticSegmentation.from_pretrained(cfg["segmentation_model"]).to(device).eval()
    print("models loaded")

    todo = [r for r in rows if str(r["product_id"]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print("to process:", len(todo))

    for p in tqdm(todo, desc="products"):
        pid = str(p["product_id"])
        try:
            available = []
            for meta in candidate_images(p, cfg):
                lp = local_image_path(cache, pid, meta["image_url"])
                if not lp.exists():
                    continue
                try:
                    im = open_rgb(lp)
                    if im.width >= 160 and im.height >= 160:
                        available.append((meta, lp, im))
                except Exception:
                    pass

            if not available:
                raise RuntimeError("no cached candidate images for product")

            images = [x[2] for x in available]
            view_results = classify_group(images, clip_model, clip_processor, view_labels, view_vectors, device)
            scene_results = classify_group(images, clip_model, clip_processor, scene_labels, scene_vectors, device)

            scored = []
            for (meta, lp, _), vr, sr in zip(available, view_results, scene_results):
                scored.append({
                    "image_url": meta["image_url"],
                    "local_path": str(lp.relative_to(ROOT)),
                    "source_type": meta.get("image_source_type"),
                    "view_prob": vr["prob"],
                    "scene_prob": sr["prob"],
                })

            front = choose_front_or_back(scored, "front", cfg)
            back = choose_front_or_back(scored, "back", cfg, excluded=[front["image_url"]] if front else None)
            texture = choose_texture(scored, cfg, excluded=[x["image_url"] for x in [front, back] if x])
            product_only = choose_product_only(scored, cfg)

            cutout_source = product_only if product_only and product_only["meets_threshold"] else front
            cutout_path, coverage = None, None

            if cutout_source:
                try:
                    im = open_rgb(ROOT / cutout_source["local_path"])
                    rgba, coverage = segment_cutout(
                        im, p.get("category") or p.get("source_category"),
                        seg_processor, seg_model, device
                    )
                    if rgba is not None:
                        cp = cutout_dir / f"{pid}.png"
                        rgba.save(cp)
                        cutout_path = str(cp.relative_to(ROOT))
                except Exception as e:
                    append_jsonl(error_log, {"product_id": pid, "stage": "cutout", "error": repr(e)})

            reasons = []
            if not front or not front["meets_threshold"]:
                reasons.append("front_review")
            if not back or not back["meets_threshold"]:
                reasons.append("back_review")
            if not texture or not texture["meets_threshold"]:
                reasons.append("texture_review")
            if not cutout_path:
                reasons.append("cutout_failed")

            q = dict(p)
            q["image_classification_version"] = "v2.3_separate_view_scene"
            q["selected_images"] = {"front": ref(front), "back": ref(back), "texture": ref(texture)}
            q["transparent_cutout_path"] = cutout_path
            q["transparent_cutout_mask_coverage"] = coverage
            q["has_front_image"] = bool(front and front["meets_threshold"])
            q["has_back_image"] = bool(back and back["meets_threshold"])
            q["has_texture_image"] = bool(texture and texture["meets_threshold"])
            q["has_transparent_cutout"] = bool(cutout_path)
            q["manual_review_required"] = bool(reasons)
            q["manual_review_reasons"] = reasons
            q["image_classification_candidates"] = scored

            append_jsonl(output, q)

        except Exception as e:
            append_jsonl(error_log, {
                "product_id": pid, "stage": "product",
                "error": repr(e), "traceback": traceback.format_exc(limit=4)
            })

    print("\nDONE")
    print("output:", output)
    print("cutouts:", cutout_dir)
    print("errors:", error_log)

if __name__ == "__main__":
    main()
