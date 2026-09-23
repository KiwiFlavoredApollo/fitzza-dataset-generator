from __future__ import annotations
import json, math, hashlib
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, SegformerImageProcessor, AutoModelForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]

LABEL_PROMPTS = {
    "front": [
        "a front view of a single fashion product photographed straight from the front",
        "the front side of a garment product",
    ],
    "back": [
        "a back view of a single fashion product photographed straight from the back",
        "the rear side of a garment product",
    ],
    "texture": [
        "a close-up macro photograph of fabric texture or material detail",
        "a close-up detail photograph showing fabric weave leather grain denim wash or textile surface",
    ],
    "product_only": [
        "a single isolated fashion product without a person",
        "an e-commerce product-only image of clothing shoes or hat",
    ],
    "model_worn": [
        "a person or fashion model wearing the product",
        "a model wearing clothes in a fashion lookbook",
    ],
    "other": [
        "a brand graphic size chart poster packaging or non-product informational image",
        "a non-fashion-product informational image",
    ],
}

# mattmdjaga/segformer_b2_clothes label names commonly exposed by the model.
CATEGORY_TARGET_LABELS = {
    "top": {"upper-clothes", "shirt", "dress"},
    "outer": {"upper-clothes", "coat", "dress"},
    "bottom": {"pants", "skirt"},
    "shoes": {"left-shoe", "right-shoe"},
    "hat": {"hat"},
}

def load_config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

def device_from_cfg(cfg):
    if cfg.get("device") != "auto":
        return cfg["device"]
    return "cuda" if torch.cuda.is_available() else "cpu"

def local_image_path(cache_dir: Path, product_id: str, url: str):
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in {".jpg",".jpeg",".png",".webp"}:
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
        if x.get("image_url") and x["image_url"] not in seen:
            seen.add(x["image_url"])
            out.append(x)
        if len(out) >= int(cfg["max_images_per_product"]):
            break
    return out

def open_rgb(path):
    im = Image.open(path)
    return ImageOps.exif_transpose(im).convert("RGB")


def _unwrap_clip_feature(output):
    """transformers version compatibility for CLIP feature outputs."""
    if torch.is_tensor(output):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 2:
                return item
        for item in output:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Unsupported CLIP feature output type: {type(output)}")

@torch.inference_mode()
def classify_images(images, model, processor, device):
    labels = list(LABEL_PROMPTS.keys())
    # Average the embeddings of two prompts per semantic class.
    prompt_texts = []
    prompt_owner = []
    for label in labels:
        for p in LABEL_PROMPTS[label]:
            prompt_texts.append(p)
            prompt_owner.append(label)

    text_inputs = processor(text=prompt_texts, return_tensors="pt", padding=True).to(device)
    text_feat = _unwrap_clip_feature(model.get_text_features(**text_inputs))
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    # aggregate prompt vectors by class
    class_vecs = []
    for label in labels:
        idx = [i for i,x in enumerate(prompt_owner) if x == label]
        v = text_feat[idx].mean(dim=0)
        v = v / v.norm()
        class_vecs.append(v)
    class_vecs = torch.stack(class_vecs)

    inputs = processor(images=images, return_tensors="pt").to(device)
    img_feat = _unwrap_clip_feature(model.get_image_features(**inputs))
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    logits = img_feat @ class_vecs.T
    probs = torch.softmax(logits * 12.0, dim=-1).cpu().numpy()

    return [{labels[j]: float(p[j]) for j in range(len(labels))} for p in probs]

def choose_best(scored, label, threshold, prefer_sources=None, excluded_urls=None):
    excluded_urls = set(excluded_urls or [])
    prefer_sources = prefer_sources or []
    pool = [x for x in scored if x["image_url"] not in excluded_urls]
    if not pool:
        return None

    def score(x):
        s = x["scores"].get(label, 0.0)
        if x["source_type"] in prefer_sources:
            s += 0.04
        if label in {"front","back"}:
            s += 0.10 * x["scores"].get("product_only", 0.0)
            s -= 0.06 * x["scores"].get("other", 0.0)
        if label == "texture":
            s += 0.04 if x["source_type"] == "goods_contents" else 0.0
        return s

    best = max(pool, key=score)
    best = dict(best)
    best["selection_score"] = float(score(best))
    best["meets_threshold"] = best["scores"].get(label, 0.0) >= threshold
    return best

def id2label_map(model):
    raw = model.config.id2label
    return {int(k): str(v).lower() for k,v in raw.items()}

@torch.inference_mode()
def segment_cutout(image, category, processor, model, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model(**inputs)
    logits = torch.nn.functional.interpolate(
        out.logits,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False
    )
    pred = logits.argmax(dim=1)[0].cpu().numpy()

    id2label = id2label_map(model)
    wanted = CATEGORY_TARGET_LABELS.get(category, set())
    wanted_ids = [idx for idx,label in id2label.items() if label in wanted]

    if not wanted_ids:
        return None, 0.0

    mask = np.isin(pred, wanted_ids).astype(np.uint8) * 255
    coverage = float((mask > 0).mean())
    if coverage < 0.015:
        return None, coverage

    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask, mode="L"))
    return rgba, coverage

def selected_ref(x):
    if not x:
        return None
    return {
        "image_url": x["image_url"],
        "local_path": x["local_path"],
        "source_type": x["source_type"],
        "scores": x["scores"],
        "selection_score": x["selection_score"],
        "meets_threshold": x["meets_threshold"],
    }

def main():
    cfg = load_config()
    device = device_from_cfg(cfg)
    print("device:", device)

    dataset = ROOT / cfg["dataset_path"]
    cache = ROOT / cfg["cache_dir"]
    output = ROOT / cfg["output_dataset"]
    cutout_dir = ROOT / cfg["cutout_dir"]
    cutout_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(x) for x in dataset.read_text(encoding="utf-8").splitlines() if x.strip()]

    clip_model = CLIPModel.from_pretrained(cfg["classification_model"]).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(cfg["classification_model"])

    seg_processor = SegformerImageProcessor.from_pretrained(cfg["segmentation_model"])
    seg_model = AutoModelForSemanticSegmentation.from_pretrained(cfg["segmentation_model"]).to(device).eval()

    completed = []

    for p in tqdm(rows, desc="products"):
        pid = str(p["product_id"])
        available = []

        for img in candidate_images(p, cfg):
            lp = local_image_path(cache, pid, img["image_url"])
            if lp.exists():
                try:
                    im = open_rgb(lp)
                    if im.width >= 160 and im.height >= 160:
                        available.append((img, lp, im))
                except Exception:
                    pass

        scored = []
        if available:
            probs = classify_images([x[2] for x in available], clip_model, clip_processor, device)
            for (meta, lp, _), sc in zip(available, probs):
                scored.append({
                    "image_url": meta["image_url"],
                    "local_path": str(lp.relative_to(ROOT)),
                    "source_type": meta.get("image_source_type"),
                    "scores": sc,
                })

        front = choose_best(
            scored, "front", float(cfg["min_front_score"]),
            prefer_sources=["thumbnail","goods_gallery"]
        )
        back = choose_best(
            scored, "back", float(cfg["min_back_score"]),
            prefer_sources=["goods_gallery"],
            excluded_urls=[front["image_url"]] if front else []
        )
        texture = choose_best(
            scored, "texture", float(cfg["min_texture_score"]),
            prefer_sources=["goods_contents","goods_gallery"],
            excluded_urls=[x["image_url"] for x in [front,back] if x]
        )

        product_only = choose_best(
            scored, "product_only", float(cfg["min_product_only_score"]),
            prefer_sources=["thumbnail","goods_gallery"]
        )

        # Best cutout source:
        # Prefer a product-only front image, otherwise selected front.
        cutout_source = product_only if product_only and product_only["meets_threshold"] else front
        cutout_path = None
        cutout_coverage = None

        if cutout_source:
            try:
                im = open_rgb(ROOT / cutout_source["local_path"])
                rgba, coverage = segment_cutout(
                    im, p.get("category") or p.get("source_category"),
                    seg_processor, seg_model, device
                )
                cutout_coverage = coverage
                if rgba is not None:
                    cp = cutout_dir / f"{pid}.png"
                    rgba.save(cp)
                    cutout_path = str(cp.relative_to(ROOT))
            except Exception:
                pass

        manual_reasons = []
        if not front or not front["meets_threshold"]:
            manual_reasons.append("front_low_confidence")
        if not back or not back["meets_threshold"]:
            manual_reasons.append("back_low_confidence")
        if not texture or not texture["meets_threshold"]:
            manual_reasons.append("texture_low_confidence")
        if not cutout_path:
            manual_reasons.append("cutout_failed")

        q = dict(p)
        q["image_classification_status"] = "classified"
        q["selected_images"] = {
            "front": selected_ref(front),
            "back": selected_ref(back),
            "texture": selected_ref(texture),
        }
        q["transparent_cutout_path"] = cutout_path
        q["transparent_cutout_mask_coverage"] = cutout_coverage

        q["has_front_image"] = bool(front and front["meets_threshold"])
        q["has_back_image"] = bool(back and back["meets_threshold"])
        q["has_texture_image"] = bool(texture and texture["meets_threshold"])
        q["has_transparent_cutout"] = bool(cutout_path)

        q["manual_review_required"] = bool(manual_reasons)
        q["manual_review_reasons"] = manual_reasons
        q["image_ai_model"] = cfg["classification_model"]
        q["segmentation_model"] = cfg["segmentation_model"]
        q["image_classification_candidates"] = scored

        completed.append(q)

    with output.open("w", encoding="utf-8") as f:
        for q in completed:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(output)

if __name__ == "__main__":
    main()
