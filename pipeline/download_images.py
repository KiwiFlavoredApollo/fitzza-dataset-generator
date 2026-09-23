from __future__ import annotations
import asyncio, hashlib, json
from pathlib import Path
from urllib.parse import urlparse
import aiohttp
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

def load_config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

def extension_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".jpg",".jpeg",".png",".webp"} else ".jpg"

def local_image_path(cache_dir: Path, product_id: str, url: str):
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:14]
    return cache_dir / str(product_id) / f"{h}{extension_from_url(url)}"

def pick_candidates(product, cfg):
    images = product.get("images") or []
    thumb = [x for x in images if x.get("image_source_type") == "thumbnail"]
    gallery = [x for x in images if x.get("image_source_type") == "goods_gallery"]
    details = [x for x in images if x.get("image_source_type") == "goods_contents"]

    details = details[: int(cfg["max_goods_contents_per_product"])]
    candidates = thumb + gallery + details

    # Stable dedupe by URL and cap.
    seen, out = set(), []
    for x in candidates:
        u = x.get("image_url")
        if u and u not in seen:
            seen.add(u)
            out.append(x)
        if len(out) >= int(cfg["max_images_per_product"]):
            break
    return out

async def fetch(session, sem, url, path):
    if path.exists() and path.stat().st_size > 0:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    async with sem:
        try:
            async with session.get(url) as r:
                if r.status != 200:
                    return False
                data = await r.read()
                if len(data) < 1000:
                    return False
                path.write_bytes(data)
                return True
        except Exception:
            return False

async def main():
    cfg = load_config()
    dataset = ROOT / cfg["dataset_path"]
    cache = ROOT / cfg["cache_dir"]
    rows = [json.loads(x) for x in dataset.read_text(encoding="utf-8").splitlines() if x.strip()]

    jobs = []
    metadata = []
    timeout = aiohttp.ClientTimeout(total=int(cfg["download_timeout_seconds"]))
    sem = asyncio.Semaphore(int(cfg["download_concurrency"]))

    headers = {"User-Agent":"Mozilla/5.0"}
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        for p in rows:
            pid = str(p["product_id"])
            for img in pick_candidates(p, cfg):
                url = img["image_url"]
                path = local_image_path(cache, pid, url)
                jobs.append(fetch(session, sem, url, path))
                metadata.append((pid, url, str(path.relative_to(ROOT))))

        results = []
        for coro in tqdm(asyncio.as_completed(jobs), total=len(jobs), desc="download"):
            results.append(await coro)

    print(f"downloaded/available: {sum(results)}/{len(results)}")

if __name__ == "__main__":
    asyncio.run(main())
