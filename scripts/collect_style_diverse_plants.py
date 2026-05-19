import csv
import hashlib
import json
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote

import requests
from PIL import Image


ROOT = Path(r"D:\SR数据集\网络数据集收集归类\绿植\style_diverse_mixed_sources_1200plus")
SOURCES_CSV = ROOT / "sources.csv"
STATE_JSON = ROOT / "collection_state.json"


@dataclass
class Bucket:
    name: str
    target: int
    queries: List[str]


BUCKETS = [
    Bucket(
        "art_landscape",
        150,
        [
            "flower landscape",
            "garden landscape flowers",
            "wildflower landscape",
            "flowering tree landscape",
            "botanical garden landscape",
        ],
    ),
    Bucket(
        "tourist_snapshot",
        150,
        [
            "tourist flower garden",
            "tourist botanical garden",
            "park flowers",
            "scenic garden flowers",
            "travel flower photo",
        ],
    ),
    Bucket(
        "phone_casual",
        150,
        [
            "houseplant",
            "potted plant",
            "balcony flowers",
            "yard flowers",
            "casual plant photo",
        ],
    ),
    Bucket(
        "telephoto_single_flower",
        150,
        [
            "rose flower",
            "orchid flower",
            "lily flower",
            "hibiscus flower",
            "single blossom close",
        ],
    ),
    Bucket(
        "telephoto_flower_cluster",
        150,
        [
            "hydrangea",
            "lavender field",
            "flower cluster close",
            "azalea flowers",
            "blossom cluster",
        ],
    ),
    Bucket(
        "garden_record",
        150,
        [
            "garden plant",
            "flower bed",
            "backyard flowers",
            "garden shrub",
            "yard plant record",
            "flower garden photograph",
            "home garden plants",
            "potted plant photograph",
            "community garden flowers",
            "botanical garden path plants",
        ],
    ),
    Bucket(
        "botanical_documentation",
        150,
        [
            "leaf closeup",
            "plant documentation",
            "branch leaves",
            "whole plant",
            "botanical specimen photo",
            "botanical documentation plant photograph",
            "plant leaf photograph",
            "stem leaf closeup photograph",
            "plant habit photograph",
            "plant detail photograph",
        ],
    ),
    Bucket(
        "environmental_wide",
        150,
        [
            "tree canopy",
            "forest flowers",
            "roadside flowers",
            "field flowers",
            "plants in habitat",
            "tree canopy photograph",
            "plants in habitat photograph",
            "botanical garden landscape wide",
            "wildflower meadow photograph",
            "park trees flowers wide",
        ],
    ),
]


def ensure_layout() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for bucket in BUCKETS:
        (ROOT / bucket.name).mkdir(parents=True, exist_ok=True)


def load_state() -> Dict:
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text(encoding="utf-8"))
    return {"seen_urls": [], "seen_hashes": [], "records": []}


def save_state(state: Dict) -> None:
    STATE_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def write_sources(records: List[Dict]) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with SOURCES_CSV.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def normalize_inat_url(url: str) -> str:
    for marker in ("/square.", "/small.", "/medium.", "/large."):
        if marker in url:
            return url.replace(marker, "/original.")
    return url


def commons_search(session: requests.Session, query: str, page: int) -> List[Dict]:
    offset = max(page - 1, 0) * 10
    api = (
        "https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        f"&gsrsearch={quote(query)}&gsrnamespace=6&gsrlimit=10&gsroffset={offset}"
        "&prop=imageinfo&iiprop=url&iiurlwidth=1600&format=json"
    )
    try:
        resp = session.get(api, timeout=45)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    pages = data.get("query", {}).get("pages", {})
    return list(pages.values())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def green_ratio(img: Image.Image) -> float:
    sample = img.convert("RGB").copy()
    sample.thumbnail((384, 384))
    pixels = list(sample.getdata())
    total = len(pixels)
    green = 0
    for r, g, b in pixels:
        if g > r * 1.03 and g > b * 1.03 and max(r, g, b) > 30:
            green += 1
    return green / total if total else 0.0


def min_green_ratio(bucket_name: str) -> float:
    if bucket_name == "environmental_wide":
        return 0.03
    return 0.05


def download_image(session: requests.Session, url: str) -> bytes | None:
    try:
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def collect_from_inaturalist(
    session: requests.Session,
    bucket: Bucket,
    state: Dict,
    per_bucket_existing: int,
    max_pages: int = 10,
) -> int:
    out_dir = ROOT / bucket.name
    count = per_bucket_existing
    seen_urls = set(state["seen_urls"])
    seen_hashes = set(state["seen_hashes"])

    for query in bucket.queries:
        if count >= bucket.target:
            break
        for page in range(1, max_pages + 1):
            if count >= bucket.target:
                break
            api = (
                "https://api.inaturalist.org/v1/observations"
                f"?iconic_taxa=Plantae&quality_grade=research&photos=true"
                f"&photo_license=CC0,CC-BY,CC-BY-NC,CC-BY-SA"
                f"&per_page=50&page={page}&q={quote(query)}"
            )
            try:
                resp = session.get(api, timeout=45)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                continue

            for obs in data.get("results", []):
                if count >= bucket.target:
                    break
                photos = obs.get("photos") or []
                taxon = obs.get("taxon") or {}
                for photo in photos[:2]:
                    if count >= bucket.target:
                        break
                    dims = photo.get("original_dimensions") or {}
                    width = dims.get("width") or 0
                    height = dims.get("height") or 0
                    if max(width, height) < 1200:
                        continue
                    src = photo.get("url")
                    if not src:
                        continue
                    original = normalize_inat_url(src)
                    if original in seen_urls:
                        continue
                    blob = download_image(session, original)
                    if not blob or len(blob) < 20_000:
                        continue
                    digest = sha256_bytes(blob)
                    if digest in seen_hashes:
                        continue
                    try:
                        img = Image.open(BytesIO(blob))
                        img.load()
                    except Exception:
                        continue
                    if green_ratio(img) < min_green_ratio(bucket.name):
                        continue

                    count += 1
                    out_file = out_dir / f"{bucket.name}_{count:04d}.jpg"
                    img.convert("RGB").save(out_file, quality=95)

                    record = {
                        "bucket": bucket.name,
                        "file": str(out_file.relative_to(ROOT)),
                        "source_family": "iNaturalist",
                        "query": query,
                        "observation_id": obs.get("id", ""),
                        "species_guess": obs.get("species_guess", ""),
                        "taxon_name": taxon.get("name", ""),
                        "license_code": photo.get("license_code") or obs.get("license_code", ""),
                        "width": img.width,
                        "height": img.height,
                        "source_url": original,
                        "source_page": obs.get("uri", ""),
                    }
                    state["records"].append(record)
                    state["seen_urls"].append(original)
                    state["seen_hashes"].append(digest)
                    seen_urls.add(original)
                    seen_hashes.add(digest)
                    time.sleep(0.08)
    return count


def collect_from_wikimedia(
    session: requests.Session,
    bucket: Bucket,
    state: Dict,
    per_bucket_existing: int,
    max_pages: int = 8,
) -> int:
    out_dir = ROOT / bucket.name
    count = per_bucket_existing
    seen_urls = set(state["seen_urls"])
    seen_hashes = set(state["seen_hashes"])

    for query in bucket.queries:
        if count >= bucket.target:
            break
        for page in range(1, max_pages + 1):
            if count >= bucket.target:
                break
            for item in commons_search(session, query, page):
                if count >= bucket.target:
                    break
                infos = item.get("imageinfo") or []
                if not infos:
                    continue
                info = infos[0]
                src = info.get("url") or info.get("thumburl")
                if not src or src in seen_urls:
                    continue
                blob = download_image(session, src)
                if not blob or len(blob) < 20_000:
                    continue
                digest = sha256_bytes(blob)
                if digest in seen_hashes:
                    continue
                try:
                    img = Image.open(BytesIO(blob))
                    img.load()
                except Exception:
                    continue
                if max(img.width, img.height) < 1200:
                    continue
                if green_ratio(img) < min_green_ratio(bucket.name):
                    continue

                count += 1
                out_file = out_dir / f"{bucket.name}_{count:04d}.jpg"
                img.convert("RGB").save(out_file, quality=95)

                record = {
                    "bucket": bucket.name,
                    "file": str(out_file.relative_to(ROOT)),
                    "source_family": "Wikimedia Commons",
                    "query": query,
                    "observation_id": "",
                    "species_guess": item.get("title", ""),
                    "taxon_name": "",
                    "license_code": "",
                    "width": img.width,
                    "height": img.height,
                    "source_url": src,
                    "source_page": info.get("descriptionurl", ""),
                }
                state["records"].append(record)
                state["seen_urls"].append(src)
                state["seen_hashes"].append(digest)
                seen_urls.add(src)
                seen_hashes.add(digest)
                time.sleep(0.08)
    return count


def bucket_file_count(bucket_name: str) -> int:
    bucket_dir = ROOT / bucket_name
    return sum(len(list(bucket_dir.glob(pattern))) for pattern in ("*.jpg", "*.jpeg", "*.png"))


def reconcile_existing_files(state: Dict) -> None:
    recorded = {record["file"] for record in state["records"]}
    for bucket in BUCKETS:
        bucket_dir = ROOT / bucket.name
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            for path in bucket_dir.glob(pattern):
                rel = str(path.relative_to(ROOT))
                if rel in recorded:
                    continue
                try:
                    with Image.open(path) as img:
                        width, height = img.size
                except Exception:
                    width, height = 0, 0
                state["records"].append(
                    {
                        "bucket": bucket.name,
                        "file": rel,
                        "source_family": "existing_untracked",
                        "query": "",
                        "observation_id": "",
                        "species_guess": "",
                        "taxon_name": "",
                        "license_code": "",
                        "width": width,
                        "height": height,
                        "source_url": "",
                        "source_page": "",
                    }
                )
                recorded.add(rel)


def main() -> None:
    ensure_layout()
    state = load_state()
    reconcile_existing_files(state)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 Codex style-diverse plant collector"})

    buckets_by_need = sorted(
        BUCKETS,
        key=lambda bucket: (bucket.target - bucket_file_count(bucket.name), bucket.name == "environmental_wide"),
        reverse=True,
    )

    for bucket in buckets_by_need:
        existing = bucket_file_count(bucket.name)
        if existing >= bucket.target:
            print(f"{bucket.name}={existing}")
            continue

        if bucket.name == "environmental_wide":
            updated = collect_from_inaturalist(session, bucket, state, existing, max_pages=20)
            if updated < bucket.target:
                updated = collect_from_wikimedia(session, bucket, state, updated, max_pages=20)
        elif bucket.name in {"garden_record", "botanical_documentation"}:
            updated = collect_from_wikimedia(session, bucket, state, existing, max_pages=20)
            if updated < bucket.target:
                updated = collect_from_inaturalist(session, bucket, state, updated, max_pages=12)
        else:
            updated = collect_from_inaturalist(session, bucket, state, existing)
            if updated < bucket.target:
                updated = collect_from_wikimedia(session, bucket, state, updated)

        print(f"{bucket.name}={updated}")
        save_state(state)
        write_sources(state["records"])


if __name__ == "__main__":
    main()
