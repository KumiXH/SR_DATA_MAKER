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


ROOT = Path(r"D:\SR数据集\网络数据集收集归类\绿植\style_diverse_mixed_sources_1200plus\telephoto_special")
SOURCES_CSV = ROOT / "sources.csv"
STATE_JSON = ROOT / "collection_state.json"
LOCK_FILE = ROOT / ".collect.lock"
TARGET = 500


@dataclass
class QueryGroup:
    source_family: str
    queries: List[str]


QUERY_GROUPS = [
    QueryGroup(
        "iNaturalist",
        [
            "flower branch",
            "tree blossom branch",
            "flower cluster branch",
            "park blossom",
            "garden flowers branch",
            "flowering tree branch",
            "distant flower",
            "shrub flowers branch",
            "branch blossom",
            "flower cluster tree",
            "tourist blossom",
            "park flowers distant",
            "flowers zoomed",
            "tree flowers far",
            "flowering shrub distant",
            "blossom from distance",
            "flowers across path",
            "flowering tree zoom",
            "park flowers zoom",
        ],
    ),
    QueryGroup(
        "Wikimedia Commons",
        [
            "flower branch photograph",
            "tree blossom branch photograph",
            "flowering tree branch photograph",
            "park flowers photograph",
            "tourist flower garden photograph",
            "cherry blossom branch photograph",
            "magnolia branch photograph",
            "telephoto flower photograph",
            "flower cluster branch photograph",
            "flowers in park photograph",
            "tree flowers distant photograph",
            "garden flowers zoom photograph",
            "tourist flowers zoom photograph",
            "flowers across street photograph",
            "blossom tree distant photograph",
            "flowering shrub distant photograph",
            "park blossom zoom photograph",
            "tree canopy flowers photograph",
        ],
    ),
]


NEGATIVE_KEYWORDS = {
    "illustration",
    "drawing",
    "painting",
    "diagram",
    "map",
    "logo",
    "poster",
    "stamp",
    "specimen",
    "herbarium",
    "microscope",
    "scan",
}


def ensure_layout() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)


def load_state() -> Dict:
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text(encoding="utf-8-sig"))
    return {"seen_urls": [], "seen_hashes": [], "records": []}


def acquire_lock() -> None:
    ensure_layout()
    try:
        fd = LOCK_FILE.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"collector already running: {LOCK_FILE}") from exc
    fd.write(str(time.time()))
    fd.close()


def release_lock() -> None:
    if LOCK_FILE.exists():
        LOCK_FILE.unlink()


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


def edge_variance(img: Image.Image) -> float:
    gray = img.convert("L").copy()
    gray.thumbnail((256, 256))
    pixels = list(gray.getdata())
    width, height = gray.size
    if width < 3 or height < 3:
        return 0.0
    diffs = []
    for y in range(height - 1):
        row = y * width
        next_row = (y + 1) * width
        for x in range(width - 1):
            p = pixels[row + x]
            dx = abs(p - pixels[row + x + 1])
            dy = abs(p - pixels[next_row + x])
            diffs.append(dx + dy)
    if not diffs:
        return 0.0
    return sum(diffs) / len(diffs)


def looks_like_telephoto_candidate(img: Image.Image) -> bool:
    width, height = img.size
    if max(width, height) < 1200:
        return False
    if green_ratio(img) < 0.03:
        return False
    if edge_variance(img) < 6.0:
        return False
    return True


def title_is_usable(text: str) -> bool:
    lower = text.lower()
    return not any(word in lower for word in NEGATIVE_KEYWORDS)


def normalize_inat_url(url: str) -> str:
    for marker in ("/square.", "/small.", "/medium.", "/large."):
        if marker in url:
            return url.replace(marker, "/original.")
    return url


def download_image(session: requests.Session, url: str) -> bytes | None:
    try:
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def commons_search(session: requests.Session, query: str, page: int) -> List[Dict]:
    offset = max(page - 1, 0) * 10
    api = (
        "https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        f"&gsrsearch={quote(query)}&gsrnamespace=6&gsrlimit=10&gsroffset={offset}"
        "&prop=imageinfo&iiprop=url&iiurlwidth=1800&format=json"
    )
    try:
        resp = session.get(api, timeout=45)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    return list(data.get("query", {}).get("pages", {}).values())


def count_files() -> int:
    return sum(len(list(ROOT.glob(pattern))) for pattern in ("*.jpg", "*.jpeg", "*.png"))


def reconcile_existing_files(state: Dict) -> None:
    recorded = {record["file"] for record in state["records"]}
    seen_hashes = set(state["seen_hashes"])
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for path in ROOT.glob(pattern):
            rel = str(path.relative_to(ROOT))
            try:
                data = path.read_bytes()
                digest = sha256_bytes(data)
                seen_hashes.add(digest)
                with Image.open(BytesIO(data)) as img:
                    width, height = img.size
            except Exception:
                width, height = 0, 0
                digest = ""
            if rel in recorded:
                continue
            state["records"].append(
                {
                    "bucket": "telephoto_special",
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
    state["seen_hashes"] = sorted(seen_hashes)


def save_candidate(img: Image.Image, source_family: str, query: str, state: Dict, meta: Dict) -> int:
    count = count_files() + 1
    out_file = ROOT / f"telephoto_special_{count:04d}.jpg"
    img.convert("RGB").save(out_file, quality=95)
    state["records"].append(
        {
            "bucket": "telephoto_special",
            "file": out_file.name,
            "source_family": source_family,
            "query": query,
            "observation_id": meta.get("observation_id", ""),
            "species_guess": meta.get("species_guess", ""),
            "taxon_name": meta.get("taxon_name", ""),
            "license_code": meta.get("license_code", ""),
            "width": img.width,
            "height": img.height,
            "source_url": meta.get("source_url", ""),
            "source_page": meta.get("source_page", ""),
        }
    )
    save_state(state)
    write_sources(state["records"])
    return count


def collect_from_inaturalist(session: requests.Session, state: Dict, max_pages: int = 35) -> int:
    seen_urls = set(state["seen_urls"])
    seen_hashes = set(state["seen_hashes"])
    count = count_files()

    for query in QUERY_GROUPS[0].queries:
        if count >= TARGET:
            break
        for page in range(1, max_pages + 1):
            if count >= TARGET:
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
                if count >= TARGET:
                    break
                if not title_is_usable(obs.get("species_guess", "")):
                    continue
                photos = obs.get("photos") or []
                taxon = obs.get("taxon") or {}
                for photo in photos[:2]:
                    if count >= TARGET:
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
                    if not blob or len(blob) < 25_000:
                        continue
                    digest = sha256_bytes(blob)
                    if digest in seen_hashes:
                        continue
                    try:
                        img = Image.open(BytesIO(blob))
                        img.load()
                    except Exception:
                        continue
                    if not looks_like_telephoto_candidate(img):
                        continue

                    count = save_candidate(
                        img,
                        "iNaturalist",
                        query,
                        state,
                        {
                            "observation_id": obs.get("id", ""),
                            "species_guess": obs.get("species_guess", ""),
                            "taxon_name": taxon.get("name", ""),
                            "license_code": photo.get("license_code") or obs.get("license_code", ""),
                            "source_url": original,
                            "source_page": obs.get("uri", ""),
                        },
                    )
                    state["seen_urls"].append(original)
                    state["seen_hashes"].append(digest)
                    seen_urls.add(original)
                    seen_hashes.add(digest)
                    time.sleep(0.08)
    return count


def collect_from_wikimedia(session: requests.Session, state: Dict, max_pages: int = 35) -> int:
    seen_urls = set(state["seen_urls"])
    seen_hashes = set(state["seen_hashes"])
    count = count_files()

    for query in QUERY_GROUPS[1].queries:
        if count >= TARGET:
            break
        for page in range(1, max_pages + 1):
            if count >= TARGET:
                break
            for item in commons_search(session, query, page):
                if count >= TARGET:
                    break
                title = item.get("title", "")
                if not title_is_usable(title):
                    continue
                infos = item.get("imageinfo") or []
                if not infos:
                    continue
                info = infos[0]
                src = info.get("url") or info.get("thumburl")
                if not src or src in seen_urls:
                    continue
                blob = download_image(session, src)
                if not blob or len(blob) < 25_000:
                    continue
                digest = sha256_bytes(blob)
                if digest in seen_hashes:
                    continue
                try:
                    img = Image.open(BytesIO(blob))
                    img.load()
                except Exception:
                    continue
                if not looks_like_telephoto_candidate(img):
                    continue

                count = save_candidate(
                    img,
                    "Wikimedia Commons",
                    query,
                    state,
                    {
                        "species_guess": title,
                        "source_url": src,
                        "source_page": info.get("descriptionurl", ""),
                    },
                )
                state["seen_urls"].append(src)
                state["seen_hashes"].append(digest)
                seen_urls.add(src)
                seen_hashes.add(digest)
                time.sleep(0.08)
    return count


def main() -> None:
    acquire_lock()
    try:
        state = load_state()
        reconcile_existing_files(state)

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 Codex telephoto-special collector"})

        count = count_files()
        if count < TARGET:
            count = collect_from_inaturalist(session, state)
        if count < TARGET:
            count = collect_from_wikimedia(session, state)
        save_state(state)
        write_sources(state["records"])
        print(f"telephoto_special={count}")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
