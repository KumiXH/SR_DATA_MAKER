# Style-Diverse Plant Dataset Design

## Goal

Build a `1200+` image plant dataset focused on **photo-style diversity** rather than plant-category diversity.

The dataset should cover the kinds of plant photos users actually take:

- artistic landscape-style photos
- tourist snapshots
- casual phone photos
- telephoto single-flower shots
- telephoto flower-cluster shots
- garden record photos
- botanical documentation photos
- environmental wide shots

This dataset is intended to broaden super-resolution training and evaluation coverage so the model does not overfit to a narrow "green leaf close-up" style.

## Storage Layout

Base directory:

`D:\SR数据集\网络数据集收集归类\绿植\style_diverse_mixed_sources_1200plus`

Subdirectories:

- `art_landscape`
- `tourist_snapshot`
- `phone_casual`
- `telephoto_single_flower`
- `telephoto_flower_cluster`
- `garden_record`
- `botanical_documentation`
- `environmental_wide`

Target size:

- preferred total: `1200+`
- expected per bucket: roughly `150`
- bucket balancing is flexible; quality and style separation matter more than exact equality

## Source Strategy

Use mixed public sources with traceable origin metadata:

- iNaturalist
- Wikimedia Commons
- other public image sources that provide direct image access and a recoverable source page

Prioritize style coverage over taxonomic balance.

Each downloaded image should have source metadata recorded in a shared `sources.csv` at the dataset root.

## Selection Rules

Keep images that materially expand style coverage:

- composition variety
- focal length variety
- subject scale variety
- background complexity variety
- lighting variety
- capture-device feel variety

Reject images that are:

- broken or unreadable
- extremely tiny
- obvious duplicates
- overly synthetic, diagrammatic, or non-photographic

## Execution Notes

The collection workflow should proceed bucket by bucket.

For each bucket:

1. search with style-oriented keywords
2. download candidates
3. filter broken and duplicate files
4. keep a broad visual spread within the bucket
5. record source metadata

This dataset is separate from:

- `iNaturalist_greenery_teacher_100`
- `iNaturalist_greenery_lowclarity_100`
- `synthetic_lowquality_from_teacher_100`
- `iNaturalist_diverse_user_scenes_240`

Those sets are still useful, but this new set is specifically for **style diversity across real user photo modes**.
