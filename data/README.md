# Competition Data

Both files use JSON Lines (JSONL): one independent JSON object per UTF-8 line. This makes it possible
to stream the 50,000-product catalog without loading the complete file into memory.

## `public_set.jsonl`

Contains 200 labeled development sessions: 80 Buying, 80 Browsing, 30 Intent Override, and 10 Boundary sessions.

Each row has this shape:

| Field | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | Unique public sample identifier |
| `scenario_type` | string | `buying`, `browsing`, `intent_override`, or `boundary` |
| `category_bucket` | string | Coarse target category used for stratification |
| `difficulty_bucket` | string | Public difficulty grouping |
| `user_profile` | object | Safe aggregate profile described below |
| `ground_truth.parent_asin` | string | Target product ID used only for local scoring |

`user_profile` contains `average_prior_rating` (number), `preference_tags` (string array),
`purchase_frequency` (string), `rating_style` (string), and a natural-language `summary` (string).

Each session contains a safe aggregate `user_profile` and public labels for local development. Direct user identifiers, timestamps, free-text reviews, raw purchase history, hidden intent cards, and simulator-policy internals are not shipped in this participant file.

## `catalog.jsonl`

The downloaded catalog contains exactly 50,000 products. Each row contains:

| Field | Typical type | Meaning |
| --- | --- | --- |
| `parent_asin` | string | Stable product-family identifier and evaluation join key |
| `title` | string | Product title |
| `features` | array | Feature bullet text |
| `description` | array | Product description text |
| `price` | number, string, or null | Listed price when available; some source rows use `from …` or `—` strings |
| `categories` | array | Category hierarchy/tags |
| `details` | object | Structured product attributes |
| `average_rating` | number | Aggregate rating |
| `rating_number` | number | Number of ratings |
| `store` | string or null | Store or brand storefront name |

From the repository root, download and verify it with:

```bash
pixi run download-data
pixi run data-info
```

The first command downloads the pinned
[`participant-kit` release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit),
checks SHA-256 `07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8`,
then installs the decompressed file at `data/catalog.jsonl`. The compressed archive is retained under
`data/releases/`; both artifacts are gitignored. The second command reports the actual field types and counts.

If automation is unavailable, download
[`catalog.jsonl.gz`](https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz)
and [`SHA256SUMS`](https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/SHA256SUMS),
verify the archive, and decompress it to `data/catalog.jsonl`.

Never place API keys, private evaluation data, or participant outputs in this directory.
