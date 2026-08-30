"""Download, validate, and summarize the participant datasets.

This module deliberately uses only the Python standard library so every operation
runs in the same way in all platforms declared in ``pixi.toml``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PUBLIC_SET_PATH = DATA_DIR / "public_set.jsonl"
CATALOG_PATH = DATA_DIR / "catalog.jsonl"
RELEASE_DIR = DATA_DIR / "releases"
ARCHIVE_PATH = RELEASE_DIR / "catalog.jsonl.gz"

RELEASE_TAG = "participant-kit"
CATALOG_URL = (
    "https://github.com/TechJam2026/techjam-conversational-search/"
    f"releases/download/{RELEASE_TAG}/catalog.jsonl.gz"
)
CATALOG_SHA256 = "07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8"
CATALOG_ROWS = 50_000
PUBLIC_ROWS = 200

CATALOG_FIELDS = {
    "parent_asin",
    "title",
    "features",
    "description",
    "price",
    "categories",
    "details",
    "average_rating",
    "rating_number",
    "store",
}
PUBLIC_FIELDS = {
    "sample_id",
    "scenario_type",
    "category_bucket",
    "difficulty_bucket",
    "user_profile",
    "ground_truth",
}
PROFILE_FIELDS = {
    "average_prior_rating",
    "preference_tags",
    "purchase_frequency",
    "rating_style",
    "summary",
}
SCENARIOS = {"buying", "browsing", "intent_override", "boundary"}


class DataValidationError(ValueError):
    """Raised when a downloaded or checked dataset violates its contract."""


def _rows(path: Path) -> Iterator[tuple[int, dict]]:
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError as error:
        raise DataValidationError(f"missing file: {path.relative_to(ROOT)}") from error
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataValidationError(f"{path.name}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise DataValidationError(f"{path.name}:{line_number}: each row must be a JSON object")
            yield line_number, value


def _require_fields(path: Path, line_number: int, row: dict, fields: set[str]) -> None:
    missing = fields.difference(row)
    if missing:
        names = ", ".join(sorted(missing))
        raise DataValidationError(f"{path.name}:{line_number}: missing fields: {names}")


def validate_public(path: Path = PUBLIC_SET_PATH) -> Counter[str]:
    scenarios: Counter[str] = Counter()
    identifiers: set[str] = set()
    count = 0
    for line_number, row in _rows(path):
        count += 1
        _require_fields(path, line_number, row, PUBLIC_FIELDS)
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise DataValidationError(f"{path.name}:{line_number}: sample_id must be a non-empty string")
        if sample_id in identifiers:
            raise DataValidationError(f"{path.name}:{line_number}: duplicate sample_id: {sample_id}")
        identifiers.add(sample_id)
        scenario = row["scenario_type"]
        if scenario not in SCENARIOS:
            raise DataValidationError(f"{path.name}:{line_number}: unknown scenario_type: {scenario!r}")
        scenarios[scenario] += 1
        profile = row["user_profile"]
        if not isinstance(profile, dict):
            raise DataValidationError(f"{path.name}:{line_number}: user_profile must be an object")
        _require_fields(path, line_number, profile, PROFILE_FIELDS)
        ground_truth = row["ground_truth"]
        if not isinstance(ground_truth, dict) or not isinstance(ground_truth.get("parent_asin"), str):
            raise DataValidationError(
                f"{path.name}:{line_number}: ground_truth.parent_asin must be a string"
            )
    if count != PUBLIC_ROWS:
        raise DataValidationError(f"{path.name}: expected {PUBLIC_ROWS:,} rows, found {count:,}")
    return scenarios


def validate_catalog(path: Path = CATALOG_PATH) -> int:
    identifiers: set[str] = set()
    count = 0
    for line_number, row in _rows(path):
        count += 1
        _require_fields(path, line_number, row, CATALOG_FIELDS)
        parent_asin = row["parent_asin"]
        if not isinstance(parent_asin, str) or not parent_asin:
            raise DataValidationError(f"{path.name}:{line_number}: parent_asin must be a non-empty string")
        if parent_asin in identifiers:
            raise DataValidationError(f"{path.name}:{line_number}: duplicate parent_asin: {parent_asin}")
        identifiers.add(parent_asin)
    if count != CATALOG_ROWS:
        raise DataValidationError(f"{path.name}: expected {CATALOG_ROWS:,} rows, found {count:,}")
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "techjam-pixi-data-downloader/1"})
    try:
        with urllib.request.urlopen(request) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except (OSError, urllib.error.URLError) as error:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"download failed: {error}") from error


def download_catalog(force: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    if CATALOG_PATH.exists() and not force:
        count = validate_catalog()
        print(f"Catalog already exists and is valid: {CATALOG_PATH.relative_to(ROOT)} ({count:,} rows)")
        return

    archive_part = ARCHIVE_PATH.with_suffix(ARCHIVE_PATH.suffix + ".part")
    catalog_part = CATALOG_PATH.with_suffix(CATALOG_PATH.suffix + ".part")
    archive_part.unlink(missing_ok=True)
    catalog_part.unlink(missing_ok=True)
    print(f"Downloading {CATALOG_URL}")
    _download(CATALOG_URL, archive_part)
    actual_digest = _sha256(archive_part)
    if actual_digest != CATALOG_SHA256:
        archive_part.unlink(missing_ok=True)
        raise DataValidationError(
            f"catalog archive SHA-256 mismatch: expected {CATALOG_SHA256}, found {actual_digest}"
        )
    os.replace(archive_part, ARCHIVE_PATH)

    print(f"SHA-256 verified: {actual_digest}")
    try:
        with gzip.open(ARCHIVE_PATH, "rb") as source, catalog_part.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        count = validate_catalog(catalog_part)
        os.replace(catalog_part, CATALOG_PATH)
    except Exception:
        catalog_part.unlink(missing_ok=True)
        raise
    print(f"Catalog ready: {CATALOG_PATH.relative_to(ROOT)} ({count:,} rows)")


def _type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _schema(path: Path) -> tuple[int, dict[str, Counter[str]]]:
    count = 0
    fields: dict[str, Counter[str]] = {}
    for _, row in _rows(path):
        count += 1
        for name in set(fields).union(row):
            fields.setdefault(name, Counter())[_type_name(row.get(name))] += 1
    return count, fields


def describe() -> None:
    public_count, public_fields = _schema(PUBLIC_SET_PATH)
    scenarios = validate_public()
    print(f"{PUBLIC_SET_PATH.relative_to(ROOT)}: {public_count:,} rows")
    print("  fields: " + ", ".join(sorted(public_fields)))
    print("  scenarios: " + ", ".join(f"{name}={scenarios[name]}" for name in sorted(scenarios)))

    if not CATALOG_PATH.exists():
        print(f"{CATALOG_PATH.relative_to(ROOT)}: not downloaded (run `pixi run download-data`)")
        return
    catalog_count, catalog_fields = _schema(CATALOG_PATH)
    validate_catalog()
    print(f"{CATALOG_PATH.relative_to(ROOT)}: {catalog_count:,} rows")
    for name in sorted(catalog_fields):
        types = ", ".join(f"{kind}={amount:,}" for kind, amount in sorted(catalog_fields[name].items()))
        print(f"  {name}: {types}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download_parser = subparsers.add_parser("download", help="download and verify catalog.jsonl")
    download_parser.add_argument("--force", action="store_true", help="replace an existing catalog")
    subparsers.add_parser("validate-public", help="validate the committed public sessions")
    subparsers.add_parser("validate", help="validate both datasets")
    subparsers.add_parser("describe", help="show row counts, fields, scenarios, and field types")
    args = parser.parse_args()

    try:
        if args.command == "download":
            download_catalog(args.force)
        elif args.command == "validate-public":
            scenarios = validate_public()
            print(f"Public set is valid: {sum(scenarios.values()):,} rows")
        elif args.command == "validate":
            scenarios = validate_public()
            catalog_count = validate_catalog()
            print(f"Datasets are valid: {sum(scenarios.values()):,} sessions, {catalog_count:,} products")
        else:
            describe()
    except (DataValidationError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

