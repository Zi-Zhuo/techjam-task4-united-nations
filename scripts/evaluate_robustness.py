"""Small, distribution-aware robustness evaluation for the production Agent.

The official final set is private, but its customer templates, scenario policy,
catalog, and metrics are public.  This runner therefore changes only the target
products: it replaces public targets with previously unseen catalog products
that resemble the public targets on observable product covariates.  A second
suite stays in the same covariate neighbourhood while preferring products whose
metadata-derived intent cards collide with more catalog rows.

The default run evaluates 20 sessions per suite (40 total).  It is deliberately
small because loading and querying the dense model is expensive on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from evaluator.local_evaluator import (
    catalog_index,
    classify_constraint,
    coarse_category,
    evaluate,
    intent_card,
    load_jsonl,
)
from starter.agent import Agent


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_SHARES = {
    "buying": 0.40,
    "browsing": 0.40,
    "intent_override": 0.15,
    "boundary": 0.05,
}
MATCH_FIELDS = (
    "log_rating_number",
    "average_rating",
    "log_price",
    "price_missing",
    "metadata_complete",
    "metadata_richness",
    "row_percentile",
    "log_first_collision",
    "log_pair_collision",
    "log_triple_collision",
)
BASE_MATCH_FIELDS = MATCH_FIELDS[:7]
CALIBRATION_FIELDS = MATCH_FIELDS
FEATURE_WEIGHTS = {
    "log_rating_number": 4.0,
    "average_rating": 0.5,
    "log_price": 0.5,
    "price_missing": 2.0,
    "metadata_complete": 2.0,
    "metadata_richness": 0.5,
    "row_percentile": 2.0,
    "log_first_collision": 1.0,
    "log_pair_collision": 1.0,
    "log_triple_collision": 1.0,
}


@dataclass(frozen=True)
class _RawProduct:
    parent_asin: str
    row_index: int
    category: str
    broad_category: str
    constraint_type: str
    rating_number: float
    average_rating: float
    price: float | None
    metadata_richness: int
    metadata_complete: bool
    constraints: tuple[str, ...]


@dataclass(frozen=True)
class ProductProfile:
    parent_asin: str
    row_index: int
    category: str
    broad_category: str
    constraint_type: str
    rating_number: float
    log_rating_number: float
    average_rating: float
    price: float | None
    log_price: float
    price_missing: float
    metadata_complete: float
    metadata_richness: float
    row_percentile: float
    first_collision: int
    pair_collision: int
    triple_collision: int
    full_collision: int
    log_first_collision: float
    log_pair_collision: float
    log_triple_collision: float

    def value(self, field: str) -> float:
        value = getattr(self, field)
        if not isinstance(value, (int, float)):
            raise TypeError(f"{field} is not numeric")
        return float(value)


def _normalise(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip(" -;,.\t\n").lower()


def _number(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else 0.0
    return 0.0


def _price(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) and numeric >= 0.0 else None
    if isinstance(value, str):
        match = re.search(r"\d[\d,]*(?:\.\d+)?", value)
        if match:
            return float(match.group(0).replace(",", ""))
    return None


def _metadata_richness(product: dict) -> int:
    count = int(bool(product.get("title"))) + int(bool(product.get("store")))
    for field in ("features", "description"):
        value = product.get(field)
        if isinstance(value, list):
            count += sum(item not in (None, "", []) for item in value)
    details = product.get("details")
    if isinstance(details, dict):
        count += sum(item not in (None, "", []) for item in details.values())
    return count


def _broad_category(values: object) -> str:
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    if isinstance(values, list):
        for value in values:
            for part in str(value).split(","):
                normalised = _normalise(part)
                if normalised and normalised not in excluded:
                    cleaned.append(normalised)
    return cleaned[0] if cleaned else "other"


def _metadata_complete(product: dict) -> bool:
    return bool(product.get("features")) and bool(product.get("details")) and bool(product.get("store"))


def _row_band(row_index: int) -> str:
    if row_index < 1_000:
        return "head"
    if row_index < 10_000:
        return "middle"
    return "tail"


def _constraint_prefix(constraints: tuple[str, ...], length: int) -> tuple[str, ...]:
    return constraints[: min(length, len(constraints))]


def build_product_profiles(products: Sequence[dict]) -> list[ProductProfile]:
    """Build agent-independent target covariates and intent-card collision sizes."""
    raw: list[_RawProduct] = []
    category_rows: dict[str, set[int]] = defaultdict(set)
    constraint_rows: dict[str, set[int]] = defaultdict(set)
    for row_index, product in enumerate(products):
        card = intent_card(product)
        constraints = tuple(
            _normalise(value)
            for value in (
                *card.get("hard_constraints", []),
                *card.get("soft_preferences", []),
            )
            if _normalise(value)
        )
        categories = product.get("categories") or []
        category = _normalise(coarse_category(categories))
        record = _RawProduct(
            parent_asin=str(product["parent_asin"]),
            row_index=row_index,
            category=category,
            broad_category=_broad_category(categories),
            constraint_type=(
                classify_constraint(str(card.get("hard_constraints", [""])[0]))
                if card.get("hard_constraints")
                else "feature"
            ),
            rating_number=_number(product.get("rating_number")),
            average_rating=_number(product.get("average_rating")),
            price=_price(product.get("price")),
            metadata_richness=_metadata_richness(product),
            metadata_complete=_metadata_complete(product),
            constraints=constraints,
        )
        raw.append(record)
        category_rows[category].add(row_index)
        for constraint in set(constraints):
            constraint_rows[constraint].add(row_index)

    denominator = max(1, len(raw) - 1)
    collision_cache: dict[tuple[str, frozenset[str]], int] = {}

    def collision_count(record: _RawProduct, length: int) -> int:
        key = (record.category, frozenset(_constraint_prefix(record.constraints, length)))
        if key in collision_cache:
            return collision_cache[key]
        rows = set(category_rows[record.category])
        for constraint in key[1]:
            rows.intersection_update(constraint_rows.get(constraint, set()))
            if not rows:
                break
        collision_cache[key] = len(rows)
        return len(rows)

    result: list[ProductProfile] = []
    for record in raw:
        collisions = {
            length: collision_count(record, length)
            for length in (1, 2, 3, 4)
        }
        result.append(
            ProductProfile(
                parent_asin=record.parent_asin,
                row_index=record.row_index,
                category=record.category,
                broad_category=record.broad_category,
                constraint_type=record.constraint_type,
                rating_number=record.rating_number,
                log_rating_number=math.log1p(record.rating_number),
                average_rating=record.average_rating,
                price=record.price,
                log_price=math.log1p(record.price or 0.0),
                price_missing=float(record.price is None),
                metadata_complete=float(record.metadata_complete),
                metadata_richness=float(record.metadata_richness),
                row_percentile=record.row_index / denominator,
                first_collision=collisions[1],
                pair_collision=collisions[2],
                triple_collision=collisions[3],
                full_collision=collisions[4],
                log_first_collision=math.log1p(collisions[1]),
                log_pair_collision=math.log1p(collisions[2]),
                log_triple_collision=math.log1p(collisions[3]),
            )
        )
    return result


def scenario_counts(sample_count: int) -> dict[str, int]:
    """Return the exact official 40/40/15/5 mix for a small suite."""
    if sample_count < 20 or sample_count % 20:
        raise ValueError("suite-size must be a positive multiple of 20")
    counts = {
        name: int(round(sample_count * share))
        for name, share in SCENARIO_SHARES.items()
    }
    counts["buying"] += sample_count - sum(counts.values())
    return counts


def _proportional_counts(total: int, capacities: dict[str, int]) -> dict[str, int]:
    population = sum(capacities.values())
    if total > population:
        raise ValueError("requested stratum sample exceeds its population")
    raw = {name: total * count / population for name, count in capacities.items()}
    result = {name: min(capacities[name], math.floor(value)) for name, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(
        capacities,
        key=lambda name: (raw[name] - result[name], capacities[name], name),
        reverse=True,
    )
    for name in order:
        if remaining == 0:
            break
        if result[name] < capacities[name]:
            result[name] += 1
            remaining -= 1
    if remaining:
        raise ValueError("could not allocate stratified sample")
    return result


def select_anchors(
    samples: Sequence[dict],
    sample_count: int,
    seed: int,
    profiles_by_id: dict[str, ProductProfile] | None = None,
) -> list[dict]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        target_id = str(sample["ground_truth"]["parent_asin"])
        band = _row_band(profiles_by_id[target_id].row_index) if profiles_by_id else "all"
        grouped[str(sample["scenario_type"])][band].append(sample)
    rng = random.Random(seed)
    selected: list[dict] = []
    for scenario, count in scenario_counts(sample_count).items():
        bands = grouped[scenario]
        if sum(len(values) for values in bands.values()) < count:
            raise ValueError(f"not enough public {scenario} sessions")
        allocation = _proportional_counts(
            count, {band: len(values) for band, values in bands.items()}
        )
        for band, band_count in allocation.items():
            choices = sorted(bands[band], key=lambda item: str(item["sample_id"]))
            selected.extend(rng.sample(choices, band_count))
    rng.shuffle(selected)
    return selected


def _scales(records: Sequence[ProductProfile], fields: Sequence[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for field in fields:
        values = [record.value(field) for record in records]
        scale = statistics.pstdev(values) if len(values) > 1 else 0.0
        result[field] = max(scale, 1e-6)
    return result


def _distance(
    left: ProductProfile,
    right: ProductProfile,
    scales: dict[str, float],
    fields: Sequence[str],
) -> float:
    return sum(
        FEATURE_WEIGHTS[field]
        * ((left.value(field) - right.value(field)) / scales[field]) ** 2
        for field in fields
    )


def _weighted_choice(
    candidates: Sequence[tuple[float, ProductProfile]], rng: random.Random
) -> tuple[float, ProductProfile]:
    if not candidates:
        raise ValueError("cannot choose from an empty candidate pool")
    distances = [item[0] for item in candidates]
    baseline = min(distances)
    positive = [distance - baseline for distance in distances if distance > baseline]
    temperature = max(0.05, statistics.median(positive) if positive else 0.05)
    weights = [math.exp(-min(50.0, (distance - baseline) / temperature)) for distance in distances]
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for item, weight in zip(candidates, weights, strict=True):
        cumulative += weight
        if cumulative >= threshold:
            return item
    return candidates[-1]


def choose_target(
    anchor: ProductProfile,
    candidates: Sequence[ProductProfile],
    excluded_ids: set[str],
    scales: dict[str, float],
    rng: random.Random,
    mode: str,
    candidate_pool_size: int,
    stress_pool_size: int,
) -> tuple[ProductProfile, float, int, bool]:
    """Choose one unseen target and return target, distance, pool size, category match."""
    unused = [candidate for candidate in candidates if candidate.parent_asin not in excluded_ids]
    # Public targets are interaction-sized rather than catalog-uniform: 73% are
    # in the catalog's first 1,000 high-popularity rows and all have complete
    # feature/details/store metadata.  Treat those as hard strata before using
    # continuous nearest-neighbour distance.  Exact leaf categories are too
    # sparse, so the broader category is the stable categorical stratum.
    tiers = (
        lambda candidate: (
            candidate.broad_category == anchor.broad_category
            and _row_band(candidate.row_index) == _row_band(anchor.row_index)
            and candidate.constraint_type == anchor.constraint_type
            and candidate.metadata_complete == anchor.metadata_complete
            and candidate.price_missing == anchor.price_missing
        ),
        lambda candidate: (
            candidate.broad_category == anchor.broad_category
            and _row_band(candidate.row_index) == _row_band(anchor.row_index)
            and candidate.constraint_type == anchor.constraint_type
            and candidate.metadata_complete == anchor.metadata_complete
        ),
        lambda candidate: (
            candidate.broad_category == anchor.broad_category
            and _row_band(candidate.row_index) == _row_band(anchor.row_index)
            and candidate.constraint_type == anchor.constraint_type
        ),
        lambda candidate: (
            candidate.broad_category == anchor.broad_category
            and _row_band(candidate.row_index) == _row_band(anchor.row_index)
        ),
        lambda candidate: candidate.broad_category == anchor.broad_category,
        lambda candidate: True,
    )
    eligible: list[ProductProfile] = []
    if mode == "ambiguous":
        # The public set contains no full-card collision group above 50, while
        # the frozen catalog contains hundreds.  Select that blind spot
        # explicitly; continuous distance still favours the most public-like
        # members of the group.
        eligible = [
            candidate
            for candidate in unused
            if candidate.full_collision > 50 and bool(candidate.metadata_complete)
        ]
    if not eligible:
        for predicate in tiers:
            eligible = [candidate for candidate in unused if predicate(candidate)]
            if eligible:
                break
    if not eligible:
        raise ValueError("no unused catalog products remain")

    fields = MATCH_FIELDS if mode == "matched" else BASE_MATCH_FIELDS
    ordered = sorted(
        ((_distance(anchor, candidate, scales, fields), candidate) for candidate in eligible),
        key=lambda item: (item[0], item[1].row_index),
    )
    if mode == "matched":
        pool = ordered[: min(candidate_pool_size, len(ordered))]
        distance, selected = _weighted_choice(pool, rng)
        return selected, distance, len(pool), selected.category == anchor.category
    if mode != "ambiguous":
        raise ValueError(f"unknown suite mode: {mode}")

    pool = ordered[: min(stress_pool_size, len(ordered))]
    # This is deliberately a local stress selection.  Covariates in
    # BASE_MATCH_FIELDS stay close to the public anchor, while larger card
    # collision sets expose the early-list coverage and tie-breaking policy.
    ranked = sorted(
        pool,
        key=lambda item: (
            item[1].full_collision > 50,
            math.log1p(item[1].full_collision)
            + 0.5 * item[1].log_triple_collision
            + 0.25 * item[1].log_pair_collision
            - 0.05 * item[0],
            item[1].full_collision,
            rng.random(),
        ),
        reverse=True,
    )
    distance, selected = ranked[0]
    return selected, distance, len(pool), selected.category == anchor.category


def _shuffle_profiles_within_scenario(anchors: Sequence[dict], seed: int) -> list[dict]:
    profiles: dict[str, list[dict]] = defaultdict(list)
    for anchor in anchors:
        profiles[str(anchor["scenario_type"])].append(anchor["user_profile"])
    rng = random.Random(seed)
    for values in profiles.values():
        rng.shuffle(values)
    offsets: Counter[str] = Counter()
    result: list[dict] = []
    for anchor in anchors:
        scenario = str(anchor["scenario_type"])
        result.append(profiles[scenario][offsets[scenario]])
        offsets[scenario] += 1
    return result


def construct_suite(
    mode: str,
    anchors: Sequence[dict],
    profiles_by_id: dict[str, ProductProfile],
    candidates: Sequence[ProductProfile],
    excluded_ids: set[str],
    scales: dict[str, float],
    seed: int,
    candidate_pool_size: int,
    stress_pool_size: int,
) -> tuple[list[dict], list[dict], list[ProductProfile]]:
    rng = random.Random(seed)
    if mode == "collision_stress":
        user_profiles = _shuffle_profiles_within_scenario(anchors, seed + 1)
    else:
        user_profiles = [anchor["user_profile"] for anchor in anchors]

    stress_anchor_ids: set[str] = set()
    if mode == "collision_stress":
        # Forty percent of this diagnostic suite is intentionally drawn from
        # locally high-collision products.  The remaining 60% is matched in the
        # same way as the central suite, so the stress does not masquerade as a
        # private-score estimate.
        target_stress_count = max(1, round(len(anchors) * 0.40))
        grouped: dict[str, list[dict]] = defaultdict(list)
        for anchor in anchors:
            grouped[str(anchor["scenario_type"])].append(anchor)
        allocation = _proportional_counts(
            target_stress_count,
            {scenario: len(values) for scenario, values in grouped.items()},
        )
        for scenario, count in allocation.items():
            choices = sorted(grouped[scenario], key=lambda item: str(item["sample_id"]))
            stress_anchor_ids.update(
                str(item["sample_id"]) for item in rng.sample(choices, count)
            )
    elif mode != "matched":
        raise ValueError(f"unknown suite mode: {mode}")

    samples: list[dict] = []
    annotations: list[dict] = []
    selected_profiles: list[ProductProfile] = []
    for index, (anchor, user_profile) in enumerate(
        zip(anchors, user_profiles, strict=True), start=1
    ):
        anchor_id = str(anchor["ground_truth"]["parent_asin"])
        anchor_profile = profiles_by_id[anchor_id]
        target_mode = (
            "ambiguous"
            if str(anchor["sample_id"]) in stress_anchor_ids
            else "matched"
        )
        selected, distance, pool_size, category_match = choose_target(
            anchor_profile,
            candidates,
            excluded_ids,
            scales,
            rng,
            target_mode,
            candidate_pool_size,
            stress_pool_size,
        )
        excluded_ids.add(selected.parent_asin)
        selected_profiles.append(selected)
        sample_id = f"robust_{mode}_{index:04d}"
        samples.append(
            {
                "sample_id": sample_id,
                "scenario_type": anchor["scenario_type"],
                "category_bucket": anchor.get("category_bucket", "clothing"),
                "difficulty_bucket": anchor.get("difficulty_bucket", "unknown"),
                "user_profile": user_profile,
                "ground_truth": {"parent_asin": selected.parent_asin},
            }
        )
        annotations.append(
            {
                "sample_id": sample_id,
                "anchor_sample_id": str(anchor["sample_id"]),
                "anchor_parent_asin": anchor_id,
                "target_parent_asin": selected.parent_asin,
                "target_category": selected.category,
                "target_broad_category": selected.broad_category,
                "target_row_band": _row_band(selected.row_index),
                "target_constraint_type": selected.constraint_type,
                "selection_mode": target_mode,
                "exact_category_match": category_match,
                "match_distance": round(distance, 6),
                "candidate_pool_size": pool_size,
                "rating_number": selected.rating_number,
                "average_rating": selected.average_rating,
                "price": selected.price,
                "metadata_richness": selected.metadata_richness,
                "row_percentile": round(selected.row_percentile, 6),
                "first_collision": selected.first_collision,
                "pair_collision": selected.pair_collision,
                "triple_collision": selected.triple_collision,
                "full_collision": selected.full_collision,
            }
        )
    return samples, annotations, selected_profiles


def _ks_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    left_sorted = sorted(left)
    right_sorted = sorted(right)
    values = sorted(set(left_sorted + right_sorted))
    left_index = right_index = 0
    maximum = 0.0
    for value in values:
        while left_index < len(left_sorted) and left_sorted[left_index] <= value:
            left_index += 1
        while right_index < len(right_sorted) and right_sorted[right_index] <= value:
            right_index += 1
        maximum = max(
            maximum,
            abs(left_index / len(left_sorted) - right_index / len(right_sorted)),
        )
    return maximum


def calibration_report(
    anchors: Sequence[ProductProfile], selected: Sequence[ProductProfile]
) -> dict:
    fields: dict[str, dict] = {}
    for field in CALIBRATION_FIELDS:
        left = [item.value(field) for item in anchors]
        right = [item.value(field) for item in selected]
        pooled_variance = (
            (statistics.pvariance(left) if len(left) > 1 else 0.0)
            + (statistics.pvariance(right) if len(right) > 1 else 0.0)
        ) / 2.0
        denominator = math.sqrt(pooled_variance)
        mean_delta = statistics.fmean(right) - statistics.fmean(left)
        smd = mean_delta / denominator if denominator else (0.0 if mean_delta == 0.0 else math.inf)
        fields[field] = {
            "anchor_mean": round(statistics.fmean(left), 6),
            "synthetic_mean": round(statistics.fmean(right), 6),
            "standardized_mean_difference": round(smd, 6),
            "ks_distance": round(_ks_distance(left, right), 6),
        }
    finite_smds = [
        abs(item["standardized_mean_difference"])
        for item in fields.values()
        if math.isfinite(item["standardized_mean_difference"])
    ]
    return {
        "sample_count": len(selected),
        "exact_leaf_category_match_rate": round(
            sum(left.category == right.category for left, right in zip(anchors, selected, strict=True))
            / max(1, len(selected)),
            6,
        ),
        "broad_category_match_rate": round(
            sum(
                left.broad_category == right.broad_category
                for left, right in zip(anchors, selected, strict=True)
            )
            / max(1, len(selected)),
            6,
        ),
        "row_band_match_rate": round(
            sum(
                _row_band(left.row_index) == _row_band(right.row_index)
                for left, right in zip(anchors, selected, strict=True)
            )
            / max(1, len(selected)),
            6,
        ),
        "max_absolute_standardized_mean_difference": round(max(finite_smds, default=0.0), 6),
        "fields": fields,
    }


def score_summary(sessions: Sequence[dict]) -> dict:
    if not sessions:
        return {
            "sample_count": 0,
            "hit_rate_at_10": 0.0,
            "mrr": 0.0,
            "mttc": None,
            "efficiency": 0.0,
            "recommended_technical_score": 0.0,
        }
    hit_rate = statistics.fmean(float(bool(item["hit"])) for item in sessions)
    mrr = statistics.fmean(float(item["reciprocal_rank"]) for item in sessions)
    mttc = statistics.fmean(
        float(item["first_hit_turn"] if item["first_hit_turn"] is not None else 11)
        for item in sessions
    )
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    score = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
    return {
        "sample_count": len(sessions),
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(score, 6),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def stratified_bootstrap_ci(
    sessions: Sequence[dict], resamples: int, seed: int
) -> dict[str, dict[str, float]]:
    if resamples < 1 or not sessions:
        return {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(resamples):
        sample: list[dict] = []
        for rows in grouped.values():
            sample.extend(rng.choice(rows) for _ in rows)
        summary = score_summary(sample)
        for metric in (
            "hit_rate_at_10",
            "mrr",
            "mttc",
            "recommended_technical_score",
        ):
            draws[metric].append(float(summary[metric]))
    return {
        metric: {
            "lower_95": round(_quantile(values, 0.025), 6),
            "upper_95": round(_quantile(values, 0.975), 6),
        }
        for metric, values in draws.items()
    }


def _band(value: float, lower: float, upper: float) -> str:
    if value <= lower:
        return "low"
    if value >= upper:
        return "high"
    return "middle"


def subgroup_report(sessions: Sequence[dict], annotations: Sequence[dict]) -> dict:
    by_id = {str(item["sample_id"]): item for item in annotations}
    rating_values = [math.log1p(float(item["rating_number"])) for item in annotations]
    collision_values = [math.log1p(float(item["full_collision"])) for item in annotations]
    row_values = [float(item["row_percentile"]) for item in annotations]
    thresholds = {
        "popularity": (_quantile(rating_values, 0.25), _quantile(rating_values, 0.75)),
        "full_card_collision": (
            _quantile(collision_values, 0.25),
            _quantile(collision_values, 0.75),
        ),
        "catalog_row": (_quantile(row_values, 0.25), _quantile(row_values, 0.75)),
    }
    grouped: dict[str, dict[str, list[dict]]] = {
        name: defaultdict(list) for name in thresholds
    }
    for session in sessions:
        annotation = by_id[str(session["sample_id"])]
        values = {
            "popularity": math.log1p(float(annotation["rating_number"])),
            "full_card_collision": math.log1p(float(annotation["full_collision"])),
            "catalog_row": float(annotation["row_percentile"]),
        }
        for name, value in values.items():
            grouped[name][_band(value, *thresholds[name])].append(session)
    return {
        name: {band: score_summary(rows) for band, rows in sorted(bands.items())}
        for name, bands in grouped.items()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_results(paths: Iterable[str]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[str(path)] = {
            key: payload.get(key)
            for key in (
                "sample_count",
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "efficiency",
                "recommended_technical_score",
            )
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--suite-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--candidate-pool-size", type=int, default=8)
    parser.add_argument("--stress-pool-size", type=int, default=128)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--output", default="robustness_results.json")
    parser.add_argument(
        "--reference-results",
        nargs="*",
        default=["results_subset_40.json", "results_subset_40_holdout.json"],
    )
    args = parser.parse_args()
    if args.candidate_pool_size < 1 or args.stress_pool_size < 1:
        parser.error("candidate pool sizes must be positive")
    scenario_counts(args.suite_size)

    catalog_path = Path(args.catalog)
    dataset_path = Path(args.dataset)
    started = time.perf_counter()
    products = load_jsonl(catalog_path)
    public_samples = load_jsonl(dataset_path)
    profiles = build_product_profiles(products)
    profiles_by_id = {profile.parent_asin: profile for profile in profiles}
    public_target_ids = {
        str(sample["ground_truth"]["parent_asin"]) for sample in public_samples
    }
    missing = public_target_ids.difference(profiles_by_id)
    if missing:
        raise ValueError(f"public targets missing from catalog: {sorted(missing)[:3]}")

    anchors = select_anchors(
        public_samples, args.suite_size, args.seed, profiles_by_id
    )
    anchor_profiles = [
        profiles_by_id[str(anchor["ground_truth"]["parent_asin"])]
        for anchor in anchors
    ]
    all_public_profiles = [profiles_by_id[parent_asin] for parent_asin in public_target_ids]
    scales = _scales(all_public_profiles, MATCH_FIELDS)
    excluded_ids = set(public_target_ids)
    suites: dict[str, dict] = {}
    suite_samples: dict[str, list[dict]] = {}
    for offset, mode in enumerate(("matched", "collision_stress")):
        samples, annotations, selected = construct_suite(
            mode,
            anchors,
            profiles_by_id,
            profiles,
            excluded_ids,
            scales,
            args.seed + 1009 * (offset + 1),
            args.candidate_pool_size,
            args.stress_pool_size,
        )
        suite_samples[mode] = samples
        suites[mode] = {
            "construction": (
                "random draw from the nearest product-covariate neighbours"
                if mode == "matched"
                else (
                    "40% high intent-card collision targets within nearby product-covariate pools; "
                    "60% ordinary matched targets"
                )
            ),
            "profile_policy": (
                "reuse the paired public profile"
                if mode == "matched"
                else "permute profiles within scenario; marginal distribution is unchanged"
            ),
            "scenario_counts": dict(Counter(str(item["scenario_type"]) for item in samples)),
            "calibration": calibration_report(anchor_profiles, selected),
            "targets": annotations,
        }

    output = {
        "experiment": "distribution-aware pseudo-private target robustness evaluation",
        "official_contract_assumptions": {
            "catalog_rows": len(profiles),
            "public_sessions": len(public_samples),
            "private_sessions": 800,
            "scenario_mix": SCENARIO_SHARES,
            "simulator_policy": (
                "same deterministic templates and ask_attribute response policy as the released evaluator"
            ),
            "target_source": "Amazon Reviews 2023 Clothing 5-core leave-last-out",
        },
        "config": {
            "suite_size": args.suite_size,
            "total_new_sessions": args.suite_size * len(suites),
            "seed": args.seed,
            "candidate_pool_size": args.candidate_pool_size,
            "stress_pool_size": args.stress_pool_size,
            "bootstrap_resamples": args.bootstrap_resamples,
            "skip_evaluation": args.skip_evaluation,
            "match_fields": MATCH_FIELDS,
        },
        "guardrails": {
            "public_targets_excluded": True,
            "synthetic_targets_unique_across_suites": True,
            "ground_truth_or_sample_id_not_passed_to_agent": True,
            "official_evaluator_functions_reused": True,
            "paraphrase_stress_excluded_from_private_score_estimate": True,
        },
        "reference_public_runs": _reference_results(args.reference_results),
        "artifacts": {
            "catalog": str(catalog_path),
            "catalog_sha256": _sha256(catalog_path),
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "script": str(Path(__file__)),
            "script_sha256": _sha256(Path(__file__)),
        },
        "index_and_generation_seconds": round(time.perf_counter() - started, 3),
        "suites": suites,
    }

    if not args.skip_evaluation:
        print(
            f"Evaluating {args.suite_size * len(suites)} synthetic sessions "
            f"across {len(suites)} suites...",
            flush=True,
        )
        catalog_ids, categories, product_index = catalog_index(catalog_path)
        agent = Agent(catalog_path)
        pooled_sessions: list[dict] = []
        try:
            for offset, mode in enumerate(("matched", "collision_stress")):
                suite_started = time.perf_counter()
                evaluation = evaluate(
                    agent,
                    suite_samples[mode],
                    catalog_ids,
                    categories,
                    product_index,
                )
                sessions = evaluation["sessions"]
                pooled_sessions.extend(sessions)
                suites[mode]["evaluation"] = {
                    **{key: value for key, value in evaluation.items() if key != "sessions"},
                    "bootstrap_95_ci": stratified_bootstrap_ci(
                        sessions,
                        args.bootstrap_resamples,
                        args.seed + 7919 * (offset + 1),
                    ),
                    "subgroups": subgroup_report(sessions, suites[mode]["targets"]),
                    "elapsed_seconds": round(time.perf_counter() - suite_started, 3),
                    "sessions": sessions,
                }
                summary = score_summary(sessions)
                print(json.dumps({mode: summary}, indent=2), flush=True)
        finally:
            agent.connection.close()
        output["pooled_evaluation"] = {
            **score_summary(pooled_sessions),
            "bootstrap_95_ci": stratified_bootstrap_ci(
                pooled_sessions, args.bootstrap_resamples, args.seed + 65537
            ),
        }
        output["total_elapsed_seconds"] = round(time.perf_counter() - started, 3)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "suites": list(suites)}, indent=2))


if __name__ == "__main__":
    main()
