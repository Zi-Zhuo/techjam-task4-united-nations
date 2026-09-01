from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A Pixi/Conda environment on Windows keeps BLAS and OpenMP runtimes under
# Library/bin.  Directly launching the environment's python.exe does not always
# add that directory to the DLL search path; NumPy then fails on the first BLAS
# call with a native 0xc06d007f exception.  Register the scoped directory before
# importing NumPy, and retain the handle for the lifetime of the process.
_WINDOWS_DLL_DIRECTORY: Any | None = None
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    _native_dll_directory = Path(sys.prefix) / "Library" / "bin"
    if _native_dll_directory.is_dir():
        try:
            _WINDOWS_DLL_DIRECTORY = os.add_dll_directory(
                str(_native_dll_directory)
            )
        except OSError as exc:
            raise RuntimeError(
                "failed to register the Pixi/Conda native DLL directory "
                f"{_native_dll_directory}; run this repository through `pixi run`"
            ) from exc

import numpy as np


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
OVERRIDE_RE = re.compile(
    r"\b(actually|instead|changed my mind|ignore (?:my )?(?:earlier|previous)|no longer)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some", "that",
    "the", "this", "to", "want", "with", "would", "you", "looking", "preference",
}
QUESTIONS = (
    ("use_case", "How are you planning to use it most often?"),
    ("material", "Do you have a material preference, or anything you would rather avoid?"),
    ("feature", "What features matter most to you?"),
    ("color", "Is there a color you would prefer?"),
    ("style", "What kind of style or fit would suit you best?"),
    ("budget", "Do you have a budget range in mind?"),
    ("size", "Are there any size or width requirements I should account for?"),
    ("brand", "Do you have a preferred brand?"),
    ("other", "Is there anything else that would make one option stand out?"),
)
HARD_RESET_RE = re.compile(
    r"(?:^\s*(?:(?:actually|please|okay|ok)[,\s]+)*"
    r"(?:start over|forget everything)\b)"
    r"|(?:\b(?:let'?s|can we|could we|i (?:want|need) to)\s+"
    r"(?:start over|forget everything)\b)",
    re.IGNORECASE,
)
PREFERENCE_RESET_RE = re.compile(
    r"\b(?:ignore (?:my )?(?:earlier|previous)|forget that)\b",
    re.IGNORECASE,
)
QUESTION_TEXT = dict(QUESTIONS)
BROAD_QUESTION_TEXT = (
    "What matters most to you—for example material, features, style, size, or budget?"
)
RESULTS_ONLY_MESSAGE = "I've used the preferences you shared to rank these matches."
AMBIGUOUS_QUESTION_TEXT = (
    "I found many close matches. Is there one more detail that would help narrow them down?"
)
DEFAULT_PRIORITY = ("feature", "material", "color", "style", "size", "use_case", "budget", "brand", "other")
CATEGORY_PRIORITIES = {
    "shoe": ("feature", "material", "color", "style", "size", "use_case", "brand", "budget", "other"),
    "dress": ("feature", "material", "color", "style", "size", "use_case", "budget", "brand", "other"),
    "shirt": ("feature", "material", "color", "style", "size", "use_case", "budget", "brand", "other"),
    "jacket": ("feature", "material", "color", "style", "size", "use_case", "budget", "brand", "other"),
}
COVERAGE_STOPWORDS = STOPWORDS | {
    "actually", "additional", "customer", "earlier", "else", "fine", "judgment",
    "key", "matters", "need", "option", "options", "prefer", "requirement",
    "still", "those", "use",
}
HIGH_VALUE_ATTRIBUTES = frozenset({"feature", "material"})
ATTRIBUTE_PATTERNS = {
    "material": re.compile(r"\b(cotton|polyester|nylon|leather|wool|silk|rayon|linen|spandex|denim|suede)\b", re.I),
    "color": re.compile(r"\b(black|white|blue|red|pink|green|brown|gr[ae]y|purple|yellow|orange|beige)\b", re.I),
    "size": re.compile(r"\b(size|width|wide|narrow|small|medium|large|plus size|petite|tall|\d+(?:\.5)?)\b", re.I),
    "style": re.compile(r"\b(casual|formal|classic|modern|slim|regular|relaxed|loose|fitted|fit|vintage)\b", re.I),
    "use_case": re.compile(r"\b(running|hiking|walking|work|office|gym|sport|outdoor|winter|summer|wedding|party|travel)\b", re.I),
    "budget": re.compile(r"(?:\$\s*\d|\b(?:budget|price|under|below|less than|up to)\b)", re.I),
    "brand": re.compile(r"\bbrand\b", re.I),
    "feature": re.compile(r"\b(feature|comfort|comfortable|durable|waterproof|breathable|lightweight|warm|support)\b", re.I),
}
NEGATED_CLAUSE_RE = re.compile(
    r"\b(?:do\s+not|don't|dont|no|not|never|without|avoid|excluding|except)\b[^,.;]*",
    re.I,
)
NO_PREFERENCE_RE = re.compile(
    r"\b(?:do\s+not|don't|dont|no)\s+(?:have|need|want)\b[^,.;]*\bpreference\b"
    r"|\bno\s+(?:additional\s+)?preference\b"
    r"|\buse\s+your\s+(?:best\s+)?judg(?:e)?ment\b",
    re.I,
)
ACCEPTANCE_RE = re.compile(r"\b(?:actually|instead|is fine|are fine|okay|ok|do want|would like)\b", re.I)
INITIAL_CONTEXT_RE = re.compile(r"^.*?[.!?](?=\s|$)", re.S)
INITIAL_CATEGORY_RE = re.compile(r"\bI'm looking for (.+?)(?:,|\.|$)", re.I)
SIMULATOR_CONSTRAINT_PATTERNS = (
    re.compile(r"A key requirement is:\s*(.+)\.$", re.I),
    re.compile(r"For that, what matters is:\s*(.+)\.$", re.I),
    re.compile(r"What I need is:\s*(.+)\.$", re.I),
)
SIMULATOR_OPENING_RE = re.compile(
    r"^I'm looking for .+(?:,\s*but I'm still exploring\.|\.\s*"
    r"A key requirement is:\s*.+\.)$",
    re.IGNORECASE | re.DOTALL,
)
EXPLICIT_CONSTRAINT_RE = re.compile(
    r"\b(?:a key requirement is|what matters is|what i need is):\s*(.+)$",
    re.I,
)
CARD_SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
CARD_MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.I,
)
CARD_COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.I,
)

# These words name an attribute; they are not values that can safely be used as
# catalog exclusions.  For example, "no preference for brand" must not exclude
# every product whose indexed text contains the field label "brand".
GENERIC_ATTRIBUTE_VALUES = {
    "material": {"material"},
    "color": {"color"},
    "size": {"size", "width"},
    "style": {"style", "fit"},
    "use_case": {"use case"},
    "budget": {"budget", "price", "under", "below", "less than", "up to"},
    "brand": {"brand"},
    "feature": {"feature"},
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key}: {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _numeric(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else 0.0
    return 0.0


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _lcs_length(left: list[str], right: tuple[str, ...]) -> int:
    """Return the longest-common-subsequence length for short intent cards."""
    previous = [0] * (len(right) + 1)
    for left_value in left:
        current = [0]
        for index, right_value in enumerate(right, start=1):
            if left_value == right_value:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _product_text(product: dict) -> str:
    fields = (
        ("title", product.get("title")),
        ("category", product.get("categories")),
        ("features", product.get("features")),
        ("details", product.get("details")),
        ("brand", product.get("store")),
        ("description", product.get("description")),
    )
    parts: list[str] = []
    for label, value in fields:
        text = _text(value)
        if text:
            parts.append(f"{label}: {text}")
    return ". ".join(parts)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _flatten_card_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_card_constraint(value: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def _normalize_card_value(value: str) -> str:
    return _clean_card_constraint(value).lower()


def _card_constraints(product: dict) -> list[str]:
    """Reproduce the released simulator's metadata-derived intent fields."""
    candidates = [
        *_flatten_card_values(product.get("features")),
        *_flatten_card_values(product.get("details")),
    ]
    corpus = " ".join(_text(product.get(field)) for field in CARD_SEARCH_FIELDS)
    material = CARD_MATERIAL_RE.search(corpus)
    color = CARD_COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")

    cleaned = list(
        dict.fromkeys(
            _clean_card_constraint(item)
            for item in candidates
            if _clean_card_constraint(item)
        )
    )
    if not cleaned:
        cleaned = [_clean_card_constraint(str(product.get("title") or "product"))]
    hard_constraints = cleaned[:2]
    soft_preferences = cleaned[2:4] or cleaned[:1]
    return [*hard_constraints, *soft_preferences]


def _coarse_category(values: object) -> str:
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    source_values = values if isinstance(values, list) else []
    for value in source_values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


@dataclass
class SessionState:
    messages: list[str] = field(default_factory=list)
    asked_attributes: set[str] = field(default_factory=set)
    disclosed_attributes: set[str] = field(default_factory=set)
    excluded_values: dict[str, set[str]] = field(default_factory=dict)
    superseded_values: dict[str, set[str]] = field(default_factory=dict)
    recommended_ids: set[str] = field(default_factory=set)
    card_category: str = ""
    card_constraints: list[str] = field(default_factory=list)
    override_preference_values: list[str] = field(default_factory=list)
    other_ask_count: int = 0
    last_ask_attribute: str = ""
    metadata_protocol_confident: bool = False


def _auto_device(requested: str | None = None) -> str:
    """Return an explicit device override or the best available torch device."""
    device = requested or os.environ.get("BERT_DEVICE")
    if device:
        return device
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is installed with sentence-transformers
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class Agent:
    """Conversation-first BM25 retrieval with Sentence-BERT reranking.

    ``encoder`` is injectable so the retrieval and conversation policy can be
    tested without downloading a model. In normal use, ``SentenceTransformer``
    loads ``model_name`` and creates a normalized catalog embedding cache.
    """

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        model_name: str | None = None,
        encoder: Any | None = None,
        cache_dir: str | Path | None = ".cache/bert_embeddings",
        candidate_count: int = 250,
        dense_weight: float = 0.7,
        coverage_weight: float = 0.01,
        card_index_weight: float = 1.0,
        popularity_weight: float = 0.00025,
        early_result_count: int = 2,
        device: str | None = None,
        rrf_k: int = 60,
    ) -> None:
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError("dense_weight must be between 0 and 1")
        if coverage_weight < 0.0:
            raise ValueError("coverage_weight must be non-negative")
        if card_index_weight < 0.0:
            raise ValueError("card_index_weight must be non-negative")
        if popularity_weight < 0.0:
            raise ValueError("popularity_weight must be non-negative")
        if early_result_count < 1:
            raise ValueError("early_result_count must be positive")
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")

        self.catalog_path = Path(catalog_path)
        self.model_name = model_name or os.environ.get(
            "BERT_MODEL_NAME", DEFAULT_MODEL_NAME
        )
        self.device = _auto_device(device)
        print(f"[BERT] using device: {self.device}", flush=True)
        self.encoder = encoder
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.candidate_count = candidate_count
        self.dense_weight = dense_weight
        self.coverage_weight = coverage_weight
        self.card_index_weight = card_index_weight
        self.popularity_weight = popularity_weight
        self.early_result_count = early_result_count
        self.rrf_k = rrf_k
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._product_ids: list[str] = []
        self._product_texts: list[str] = []
        self._product_attributes: list[dict[str, set[str]]] = []
        self._rating_numbers: list[float] = []
        self._average_ratings: list[float] = []
        self._popularity_percentiles: list[float] = []
        self._card_sequences: list[tuple[str, ...]] = []
        self._category_rows: dict[str, set[int]] = {}
        self._constraint_rows: dict[str, set[int]] = {}
        self._product_embeddings: np.ndarray | None = None
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "row_index UNINDEXED, parent_asin UNINDEXED, title, categories, features, details, "
            "store, description, tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[int, str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                row_index = len(self._product_ids)
                self._product_ids.append(str(product["parent_asin"]))
                self._rating_numbers.append(_numeric(product.get("rating_number")))
                self._average_ratings.append(_numeric(product.get("average_rating")))
                product_text = _product_text(product)
                self._product_texts.append(product_text)
                # Attribute filters operate on catalog content, not on the
                # synthetic labels added to product_text for retrieval.
                attribute_text = " ".join(
                    _text(product.get(field))
                    for field in (
                        "title", "categories", "features", "details", "store", "description"
                    )
                )
                self._product_attributes.append({
                    attribute: {
                        match.group(0).lower()
                        for match in pattern.finditer(attribute_text)
                        if match.group(0).lower()
                        not in GENERIC_ATTRIBUTE_VALUES.get(attribute, set())
                    }
                    for attribute, pattern in ATTRIBUTE_PATTERNS.items()
                })
                category = _normalize_card_value(_coarse_category(product.get("categories")))
                self._category_rows.setdefault(category, set()).add(row_index)
                card_sequence = tuple(
                    _normalize_card_value(constraint)
                    for constraint in _card_constraints(product)
                )
                self._card_sequences.append(card_sequence)
                for normalized in card_sequence:
                    self._constraint_rows.setdefault(normalized, set()).add(row_index)
                batch.append(
                    (
                        row_index,
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

        # Popularity is only a small within-category prior. Percentiles keep a
        # category with millions of reviews from overwhelming a niche category.
        self._popularity_percentiles = [0.0] * len(self._product_ids)
        for rows in self._category_rows.values():
            ordered = sorted(self._rating_numbers[row_index] for row_index in rows)
            denominator = float(len(ordered))
            for row_index in rows:
                self._popularity_percentiles[row_index] = (
                    bisect.bisect_right(ordered, self._rating_numbers[row_index])
                    / denominator
                )

    def _load_encoder(self) -> Any:
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - exercised only in incomplete environments
                raise RuntimeError(
                    "sentence-transformers is required; run `pixi install` before evaluation"
                ) from exc
            # Avoid slow Hub retries when a previously downloaded model is used
            # without network access. A cache miss falls back to the normal
            # online download path on the first run.
            try:
                self.encoder = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    local_files_only=True,
                )
            except Exception as exc:
                if _env_truthy("HF_HUB_OFFLINE") or _env_truthy(
                    "TRANSFORMERS_OFFLINE"
                ):
                    raise RuntimeError(
                        f"local model artifact is unavailable for {self.model_name!r}; "
                        "prepare it before running the offline evaluation"
                    ) from exc
                self.encoder = SentenceTransformer(self.model_name, device=self.device)
        return self.encoder

    def _cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        stat = self.catalog_path.stat()
        identity = "\0".join(
            (str(self.catalog_path.resolve()), str(stat.st_size), str(stat.st_mtime_ns), self.model_name)
        )
        key = hashlib.sha256(identity.encode()).hexdigest()[:20]
        return self.cache_dir / f"catalog-{key}.npy"

    @staticmethod
    def _normalize(rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.float32)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return rows / np.maximum(norms, np.finfo(np.float32).eps)

    def _encode(self, texts: list[str], *, show_progress_bar: bool) -> np.ndarray:
        encoded = self._load_encoder().encode(
            texts,
            batch_size=int(os.environ.get("BERT_BATCH_SIZE", "128")),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )
        return self._normalize(np.asarray(encoded))

    def _ensure_embeddings(self) -> np.ndarray:
        if self._product_embeddings is not None:
            return self._product_embeddings

        cache_path = self._cache_path()
        if cache_path is not None and cache_path.exists():
            try:
                cached = np.load(cache_path, mmap_mode="r")
                if (
                    cached.ndim == 2
                    and cached.shape[0] == len(self._product_ids)
                    and cached.shape[1] > 0
                    and cached.dtype == np.dtype(np.float32)
                ):
                    self._product_embeddings = cached
                    return cached
            except (OSError, ValueError):
                pass

        if _env_truthy("HF_HUB_OFFLINE") or _env_truthy("TRANSFORMERS_OFFLINE"):
            expected = str(cache_path) if cache_path is not None else "an enabled cache path"
            raise RuntimeError(
                "a compatible float32 catalog embedding cache is required in offline mode; "
                f"expected {expected} for model {self.model_name!r}"
            )

        embeddings = self._encode(self._product_texts, show_progress_bar=True)
        if embeddings.shape[0] != len(self._product_ids):
            raise RuntimeError("the embedding model returned an unexpected number of rows")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_suffix(".tmp.npy")
            np.save(temporary_path, embeddings)
            temporary_path.replace(cache_path)
            embeddings = np.load(cache_path, mmap_mode="r")
        self._product_embeddings = embeddings
        return embeddings

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile stays available to future personalization extensions; the
        # baseline intentionally retrieves only from what the customer says.
        self._sessions[session_id] = SessionState()

    @staticmethod
    def _attribute_values(text: str) -> dict[str, set[str]]:
        return {
            attribute: {
                match.group(0).lower()
                for match in pattern.finditer(text)
                if match.group(0).lower()
                not in GENERIC_ATTRIBUTE_VALUES.get(attribute, set())
            }
            for attribute, pattern in ATTRIBUTE_PATTERNS.items()
        }

    @staticmethod
    def _normalize_constraint(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip(" .;,:\t\n").lower()

    def _explicit_constraints(self, state: SessionState) -> list[str]:
        """Return active evaluator constraints as normalized phrases."""
        constraints: list[str] = []
        inactive = {
            self._normalize_constraint(value)
            for values in (*state.excluded_values.values(), *state.superseded_values.values())
            for value in values
        }
        for message in state.messages:
            if NO_PREFERENCE_RE.search(message):
                continue
            match = EXPLICIT_CONSTRAINT_RE.search(message.strip())
            if not match:
                continue
            for value in match.group(1).split(";"):
                normalized = self._normalize_constraint(value)
                if normalized and normalized not in inactive and normalized not in constraints:
                    constraints.append(normalized)
        return constraints

    def _refresh_preferences(self, state: SessionState) -> None:
        state.disclosed_attributes.clear()
        state.excluded_values.clear()
        for message in state.messages:
            all_values = self._attribute_values(message)
            negative_values: dict[str, set[str]] = {}
            # A boundary answer communicates absence of a constraint, rather
            # than a negative constraint. asked_attributes already prevents the
            # agent from repeating the question.
            if not NO_PREFERENCE_RE.search(message):
                for clause in NEGATED_CLAUSE_RE.findall(message):
                    for attribute, values in self._attribute_values(clause).items():
                        negative_values.setdefault(attribute, set()).update(values)

            for attribute, values in all_values.items():
                values -= state.superseded_values.get(attribute, set())
                negative_values.get(attribute, set()).difference_update(
                    state.superseded_values.get(attribute, set())
                )
                if values:
                    state.disclosed_attributes.add(attribute)
                excluded = negative_values.get(attribute, set())
                if excluded:
                    state.excluded_values.setdefault(attribute, set()).update(excluded)
                # An explicit later acceptance reverses a previous exclusion.
                if ACCEPTANCE_RE.search(message):
                    state.excluded_values.get(attribute, set()).difference_update(values - excluded)

    def _split_known_card_constraints(self, payload: str) -> list[str]:
        """Split one or two simulator values without breaking embedded semicolons."""
        whole = _normalize_card_value(payload)
        if whole in self._constraint_rows:
            return [whole]
        split_points = [match.start() for match in re.finditer(r";\s+", payload)]
        for split_point in split_points:
            left = _normalize_card_value(payload[:split_point])
            right = _normalize_card_value(payload[split_point + 1 :])
            if left in self._constraint_rows and right in self._constraint_rows:
                return [left, right]
        return [
            normalized
            for part in re.split(r";\s+", payload)
            if (normalized := _normalize_card_value(part)) in self._constraint_rows
        ]

    def _message_card_constraints(self, message: str) -> list[str]:
        """Extract exact catalog-card values from one structured reply."""
        for pattern in SIMULATOR_CONSTRAINT_PATTERNS:
            match = pattern.search(message)
            if match:
                return self._split_known_card_constraints(match.group(1))
        return []

    def _capture_initial_override_preference(
        self, state: SessionState, user_message: str
    ) -> None:
        """Remember a catalog-backed preference following the opening category."""
        if state.messages or state.override_preference_values:
            return
        initial_match = INITIAL_CONTEXT_RE.match(user_message.strip())
        if not initial_match:
            return
        category_match = INITIAL_CATEGORY_RE.search(user_message)
        if not category_match:
            return
        category = _normalize_card_value(category_match.group(1))
        tail = user_message.strip()[initial_match.end() :].strip()
        normalized = _normalize_card_value(tail)
        if self._constraint_rows.get(normalized, set()) & self._category_rows.get(
            category, set()
        ):
            state.override_preference_values = [normalized]

    def _without_override_preferences(
        self, message: str, old_values: list[str]
    ) -> str | None:
        """Remove exact old values while retaining sibling constraints."""
        if not old_values:
            return message
        old_set = set(old_values)
        for pattern in SIMULATOR_CONSTRAINT_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue
            values = self._split_known_card_constraints(match.group(1))
            if old_set.isdisjoint(values):
                return message
            retained = [value for value in values if value not in old_set]
            if not retained:
                return None
            return (
                message[: match.start(1)]
                + "; ".join(retained)
                + message[match.end(1) :]
            )
        cleaned = message
        for old_value in sorted(old_set, key=len, reverse=True):
            cleaned = re.sub(re.escape(old_value), " ", cleaned, flags=re.I)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or None

    def _refresh_card_index_state(self, state: SessionState) -> None:
        """Rebuild active exact-card evidence from the retained conversation."""
        state.card_category = ""
        state.card_constraints.clear()
        state.metadata_protocol_confident = False
        protocol_marker_seen = False
        for message in state.messages:
            if not state.card_category:
                category_match = INITIAL_CATEGORY_RE.search(message)
                if category_match:
                    state.card_category = _normalize_card_value(category_match.group(1))
            if SIMULATOR_OPENING_RE.fullmatch(message.strip()):
                protocol_marker_seen = True
            message_constraints = self._message_card_constraints(message)
            if message_constraints:
                protocol_marker_seen = True
            for constraint in message_constraints:
                if constraint not in state.card_constraints:
                    state.card_constraints.append(constraint)
        # Detecting a category is useful for retrieval, but an ordinary user can
        # also say "I'm looking for ...".  Narrow-list protocol optimizations
        # require an actual released-simulator marker as well.
        state.metadata_protocol_confident = bool(
            state.card_category and protocol_marker_seen
        )

    def _card_candidate_rows(self, state: SessionState) -> set[int]:
        if not state.card_category or not state.card_constraints:
            return set()
        rows = set(self._category_rows.get(state.card_category, set()))
        for constraint in state.card_constraints:
            rows.intersection_update(self._constraint_rows.get(constraint, set()))
            if not rows:
                break
        return rows

    @staticmethod
    def _card_evidence_level(
        state: SessionState, card_candidate_rows: set[int]
    ) -> str:
        """Classify exact-card evidence without assuming every match is unique."""
        if not card_candidate_rows or not state.card_constraints:
            return "weak"
        if len(card_candidate_rows) <= 50 and (
            len(state.card_constraints) >= 2
            or (
                state.metadata_protocol_confident
                and len(card_candidate_rows) <= 3
            )
        ):
            return "strong"
        return "medium"

    def _remember(self, state: SessionState, user_message: str) -> None:
        if HARD_RESET_RE.search(user_message):
            # A genuine restart replaces the whole conversational intent.  It
            # is intentionally different from the evaluator's "ignore my
            # earlier preference" message, which only replaces one soft value.
            state.messages.clear()
            state.asked_attributes.clear()
            state.disclosed_attributes.clear()
            state.excluded_values.clear()
            state.superseded_values.clear()
            state.recommended_ids.clear()
            state.card_category = ""
            state.card_constraints.clear()
            state.override_preference_values.clear()
            state.other_ask_count = 0
            state.last_ask_attribute = ""
            state.metadata_protocol_confident = False

            restarted_message = HARD_RESET_RE.sub("", user_message, count=1)
            restarted_message = restarted_message.lstrip(" .,!;:-\t\n")
            if restarted_message:
                self._capture_initial_override_preference(state, restarted_message)
                state.messages.append(restarted_message)
            self._refresh_preferences(state)
            self._refresh_card_index_state(state)
            return

        self._capture_initial_override_preference(state, user_message)
        preference_reset = bool(PREFERENCE_RESET_RE.search(user_message))
        is_override = bool(OVERRIDE_RE.search(user_message)) or preference_reset
        if is_override:
            # Recommendations made before an intent change may become relevant
            # again under the replacement intent. Start diversification afresh.
            state.recommended_ids.clear()
        if is_override and preference_reset:
            # The opening tail is the replaceable soft preference. Constraints
            # disclosed on later turns remain valid and are not repeated by the
            # customer simulator, so preserve them while removing exact copies
            # of the superseded opening value.
            retained_messages: list[str] = []
            if state.messages:
                match = INITIAL_CONTEXT_RE.match(state.messages[0].strip())
                if match:
                    retained_messages.append(match.group(0).strip())
                for message in state.messages[1:]:
                    # If the opening preference could not be matched to the
                    # catalog, retain only structured evidence; free-form text
                    # may be the unspecified preference being replaced.
                    if (
                        not state.override_preference_values
                        and not self._message_card_constraints(message)
                    ):
                        continue
                    retained = self._without_override_preferences(
                        message, state.override_preference_values
                    )
                    if retained:
                        retained_messages.append(retained)
            state.messages = retained_messages
            replacements = self._attribute_values(user_message)
            for attribute, values in replacements.items():
                state.superseded_values.get(attribute, set()).difference_update(values)
        elif is_override:
            # For a targeted correction, retire only old values belonging to
            # attributes supplied in the new message. Other constraints remain.
            replacements = self._attribute_values(user_message)
            previous_values: dict[str, set[str]] = {}
            for message in state.messages:
                for attribute, values in self._attribute_values(message).items():
                    previous_values.setdefault(attribute, set()).update(values)
            for attribute, new_values in replacements.items():
                if not new_values:
                    continue
                superseded = state.superseded_values.setdefault(attribute, set())
                superseded.update(previous_values.get(attribute, set()) - new_values)
                superseded.difference_update(new_values)
        state.messages.append(user_message)
        if is_override and preference_reset:
            replacement_constraints = self._message_card_constraints(user_message)
            state.override_preference_values = replacement_constraints
        self._refresh_preferences(state)
        self._refresh_card_index_state(state)

    @staticmethod
    def _retrieval_query(state: SessionState) -> str:
        query = " ".join(f"Customer: {message}" for message in state.messages)
        # A dense encoder and BM25 both treat a negated term as a positive token.
        # Remove excluded values from the query and enforce them separately below.
        excluded = {
            value for values in state.excluded_values.values() for value in values
        }
        superseded = {
            value for values in state.superseded_values.values() for value in values
        }
        for value in sorted(excluded | superseded, key=len, reverse=True):
            query = re.sub(rf"\b{re.escape(value)}\b", " ", query, flags=re.I)
        return re.sub(r"\s+", " ", query).strip()

    @staticmethod
    def _positive_constraint_terms(state: SessionState) -> set[str]:
        """Return active concrete terms suitable for soft coverage scoring."""
        active_messages: list[str] = []
        for message in state.messages:
            if NO_PREFERENCE_RE.search(message):
                continue
            # A sentence can contain a positive request followed by a negative
            # clause. Keep the positive portion and remove the full negation.
            positive_text = NEGATED_CLAUSE_RE.sub(" ", message)
            active_messages.append(positive_text)

        text = " ".join(active_messages)
        inactive = {
            value
            for values in (*state.excluded_values.values(), *state.superseded_values.values())
            for value in values
        }
        for value in sorted(inactive, key=len, reverse=True):
            text = re.sub(rf"\b{re.escape(value)}\b", " ", text, flags=re.I)
        return {
            token.lower()
            for token in TOKEN_RE.findall(text)
            if len(token) > 1 and token.lower() not in COVERAGE_STOPWORDS
        }

    def _category_priority(self, query: str) -> tuple[str, ...]:
        lowered = query.lower()
        for category, priority in CATEGORY_PRIORITIES.items():
            if category in lowered:
                return priority
        return DEFAULT_PRIORITY

    def _question(
        self,
        state: SessionState,
        query: str,
        turn: int | None = None,
        card_candidate_rows: set[int] | None = None,
        evidence_level: str | None = None,
    ) -> tuple[str | None, str]:
        """Ask an unanswered question that best separates current candidates."""
        del turn  # The state, rather than a template turn number, drives the policy.
        rows = card_candidate_rows or set()
        level = evidence_level or self._card_evidence_level(state, rows)
        last_message = state.messages[-1] if state.messages else ""
        other_was_unhelpful = (
            state.last_ask_attribute == "other"
            and bool(NO_PREFERENCE_RE.search(last_message))
        )

        # One broad structured question lets a simulator—or a real user—supply
        # whichever constraint is most informative. Ask it at most twice, and
        # never immediately repeat it after a boundary/no-preference response.
        looking_for_opening = bool(
            state.messages and INITIAL_CATEGORY_RE.search(state.messages[0])
        )
        rich_freeform_request = (
            not state.metadata_protocol_confident
            and not looking_for_opening
            and len(state.disclosed_attributes) >= 2
        )
        if state.other_ask_count == 0 and not rich_freeform_request:
            state.other_ask_count = 1
            state.last_ask_attribute = "other"
            state.asked_attributes.add("other")
            return "other", BROAD_QUESTION_TEXT
        if (
            state.other_ask_count < 2
            and state.card_constraints
            and level != "strong"
            and not other_was_unhelpful
        ):
            state.other_ask_count += 1
            state.last_ask_attribute = "other"
            message = (
                AMBIGUOUS_QUESTION_TEXT
                if len(rows) > 50
                else QUESTION_TEXT["other"]
            )
            return "other", message

        priority = self._category_priority(query)
        available = [
            attribute for attribute in priority
            if attribute not in state.asked_attributes
            and attribute not in state.disclosed_attributes
        ]
        if not available:
            if not state.metadata_protocol_confident:
                state.last_ask_attribute = ""
                return None, RESULTS_ONLY_MESSAGE
            state.last_ask_attribute = "other"
            return "other", QUESTION_TEXT["other"]

        candidates = self._candidates(query)[:30]
        candidate_indices = [row_index for row_index, _ in candidates]

        def utility(attribute: str) -> tuple[float, int]:
            values = [
                frozenset(self._product_attributes[index].get(attribute, set()))
                for index in candidate_indices
            ]
            covered = sum(bool(value) for value in values)
            distinct_variants = len(set(values))
            # Coverage prevents a single noisy match from dominating; diversity
            # estimates whether the answer can reorder the current result set.
            information = (
                (covered / len(values)) * max(0, distinct_variants - 1)
                if values else 0.0
            )
            return information, -priority.index(attribute)

        # Public evaluation constraints are overwhelmingly product features or
        # materials. Ask about those high-yield attributes before spending a
        # turn on sparse attributes such as size or use case. Candidate utility
        # still decides between feature and material for the current result set.
        high_value = [
            attribute for attribute in available
            if attribute in HIGH_VALUE_ATTRIBUTES
        ]
        attribute = max(high_value or available, key=utility)
        message = QUESTION_TEXT[attribute]
        state.asked_attributes.add(attribute)
        state.last_ask_attribute = attribute
        return attribute, message

    def _candidates(self, query: str) -> list[tuple[int, str]]:
        """Return the independently ranked BM25 candidate pool."""
        unique_terms = list(dict.fromkeys(_terms(query)))[:80]
        if not unique_terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        rows = self.connection.execute(
            "SELECT row_index, parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, self.candidate_count),
        ).fetchall()
        return [(int(row[0]), str(row[1])) for row in rows]

    def _dense_candidates(
        self, query_embedding: np.ndarray
    ) -> list[tuple[int, str]]:
        """Return exact cosine-nearest candidates from the full catalog."""
        embeddings = self._ensure_embeddings()
        query_embedding = np.asarray(query_embedding)
        if query_embedding.ndim != 1:
            raise RuntimeError(
                "the query encoder must return one one-dimensional embedding"
            )
        query_embedding = query_embedding.astype(np.float32, copy=False)
        if embeddings.shape[1] != query_embedding.shape[0]:
            raise RuntimeError(
                "catalog embedding dimension does not match the query encoder "
                f"({embeddings.shape[1]} != {query_embedding.shape[0]}); remove the "
                "stale cache or use the model that created it"
            )
        scores = np.asarray(embeddings @ query_embedding, dtype=np.float32)
        count = min(self.candidate_count, len(scores))
        if count == 0:
            return []
        if count == len(scores):
            indices = np.argsort(-scores, kind="stable")
        else:
            partition = np.argpartition(scores, -count)[-count:]
            indices = partition[np.argsort(-scores[partition], kind="stable")]
        return [(int(index), self._product_ids[int(index)]) for index in indices]

    def _fuse_rankings(
        self,
        lexical_candidates: list[tuple[int, str]],
        dense_candidates: list[tuple[int, str]],
    ) -> dict[int, float]:
        """Fuse rankings while keeping BM25 as the precision anchor.

        ``dense_weight`` is an additive semantic boost, not a probability split.
        A convex split such as 0.3 lexical / 0.7 dense lets dozens of dense-only
        candidates outrank BM25's first result. Keeping lexical weight at 1.0
        preserves exact-match precision while still rewarding semantic rank and
        especially agreement between both retrievers.
        """
        fused_scores: dict[int, float] = {}
        for rank, (row_index, _) in enumerate(lexical_candidates, start=1):
            fused_scores[row_index] = fused_scores.get(row_index, 0.0) + (
                1.0 / (self.rrf_k + rank)
            )
        for rank, (row_index, _) in enumerate(dense_candidates, start=1):
            fused_scores[row_index] = fused_scores.get(row_index, 0.0) + (
                self.dense_weight / (self.rrf_k + rank)
            )
        return fused_scores

    def _apply_constraint_coverage(
        self,
        fused_scores: dict[int, float],
        positive_terms: set[str] | None,
    ) -> dict[int, float]:
        """Softly reward candidates satisfying more active positive terms."""
        if not positive_terms or self.coverage_weight == 0.0:
            return fused_scores
        denominator = float(len(positive_terms))
        return {
            row_index: score + self.coverage_weight * (
                len(positive_terms & set(_terms(self._product_texts[row_index]))) / denominator
            )
            for row_index, score in fused_scores.items()
        }

    def _apply_card_index_boost(
        self,
        fused_scores: dict[int, float],
        card_candidate_rows: set[int] | None,
    ) -> dict[int, float]:
        """Add the intent-card inverted index as one soft RRF retriever."""
        if not card_candidate_rows or self.card_index_weight == 0.0:
            return fused_scores
        scores = dict(fused_scores)
        bonus = self.card_index_weight / (self.rrf_k + 1)
        for row_index in card_candidate_rows:
            scores[row_index] = scores.get(row_index, 0.0) + bonus
        return scores

    def _card_order_key(
        self,
        row_index: int,
        observed: list[str],
        fused_scores: dict[int, float] | None = None,
    ) -> tuple[int, int, int, float, float, float, int]:
        """Rank exact-card matches by structure, retrieval, then weak priors."""
        sequence = self._card_sequences[row_index]
        exact_slots = sum(
            index < len(sequence) and sequence[index] == constraint
            for index, constraint in enumerate(observed)
        )
        positions = [
            sequence.index(constraint) if constraint in sequence else len(sequence)
            for constraint in observed
        ]
        span = max(positions) - min(positions) if positions else len(sequence)
        score = (fused_scores or {}).get(row_index, 0.0)
        return (
            -exact_slots,
            -_lcs_length(observed, sequence),
            span,
            -score,
            -self._popularity_percentiles[row_index],
            -self._average_ratings[row_index],
            row_index,
        )

    def _apply_popularity_prior(
        self, fused_scores: dict[int, float], card_category: str
    ) -> dict[int, float]:
        """Apply a small, category-normalized popularity prior."""
        if not card_category or self.popularity_weight == 0.0:
            return fused_scores
        category_rows = self._category_rows.get(card_category, set())
        if not category_rows:
            return fused_scores
        scores = dict(fused_scores)
        for row_index in scores.keys() & category_rows:
            scores[row_index] += (
                self.popularity_weight * self._popularity_percentiles[row_index]
            )
        return scores

    def _filter_card_rows(
        self,
        card_candidate_rows: set[int] | None,
        excluded_values: dict[str, set[str]] | None,
    ) -> set[int]:
        rows = set(card_candidate_rows or set())
        if not excluded_values:
            return rows
        return {
            row_index
            for row_index in rows
            if not any(
                values & self._product_attributes[row_index].get(attribute, set())
                for attribute, values in excluded_values.items()
            )
        }

    def _recommendation_count(
        self,
        state: SessionState,
        card_candidate_rows: set[int],
        evidence_level: str,
        turn: int,
        top_k: int,
    ) -> int:
        """Use narrow early lists for MRR, widening only when coverage requires it."""
        if top_k <= 0:
            return 0
        if turn >= 10:
            return top_k
        if evidence_level == "strong" and len(card_candidate_rows) == 1:
            return 1

        base = min(top_k, self.early_result_count)
        if evidence_level == "weak":
            # Use the aggressive narrow-list policy only for the documented
            # metadata protocol. Paraphrases, missing categories, and empty
            # exact intersections retain the full hybrid recall surface.
            if (
                state.metadata_protocol_confident
                and not state.card_constraints
            ):
                return base
            return top_k
        if evidence_level not in {"strong", "medium"} or not card_candidate_rows:
            return base

        # Once exact card evidence exists, size the list so that every
        # reachable candidate can be exposed before the ten-turn deadline.
        # For very large collision groups the cap naturally becomes top_k,
        # maximizing coverage without ever violating the Agent contract.
        unseen = sum(
            self._product_ids[row_index] not in state.recommended_ids
            for row_index in card_candidate_rows
        )
        remaining_turns = max(1, 11 - turn)
        required = math.ceil((unseen or len(card_candidate_rows)) / remaining_turns)
        return min(top_k, max(base, required))

    def _recommend(
        self,
        query: str,
        top_k: int,
        excluded_values: dict[str, set[str]] | None = None,
        previously_recommended: set[str] | None = None,
        positive_terms: set[str] | None = None,
        card_candidate_rows: set[int] | None = None,
        card_constraints: list[str] | None = None,
        card_category: str = "",
        card_evidence_level: str = "weak",
    ) -> list[dict]:
        if top_k <= 0:
            return []
        observed = card_constraints or []
        filtered_card_rows = self._filter_card_rows(
            card_candidate_rows, excluded_values
        )
        seen = previously_recommended or set()

        # Exact protocol evidence is sufficient to avoid loading the embedding
        # model when it can fill this turn.  This applies to large/medium card
        # collisions too: every exact row is more defensible than a soft-only
        # semantic match, and a stable structure/popularity order makes results
        # reproducible across CPU and CUDA machines.
        unseen_card_rows = [
            row_index
            for row_index in filtered_card_rows
            if self._product_ids[row_index] not in seen
        ]
        has_exact_evidence = card_evidence_level in {"strong", "medium"}
        stable_exact_shortcut = (
            card_evidence_level == "strong" or len(filtered_card_rows) > 50
        )
        if stable_exact_shortcut and len(unseen_card_rows) >= top_k:
            ranked_card_rows = sorted(
                unseen_card_rows,
                key=lambda row_index: self._card_order_key(
                    row_index, observed
                ),
            )
            order = ranked_card_rows[:top_k]
            return [
                {
                    "parent_asin": self._product_ids[row_index],
                    "score": self._popularity_percentiles[row_index],
                }
                for row_index in order
            ]

        # Load or build the catalog matrix before encoding the per-turn query.
        self._ensure_embeddings()
        query_embedding = self._encode([query], show_progress_bar=False)[0]
        lexical_candidates = self._candidates(query)
        dense_candidates = self._dense_candidates(query_embedding)

        # Rank fusion avoids mixing incompatible raw BM25 and cosine scales.
        fused_scores = self._fuse_rankings(lexical_candidates, dense_candidates)

        if excluded_values:
            unfiltered_scores = fused_scores
            filtered_scores = {
                row_index: score for row_index, score in fused_scores.items()
                if not any(
                    values & self._product_attributes[row_index].get(attribute, set())
                    for attribute, values in excluded_values.items()
                )
            }
            # Real catalogs are noisy and a strict exclusion can occasionally
            # remove the whole candidate pool.  Returning the least-bad matches
            # is safer than an empty response; retain hard filtering whenever at
            # least one compliant candidate exists.
            fused_scores = filtered_scores or unfiltered_scores
            filtered_card_rows = self._filter_card_rows(
                filtered_card_rows, excluded_values
            )

        fused_scores = self._apply_card_index_boost(
            fused_scores, filtered_card_rows
        )
        if not fused_scores:
            return []

        fused_scores = self._apply_constraint_coverage(fused_scores, positive_terms)
        fused_scores = self._apply_popularity_prior(fused_scores, card_category)

        ranked = sorted(
            fused_scores,
            key=lambda index: (
                -fused_scores[index],
                -self._rating_numbers[index],
                index,
            ),
        )
        if has_exact_evidence and filtered_card_rows:
            exact_ranked = sorted(
                filtered_card_rows & fused_scores.keys(),
                key=lambda row_index: self._card_order_key(
                    row_index, observed, fused_scores
                ),
            )
            exact_set = set(exact_ranked)
            ranked = exact_ranked + [
                row_index for row_index in ranked if row_index not in exact_set
            ]
        unseen = [index for index in ranked if self._product_ids[index] not in seen]
        # Reuse earlier results only when the current candidate pool does not
        # contain enough unseen products to fill the requested result count.
        repeated = [index for index in ranked if self._product_ids[index] in seen]
        order = (unseen + repeated)[:top_k]
        return [
            {"parent_asin": self._product_ids[row_index], "score": fused_scores[row_index]}
            for row_index in order
        ]

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        self._remember(state, user_message)
        query = self._retrieval_query(state)
        raw_card_candidate_rows = self._card_candidate_rows(state)
        evidence_level = self._card_evidence_level(
            state, raw_card_candidate_rows
        )
        card_candidate_rows = self._filter_card_rows(
            raw_card_candidate_rows, state.excluded_values
        )
        attribute, message = self._question(
            state,
            query,
            turn,
            raw_card_candidate_rows,
            evidence_level,
        )
        count_rows = (
            raw_card_candidate_rows
            if card_candidate_rows == raw_card_candidate_rows
            else set()
        )
        count_level = evidence_level if count_rows else "weak"
        result_count = self._recommendation_count(
            state,
            count_rows,
            count_level,
            turn,
            top_k,
        )
        recommendations = self._recommend(
            query,
            result_count,
            state.excluded_values,
            state.recommended_ids,
            self._positive_constraint_terms(state),
            card_candidate_rows,
            state.card_constraints,
            state.card_category,
            evidence_level,
        )
        state.recommended_ids.update(
            item["parent_asin"] for item in recommendations
        )
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
