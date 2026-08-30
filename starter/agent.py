from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


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
FULL_RESET_RE = re.compile(
    r"\b(ignore (?:my )?(?:earlier|previous)|start over|forget (?:that|everything))\b",
    re.IGNORECASE,
)
QUESTION_TEXT = dict(QUESTIONS)
DEFAULT_PRIORITY = ("feature", "use_case", "material", "style", "color", "size", "budget", "brand", "other")
CATEGORY_PRIORITIES = {
    "shoe": ("use_case", "size", "material", "style", "color", "brand", "budget", "feature", "other"),
    "dress": ("style", "size", "color", "material", "use_case", "budget", "brand", "feature", "other"),
    "shirt": ("size", "style", "material", "color", "use_case", "budget", "brand", "feature", "other"),
    "jacket": ("use_case", "material", "size", "style", "color", "budget", "brand", "feature", "other"),
}
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
ACCEPTANCE_RE = re.compile(r"\b(?:actually|instead|is fine|are fine|okay|ok|do want|would like)\b", re.I)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key}: {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


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


@dataclass
class SessionState:
    messages: list[str] = field(default_factory=list)
    asked_attributes: set[str] = field(default_factory=set)
    disclosed_attributes: set[str] = field(default_factory=set)
    excluded_values: dict[str, set[str]] = field(default_factory=dict)
    superseded_values: dict[str, set[str]] = field(default_factory=dict)


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
        rrf_k: int = 60,
    ) -> None:
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError("dense_weight must be between 0 and 1")
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")

        self.catalog_path = Path(catalog_path)
        self.model_name = model_name or os.environ.get(
            "BERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.encoder = encoder
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.candidate_count = candidate_count
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._product_ids: list[str] = []
        self._product_texts: list[str] = []
        self._product_attributes: list[dict[str, set[str]]] = []
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
                product_text = _product_text(product)
                self._product_texts.append(product_text)
                self._product_attributes.append({
                    attribute: {match.lower() for match in pattern.findall(product_text)}
                    for attribute, pattern in ATTRIBUTE_PATTERNS.items()
                })
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
                self.encoder = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self.encoder = SentenceTransformer(self.model_name)
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
                if cached.ndim == 2 and cached.shape[0] == len(self._product_ids):
                    self._product_embeddings = cached
                    return cached
            except (OSError, ValueError):
                pass

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
            attribute: {match.group(0).lower() for match in pattern.finditer(text)}
            for attribute, pattern in ATTRIBUTE_PATTERNS.items()
        }

    def _refresh_preferences(self, state: SessionState) -> None:
        state.disclosed_attributes.clear()
        state.excluded_values.clear()
        for message in state.messages:
            all_values = self._attribute_values(message)
            negative_values: dict[str, set[str]] = {}
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

    def _remember(self, state: SessionState, user_message: str) -> None:
        is_override = bool(OVERRIDE_RE.search(user_message))
        if is_override and FULL_RESET_RE.search(user_message):
            # Explicit broad reset language still means all intermediate
            # preferences should be discarded.
            state.messages = state.messages[:1]
            state.superseded_values.clear()
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
        self._refresh_preferences(state)

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

    def _category_priority(self, query: str) -> tuple[str, ...]:
        lowered = query.lower()
        for category, priority in CATEGORY_PRIORITIES.items():
            if category in lowered:
                return priority
        return DEFAULT_PRIORITY

    def _question(self, state: SessionState, query: str) -> tuple[str, str]:
        """Ask an unanswered question that best separates current candidates."""
        priority = self._category_priority(query)
        available = [
            attribute for attribute in priority
            if attribute not in state.asked_attributes
            and attribute not in state.disclosed_attributes
        ]
        if not available:
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

        attribute = max(available, key=utility)
        message = QUESTION_TEXT[attribute]
        state.asked_attributes.add(attribute)
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

    def _recommend(
        self, query: str, top_k: int, excluded_values: dict[str, set[str]] | None = None
    ) -> list[dict]:
        # Load or build the catalog matrix before encoding the per-turn query.
        self._ensure_embeddings()
        query_embedding = self._encode([query], show_progress_bar=False)[0]
        lexical_candidates = self._candidates(query)
        dense_candidates = self._dense_candidates(query_embedding)

        # BM25 and cosine similarity have unrelated numeric scales. Weighted
        # Reciprocal Rank Fusion combines their rank positions instead, while
        # retaining dense_weight as an intuitive balance between retrievers.
        fused_scores: dict[int, float] = {}
        for rank, (row_index, _) in enumerate(lexical_candidates, start=1):
            fused_scores[row_index] = fused_scores.get(row_index, 0.0) + (
                (1.0 - self.dense_weight) / (self.rrf_k + rank)
            )
        for rank, (row_index, _) in enumerate(dense_candidates, start=1):
            fused_scores[row_index] = fused_scores.get(row_index, 0.0) + (
                self.dense_weight / (self.rrf_k + rank)
            )

        if excluded_values:
            fused_scores = {
                row_index: score for row_index, score in fused_scores.items()
                if not any(
                    values & self._product_attributes[row_index].get(attribute, set())
                    for attribute, values in excluded_values.items()
                )
            }
        if not fused_scores:
            return []

        order = sorted(fused_scores, key=lambda index: (-fused_scores[index], index))[:top_k]
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
        attribute, message = self._question(state, query)
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": self._recommend(query, top_k, state.excluded_values),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
