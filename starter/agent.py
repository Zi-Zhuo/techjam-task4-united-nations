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
EARLY_QUESTIONS = (
    ("feature", "To narrow this down, what features matter most to you?"),
    ("material", "Thanks. Do you have a material preference, or anything you would rather avoid?"),
    ("use_case", "Got it. How are you planning to use it most often?"),
)
LATER_QUESTIONS = (
    ("color", "Is there a color you would prefer?"),
    ("style", "What kind of style or fit would suit you best?"),
    ("budget", "Do you have a budget range in mind?"),
    ("size", "Are there any size or width requirements I should account for?"),
    ("brand", "Do you have a preferred brand?"),
    ("other", "Is there anything else that would make one option stand out?"),
)


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
    ) -> None:
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError("dense_weight must be between 0 and 1")

        self.catalog_path = Path(catalog_path)
        self.model_name = model_name or os.environ.get(
            "BERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.encoder = encoder
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.candidate_count = candidate_count
        self.dense_weight = dense_weight
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._product_ids: list[str] = []
        self._product_texts: list[str] = []
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
                self._product_texts.append(_product_text(product))
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

    def _remember(self, state: SessionState, user_message: str) -> None:
        if OVERRIDE_RE.search(user_message):
            # Preserve the initial category request and replace later preferences.
            state.messages = state.messages[:1]
        state.messages.append(user_message)

    def _question(self, state: SessionState, turn: int) -> tuple[str, str]:
        if turn <= len(EARLY_QUESTIONS):
            attribute, message = EARLY_QUESTIONS[turn - 1]
        else:
            available = [item for item in LATER_QUESTIONS if item[0] not in state.asked_attributes]
            attribute, message = (available or [LATER_QUESTIONS[-1]])[0]
        state.asked_attributes.add(attribute)
        return attribute, message

    def _candidates(self, query: str) -> list[tuple[int, str]]:
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

    def _recommend(self, query: str, top_k: int) -> list[dict]:
        candidates = self._candidates(query)
        if not candidates:
            return []
        embeddings = self._ensure_embeddings()
        query_embedding = self._encode([query], show_progress_bar=False)[0]
        indices = np.fromiter((row_index for row_index, _ in candidates), dtype=np.int64)
        dense_scores = np.asarray(embeddings[indices] @ query_embedding, dtype=np.float32)
        if len(candidates) == 1:
            lexical_scores = np.ones(1, dtype=np.float32)
        else:
            lexical_scores = 1.0 - np.arange(len(candidates), dtype=np.float32) / (len(candidates) - 1)
        scores = self.dense_weight * dense_scores + (1.0 - self.dense_weight) * lexical_scores
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {"parent_asin": candidates[int(position)][1], "score": float(scores[position])}
            for position in order
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
        query = " ".join(f"Customer: {message}" for message in state.messages)
        attribute, message = self._question(state, turn)
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": self._recommend(query, top_k),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
