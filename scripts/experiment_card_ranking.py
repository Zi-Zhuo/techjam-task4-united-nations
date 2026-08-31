from __future__ import annotations

import argparse
import bisect
import json
import time
from collections import defaultdict
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent, SessionState, _card_constraints, _normalize_card_value


SCENARIO_SHARES = {
    "buying": 0.40,
    "browsing": 0.40,
    "intent_override": 0.15,
    "boundary": 0.05,
}
MODE_LABELS = {
    "baseline": "current equal-bonus soft card index",
    "ordered": "ranked card retriever without popularity",
    "popularity": "equal-bonus card index plus category popularity",
    "combined": "ranked card retriever plus category popularity",
}


def _stratified_subset(
    samples: list[dict], sample_count: int, offset_per_scenario: int
) -> list[dict]:
    if sample_count >= len(samples) and offset_per_scenario == 0:
        return samples
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["scenario_type"])].append(sample)

    requested = {
        name: int(round(sample_count * share))
        for name, share in SCENARIO_SHARES.items()
    }
    requested["buying"] += sample_count - sum(requested.values())
    selected: list[dict] = []
    for name in SCENARIO_SHARES:
        start = int(round(offset_per_scenario * SCENARIO_SHARES[name]))
        end = start + requested[name]
        if end > len(grouped[name]):
            raise ValueError(f"not enough {name} samples for offset {offset_per_scenario}")
        selected.extend(grouped[name][start:end])
    return selected


def _lcs_length(left: list[str], right: list[str]) -> int:
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


class CardRankingExperimentAgent(Agent):
    """Production Agent with isolated card-order and popularity switches."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        mode: str = "baseline",
        popularity_weight: float = 0.005,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        super().__init__(catalog_path, model_name=model_name)
        self.mode = mode
        self.popularity_weight = popularity_weight
        self._active_state: SessionState | None = None
        self._card_sequences: list[list[str]] = []
        rating_numbers: list[float] = []

        with Path(catalog_path).open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                product = json.loads(line)
                if str(product["parent_asin"]) != self._product_ids[row_index]:
                    raise RuntimeError("catalog order changed while building experiment metadata")
                self._card_sequences.append(
                    [_normalize_card_value(value) for value in _card_constraints(product)]
                )
                value = product.get("rating_number")
                rating_numbers.append(float(value) if isinstance(value, (int, float)) else 0.0)

        self._popularity_percentiles = [0.0] * len(self._product_ids)
        for rows in self._category_rows.values():
            ordered_values = sorted(rating_numbers[row_index] for row_index in rows)
            denominator = float(len(ordered_values))
            for row_index in rows:
                self._popularity_percentiles[row_index] = (
                    bisect.bisect_right(ordered_values, rating_numbers[row_index]) / denominator
                )

    def _card_order_key(
        self,
        row_index: int,
        observed: list[str],
        fused_scores: dict[int, float],
    ) -> tuple[int, int, int, float, int]:
        sequence = self._card_sequences[row_index]
        exact_slots = sum(
            index < len(sequence) and sequence[index] == constraint
            for index, constraint in enumerate(observed)
        )
        lcs = _lcs_length(observed, sequence)
        positions = [
            sequence.index(constraint) if constraint in sequence else len(sequence)
            for constraint in observed
        ]
        span = (max(positions) - min(positions)) if positions else len(sequence)
        return (-exact_slots, -lcs, span, -fused_scores.get(row_index, 0.0), row_index)

    def _apply_card_index_boost(
        self,
        fused_scores: dict[int, float],
        card_candidate_rows: set[int] | None,
    ) -> dict[int, float]:
        rows = card_candidate_rows or set()
        if self.mode in {"ordered", "combined"} and rows and self.card_index_weight > 0.0:
            scores = dict(fused_scores)
            observed = self._active_state.card_constraints if self._active_state else []
            ranked_rows = sorted(
                rows,
                key=lambda row_index: self._card_order_key(
                    row_index, observed, fused_scores
                ),
            )
            for rank, row_index in enumerate(ranked_rows, start=1):
                scores[row_index] = scores.get(row_index, 0.0) + (
                    self.card_index_weight / (self.rrf_k + rank)
                )
        else:
            scores = super()._apply_card_index_boost(fused_scores, rows)

        if self.mode not in {"popularity", "combined"} or self.popularity_weight == 0.0:
            return scores
        if self._active_state is None or not self._active_state.card_category:
            return scores
        category_rows = self._category_rows.get(self._active_state.card_category, set())
        for row_index in scores.keys() & category_rows:
            scores[row_index] += (
                self.popularity_weight * self._popularity_percentiles[row_index]
            )
        return scores

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
        self._active_state = state
        try:
            recommendations = self._recommend(
                query,
                top_k,
                state.excluded_values,
                state.recommended_ids,
                self._positive_constraint_terms(state),
                self._card_candidate_rows(state),
            )
        finally:
            self._active_state = None
        state.recommended_ids.update(item["parent_asin"] for item in recommendations)
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ordered-card/popularity 2x2 ablation")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--offset-per-scenario", type=int, default=0)
    parser.add_argument("--popularity-weight", type=float, default=0.005)
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODE_LABELS),
        default=list(MODE_LABELS),
    )
    parser.add_argument("--output", default="experiment_card_ranking_results.json")
    args = parser.parse_args()

    samples = _stratified_subset(
        load_jsonl(args.dataset), args.sample_count, args.offset_per_scenario
    )
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = CardRankingExperimentAgent(
        args.catalog,
        popularity_weight=args.popularity_weight,
        model_name=args.model_name,
    )
    results: dict[str, dict] = {}
    try:
        for mode in args.modes:
            agent.mode = mode
            started = time.perf_counter()
            result = evaluate(agent, samples, catalog_ids, categories, products)
            results[mode] = {
                "label": MODE_LABELS[mode],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                **result,
            }
            summary = {key: value for key, value in results[mode].items() if key != "sessions"}
            print(json.dumps({mode: summary}, indent=2), flush=True)
    finally:
        agent.connection.close()

    output = {
        "sample_count": len(samples),
        "offset_per_scenario": args.offset_per_scenario,
        "popularity_weight": args.popularity_weight,
        "model_name": args.model_name,
        "sample_ids": [sample["sample_id"] for sample in samples],
        "controlled_variables": (
            "Same samples, question policy, BM25/dense retrieval, RRF parameters, "
            "constraint coverage, intent override, and diversification."
        ),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
