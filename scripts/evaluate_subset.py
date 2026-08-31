from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent


SCENARIO_SHARES = {
    "buying": 0.40,
    "browsing": 0.40,
    "intent_override": 0.15,
    "boundary": 0.05,
}


def stratified_subset(
    samples: list[dict], sample_count: int, offset_per_scenario: int
) -> list[dict]:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["scenario_type"])].append(sample)

    requested = {
        name: int(round(sample_count * share))
        for name, share in SCENARIO_SHARES.items()
    }
    requested["buying"] += sample_count - sum(requested.values())
    selected: list[dict] = []
    for name, share in SCENARIO_SHARES.items():
        start = int(round(offset_per_scenario * share))
        end = start + requested[name]
        if end > len(grouped[name]):
            raise ValueError(
                f"not enough {name} samples for count={sample_count}, "
                f"offset={offset_per_scenario}"
            )
        selected.extend(grouped[name][start:end])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Small stratified production-Agent evaluation")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--sample-count", type=int, default=40)
    parser.add_argument("--offset-per-scenario", type=int, default=0)
    parser.add_argument("--output", default="results_subset.json")
    args = parser.parse_args()

    samples = stratified_subset(
        load_jsonl(args.dataset), args.sample_count, args.offset_per_scenario
    )
    catalog_ids, categories, products = catalog_index(args.catalog)
    started = time.perf_counter()
    agent = Agent(args.catalog)
    try:
        result = evaluate(agent, samples, catalog_ids, categories, products)
    finally:
        agent.connection.close()

    result["subset"] = {
        "sample_count": len(samples),
        "offset_per_scenario": args.offset_per_scenario,
        "scenario_counts": dict(
            sorted(Counter(str(sample["scenario_type"]) for sample in samples).items())
        ),
        "sample_ids": [str(sample["sample_id"]) for sample in samples],
        "model_name": agent.model_name,
        "device": agent.device,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "sessions"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
