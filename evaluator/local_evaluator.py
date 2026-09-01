from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import uuid
from collections import defaultdict
from pathlib import Path

from starter.agent import Agent


MAX_TURNS = 10
TOP_K = 10
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)


def searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_constraint(value: str, limit: int) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def intent_card(product: dict, limit: int = 180) -> dict:
    title = _clean_constraint(str(product.get("title") or "product"), limit)
    candidates = [*_flatten_values(product.get("features")), *_flatten_values(product.get("details"))]
    corpus = searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(dict.fromkeys(_clean_constraint(item, limit) for item in candidates if _clean_constraint(item, limit)))
    if not cleaned:
        cleaned = [title]
    return {
        "target_category": title,
        "hard_constraints": cleaned[:2],
        "soft_preferences": cleaned[2:4] or cleaned[:1],
    }


def behavior_for(scenario: str, card: dict, rng: random.Random) -> dict:
    behavior: dict = {"scenario_type": scenario}
    if scenario == "intent_override":
        hard = card["hard_constraints"]
        soft = card["soft_preferences"]
        old_value = soft[-1] if soft else "I prefer a different style."
        new_value = hard[0] if hard else "Please prioritize the target requirements."
        behavior["override"] = {
            "turn": rng.choice([3, 4]),
            "old_value": old_value,
            "new_value": new_value,
            "message": f"Actually, ignore my earlier preference. What I need is: {new_value}.",
        }
    return behavior


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_recommendations(payload: object, catalog_ids: set[str]) -> list[str]:
    if not isinstance(payload, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in payload:
        value = item.get("parent_asin", "") if isinstance(item, dict) else item
        parent_asin = str(value).strip()
        if not parent_asin or parent_asin in seen or parent_asin not in catalog_ids:
            continue
        seen.add(parent_asin)
        result.append(parent_asin)
        if len(result) >= TOP_K:
            break
    return result


def catalog_index(catalog_path: str | Path) -> tuple[set[str], dict[str, list[str]], dict[str, dict]]:
    identifiers: set[str] = set()
    categories: dict[str, list[str]] = {}
    products: dict[str, dict] = {}
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            parent_asin = str(product["parent_asin"])
            identifiers.add(parent_asin)
            categories[parent_asin] = [str(value) for value in product.get("categories") or []]
            products[parent_asin] = product
    return identifiers, categories, products


def coarse_category(values: list[str]) -> str:
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def classify_constraint(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def initial_message(sample: dict, category: str, disclosed: set[str]) -> str:
    scenario = sample["scenario_type"]
    if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
        constraint = str(sample["intent_card"]["hard_constraints"][0])
        disclosed.add(constraint)
        return f"I'm looking for {category}. A key requirement is: {constraint}."
    if scenario == "intent_override":
        old_value = str(sample["behavior"]["override"]["old_value"])
        return f"I'm looking for {category}. {old_value}"
    return f"I'm looking for {category}, but I'm still exploring."


def customer_reply(sample: dict, ask_attribute: object, disclosed: set[str], boundary_used: bool) -> tuple[str, bool]:
    attribute = ask_attribute if isinstance(ask_attribute, str) else None
    if sample["scenario_type"] == "boundary" and not boundary_used and attribute:
        return f"I don't have a preference for {attribute}; please use your judgment.", True
    if not attribute:
        return "Those options are not quite right yet. Ask me about one specific attribute.", boundary_used
    if attribute not in ALLOWED_ATTRIBUTES:
        attribute = "other"
    constraints = [
        *[str(value) for value in sample["intent_card"].get("hard_constraints", [])],
        *[str(value) for value in sample["intent_card"].get("soft_preferences", [])],
    ]
    matches = [
        value for value in constraints
        if value not in disclosed and (attribute == "other" or classify_constraint(value) == attribute)
    ][:2]
    if not matches:
        return f"I don't have an additional preference for {attribute}.", boundary_used
    disclosed.update(matches)
    return "For that, what matters is: " + "; ".join(matches) + ".", boundary_used


def metric_summary(sessions: list[dict]) -> dict:
    if not sessions:
        return {"sample_count": 0, "hit_rate_at_10": 0.0, "mrr": 0.0, "mttc": None}
    hit_rate = sum(int(item["hit"]) for item in sessions) / len(sessions)
    mrr = statistics.fmean(item["reciprocal_rank"] for item in sessions)
    mttc = statistics.fmean(
        item["first_hit_turn"] if item["first_hit_turn"] is not None else MAX_TURNS + 1 for item in sessions
    )
    return {
        "sample_count": len(sessions),
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
    }


def materialize_hidden_fields(sample: dict, products: dict[str, dict]) -> tuple[dict, dict]:
    if "intent_card" in sample and "behavior" in sample:
        return sample["intent_card"], sample["behavior"]
    target = str(sample["ground_truth"]["parent_asin"])
    product = products[target]
    card = intent_card(product)
    seed_source = f"{sample.get('sample_id', '')}\0{sample.get('scenario_type', '')}"
    rng = random.Random(seed_source)
    behavior = behavior_for(str(sample["scenario_type"]), card, rng)
    return card, behavior


class EvaluationSession:
    """Run one public-evaluator session, one turn at a time.

    The batch evaluator and the local browser demo both use this state machine so
    their simulator, override, normalization, and stopping semantics cannot drift.
    Target-related fields remain private to the state machine until a caller
    explicitly chooses to reveal them after the session has ended.
    """

    def __init__(
        self,
        agent: Agent,
        sample: dict,
        catalog_ids: set[str],
        categories: dict[str, list[str]],
        products: dict[str, dict],
        *,
        session_id: str | None = None,
    ) -> None:
        self.agent = agent
        self.sample = sample
        self.catalog_ids = catalog_ids
        self.categories = categories
        self.products = products
        self.session_id = session_id or f"public_{uuid.uuid4().hex}"

        # Keep the initialization order identical to evaluate()'s original loop.
        self.agent.reset(self.session_id, sample["user_profile"])
        self.target = str(sample["ground_truth"]["parent_asin"])
        effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
        self.effective_sample = {
            **sample,
            "intent_card": effective_intent_card,
            "behavior": effective_behavior,
        }
        self.disclosed: set[str] = set()
        self.boundary_used = False
        self.override_applied = sample["scenario_type"] != "intent_override"
        self.user_message = initial_message(
            self.effective_sample,
            coarse_category(categories.get(self.target, [])),
            self.disclosed,
        )
        self.user_message_source = "initial"
        self.turn = 1
        self.hit_turn: int | None = None
        self.best_rank: int | None = None
        self.done = False
        self.termination_reason: str | None = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.turns: list[dict] = []

    @staticmethod
    def _display_recommendations(response: dict, ranked: list[str]) -> list[dict]:
        """Retain optional Agent scores without changing official normalization."""
        raw = response.get("recommendations")
        scores: dict[str, int | float] = {}
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                parent_asin = str(item.get("parent_asin", "")).strip()
                score = item.get("score")
                if (
                    parent_asin not in scores
                    and isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    and math.isfinite(float(score))
                ):
                    scores[parent_asin] = score
        return [
            {
                "parent_asin": parent_asin,
                **({"score": scores[parent_asin]} if parent_asin in scores else {}),
            }
            for parent_asin in ranked
        ]

    def step(self) -> dict:
        if self.done:
            raise RuntimeError("evaluation session has already ended")

        turn = self.turn
        user_message = self.user_message
        user_message_source = self.user_message_source
        boundary_used_before = self.boundary_used
        override_applied_before = self.override_applied
        disclosed_before = sorted(self.disclosed)
        degraded_reason: str | None = None

        try:
            response = self.agent.respond(self.session_id, user_message, turn, TOP_K)
        except Exception as exc:
            degraded_reason = f"Agent raised {type(exc).__name__}; evaluator used an empty response."
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        if not isinstance(response, dict) or not isinstance(response.get("message"), str):
            degraded_reason = "Agent returned an invalid top-level response; evaluator used an empty response."
            response = {"message": "", "ask_attribute": None, "recommendations": []}

        usage = response.get("usage")
        turn_prompt_tokens = 0
        turn_completion_tokens = 0
        if isinstance(usage, dict):
            if isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] >= 0:
                turn_prompt_tokens = usage["prompt_tokens"]
                self.prompt_tokens += turn_prompt_tokens
            if isinstance(usage.get("completion_tokens"), int) and usage["completion_tokens"] >= 0:
                turn_completion_tokens = usage["completion_tokens"]
                self.completion_tokens += turn_completion_tokens

        ranked = normalize_recommendations(response.get("recommendations"), self.catalog_ids)
        target_rank = ranked.index(self.target) + 1 if self.target in ranked else None
        hit_rank = target_rank if self.override_applied else None
        transition: str
        next_user_message: str | None = None
        next_user_message_source: str | None = None

        if hit_rank is not None:
            self.best_rank = hit_rank
            self.hit_turn = turn
            self.done = True
            self.termination_reason = "hit"
            transition = "hit"
        elif turn == MAX_TURNS:
            self.done = True
            self.termination_reason = "max_turns"
            transition = "max_turns"
        else:
            override = self.effective_sample.get("behavior", {}).get("override") or {}
            if not self.override_applied and turn + 1 == int(override.get("turn", 3)):
                self.override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    self.disclosed.add(new_value)
                next_user_message = str(
                    override.get("message", "Actually, please ignore my earlier preference.")
                )
                next_user_message_source = "intent_override"
                transition = "override"
            else:
                next_user_message, self.boundary_used = customer_reply(
                    self.effective_sample,
                    response.get("ask_attribute"),
                    self.disclosed,
                    self.boundary_used,
                )
                next_user_message_source = "simulator_reply"
                transition = "customer_reply"
            self.turn += 1
            self.user_message = next_user_message
            self.user_message_source = next_user_message_source

        event = {
            "turn": turn,
            "user_message": user_message,
            "user_message_source": user_message_source,
            "request_payload": {
                "session_id": self.session_id,
                "user_message": user_message,
                "turn": turn,
                "top_k": TOP_K,
            },
            "message": response["message"],
            "ask_attribute": (
                response.get("ask_attribute")
                if isinstance(response.get("ask_attribute"), str)
                else None
            ),
            "recommendations": self._display_recommendations(response, ranked),
            "ranked_ids": ranked,
            "target_was_eligible": override_applied_before,
            "target_rank": target_rank,
            "hit_rank": hit_rank,
            "transition": transition,
            "next_user_message": next_user_message,
            "next_user_message_source": next_user_message_source,
            "boundary_used_before": boundary_used_before,
            "boundary_used_after": self.boundary_used,
            "override_applied_before": override_applied_before,
            "override_applied_after": self.override_applied,
            "disclosed_before": disclosed_before,
            "disclosed_after": sorted(self.disclosed),
            "usage": {
                "prompt_tokens": turn_prompt_tokens,
                "completion_tokens": turn_completion_tokens,
                "total_tokens": turn_prompt_tokens + turn_completion_tokens,
            },
            "degraded": degraded_reason is not None,
            "degraded_reason": degraded_reason,
        }
        self.turns.append(event)
        return event

    def result(self) -> dict:
        return {
            "sample_id": self.sample["sample_id"],
            "scenario_type": self.sample["scenario_type"],
            "hit": self.hit_turn is not None,
            "first_hit_turn": self.hit_turn,
            "best_rank": self.best_rank,
            "reciprocal_rank": 0.0 if self.best_rank is None else 1.0 / self.best_rank,
        }


def evaluate(
    agent: Agent,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> dict:
    sessions: list[dict] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for sample in samples:
        session = EvaluationSession(agent, sample, catalog_ids, categories, products)
        while not session.done:
            session.step()
        total_prompt_tokens += session.prompt_tokens
        total_completion_tokens += session.completion_tokens
        sessions.append(session.result())

    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical_score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "reported_token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
        "scenario_metrics": {name: metric_summary(grouped[name]) for name in sorted(grouped)},
        "sessions": sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TechJam public-set local evaluator")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    result = evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
