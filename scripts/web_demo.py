from __future__ import annotations

import argparse
import json
import math
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from evaluator.local_evaluator import (
    MAX_TURNS,
    EvaluationSession,
    catalog_index,
    load_jsonl,
)
from starter.agent import Agent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data" / "catalog.jsonl"
DEFAULT_DATASET = ROOT / "data" / "public_set.jsonl"
INDEX_HTML = ROOT / "demo" / "index.html"
MAX_BODY_BYTES = 64 * 1024


class DemoAPIError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _finite_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


class DemoApplication:
    """Adapter from the shared evaluator state machine to a target-safe UI API."""

    def __init__(
        self,
        agent: Agent,
        samples: list[dict],
        catalog_ids: set[str],
        categories: dict[str, list[str]],
        products: dict[str, dict],
    ) -> None:
        self.agent = agent
        self.samples = samples
        self.samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
        self.catalog_ids = catalog_ids
        self.categories = categories
        self.products = products
        self.sessions: dict[str, dict[str, Any]] = {}

    def sample_options(self) -> list[dict]:
        return [
            {
                "sample_id": str(sample["sample_id"]),
                "scenario_type": str(sample["scenario_type"]),
                "difficulty_bucket": str(sample.get("difficulty_bucket", "unknown")),
                "category_bucket": str(sample.get("category_bucket", "unknown")),
            }
            for sample in self.samples
        ]

    def create_session(self, sample_id: object) -> dict:
        if not isinstance(sample_id, str) or sample_id not in self.samples_by_id:
            raise DemoAPIError(HTTPStatus.BAD_REQUEST, "Unknown public sample_id.")
        token = uuid.uuid4().hex
        engine = EvaluationSession(
            self.agent,
            self.samples_by_id[sample_id],
            self.catalog_ids,
            self.categories,
            self.products,
            session_id=f"demo_{uuid.uuid4().hex}",
        )
        self.sessions[token] = {"engine": engine, "revealed": False}
        return self.public_state(token)

    def step_session(self, token: object) -> dict:
        wrapper = self._session(token)
        engine: EvaluationSession = wrapper["engine"]
        if engine.done:
            raise DemoAPIError(HTTPStatus.CONFLICT, "This evaluator session has already ended.")
        engine.step()
        return self.public_state(str(token))

    def reveal_session(self, token: object) -> dict:
        wrapper = self._session(token)
        engine: EvaluationSession = wrapper["engine"]
        if not engine.done:
            raise DemoAPIError(
                HTTPStatus.CONFLICT,
                "The target stays sealed until the evaluator session ends.",
            )
        wrapper["revealed"] = True
        return self.public_state(str(token))

    def _session(self, token: object) -> dict[str, Any]:
        if not isinstance(token, str) or token not in self.sessions:
            raise DemoAPIError(HTTPStatus.NOT_FOUND, "Unknown or expired demo session.")
        return self.sessions[token]

    def _product(self, parent_asin: str) -> dict:
        product = self.products.get(parent_asin, {})
        categories = [str(value) for value in product.get("categories") or []]
        features = [str(value) for value in product.get("features") or []]
        return {
            "parent_asin": parent_asin,
            "title": str(product.get("title") or "Untitled catalog product"),
            "store": str(product.get("store") or ""),
            "price": _finite_number(product.get("price")),
            "average_rating": _finite_number(product.get("average_rating")),
            "rating_number": _finite_number(product.get("rating_number")),
            "category": " · ".join(categories[-2:]),
            "feature_preview": features[0] if features else "",
        }

    def public_state(self, token: str) -> dict:
        wrapper = self._session(token)
        engine: EvaluationSession = wrapper["engine"]
        revealed = bool(wrapper["revealed"])
        sample = engine.sample
        seen: set[str] = set()
        turns: list[dict] = []

        for event in engine.turns:
            recommendations: list[dict] = []
            for rank, recommendation in enumerate(event["recommendations"], start=1):
                parent_asin = recommendation["parent_asin"]
                item = {
                    "rank": rank,
                    "product": self._product(parent_asin),
                    "score": _finite_number(recommendation.get("score")),
                    "repeated": parent_asin in seen,
                }
                if revealed:
                    is_target = parent_asin == engine.target
                    item["is_target"] = is_target
                    if is_target:
                        item["target_status"] = (
                            "eligible_hit" if event["target_was_eligible"] else "pre_override_unscored"
                        )
                recommendations.append(item)
                seen.add(parent_asin)

            turn = {
                "turn": event["turn"],
                "user": {
                    "message": event["user_message"],
                    "source": event["user_message_source"],
                },
                "agent": {
                    "message": event["message"],
                    "ask_attribute": event["ask_attribute"],
                },
                "recommendations": recommendations,
                "normalized_count": len(event["ranked_ids"]),
                "scoring_active": event["target_was_eligible"],
                "transition": event["transition"],
                "next_user": (
                    {
                        "message": event["next_user_message"],
                        "source": event["next_user_message_source"],
                    }
                    if event["next_user_message"] is not None
                    else None
                ),
                "boundary_consumed": (
                    not event["boundary_used_before"] and event["boundary_used_after"]
                ),
                "override_activated": (
                    not event["override_applied_before"] and event["override_applied_after"]
                ),
                "disclosed": event["disclosed_after"],
                "usage": event["usage"],
                "degraded": event["degraded"],
                "degraded_reason": event["degraded_reason"],
                "request_payload": event["request_payload"],
            }
            if revealed:
                turn["target_rank"] = event["target_rank"]
                turn["target_was_eligible"] = event["target_was_eligible"]
            turns.append(turn)

        result = engine.result()
        state = {
            "session_token": token,
            "sample": {
                "sample_id": str(sample["sample_id"]),
                "scenario_type": str(sample["scenario_type"]),
                "difficulty_bucket": str(sample.get("difficulty_bucket", "unknown")),
                "category_bucket": str(sample.get("category_bucket", "unknown")),
            },
            "profile": sample["user_profile"],
            "status": {
                "done": engine.done,
                "completed_turns": len(engine.turns),
                "next_turn": None if engine.done else engine.turn,
                "max_turns": MAX_TURNS,
                "termination_reason": engine.termination_reason,
                "hit": result["hit"] if engine.done else None,
                "first_hit_turn": result["first_hit_turn"] if engine.done else None,
                "hit_rank": result["best_rank"] if engine.done else None,
                "reciprocal_rank": result["reciprocal_rank"] if engine.done else None,
                "usage": {
                    "prompt_tokens": engine.prompt_tokens,
                    "completion_tokens": engine.completion_tokens,
                    "total_tokens": engine.prompt_tokens + engine.completion_tokens,
                },
            },
            "pending_user": (
                {
                    "turn": engine.turn,
                    "message": engine.user_message,
                    "source": engine.user_message_source,
                }
                if not engine.done
                else None
            ),
            "turns": turns,
            "revealed": revealed,
            "reveal_available": engine.done and not revealed,
        }
        if revealed:
            state["target"] = self._product(engine.target)
            state["hidden_simulator_state"] = {
                "intent_card": engine.effective_sample["intent_card"],
                "behavior": engine.effective_sample["behavior"],
            }
        return state

    def close(self) -> None:
        connection = getattr(self.agent, "connection", None)
        if connection is not None:
            connection.close()


def make_handler(application: DemoApplication, index_html: bytes) -> type[BaseHTTPRequestHandler]:
    class DemoRequestHandler(BaseHTTPRequestHandler):
        server_version = "EvaluatorDemo/1.0"
        sys_version = ""

        def _common_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'",
            )

        def _send_json(self, status: HTTPStatus, payload: object) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self._common_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self) -> None:
            self.send_response(HTTPStatus.OK)
            self._common_headers()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(index_html)))
            self.end_headers()
            self.wfile.write(index_html)

        def _read_json(self) -> dict:
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                raise DemoAPIError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json.")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise DemoAPIError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length.") from exc
            if length < 0 or length > MAX_BODY_BYTES:
                raise DemoAPIError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "JSON request body is too large.")
            try:
                payload = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise DemoAPIError(HTTPStatus.BAD_REQUEST, "Invalid JSON request body.") from exc
            if not isinstance(payload, dict):
                raise DemoAPIError(HTTPStatus.BAD_REQUEST, "JSON request body must be an object.")
            return payload

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlsplit(self.path).path
            if path == "/":
                self._send_html()
            elif path == "/healthz":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ready": True,
                        "model": getattr(application.agent, "model_name", "unknown"),
                        "device": getattr(application.agent, "device", "unknown"),
                        "sample_count": len(application.samples),
                    },
                )
            elif path == "/api/samples":
                self._send_json(HTTPStatus.OK, {"samples": application.sample_options()})
            elif path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self._common_headers()
                self.end_headers()
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlsplit(self.path).path
            try:
                if path not in {"/api/session", "/api/step", "/api/reveal"}:
                    raise DemoAPIError(HTTPStatus.NOT_FOUND, "Not found.")
                payload = self._read_json()
                if path == "/api/session":
                    result = application.create_session(payload.get("sample_id"))
                    status = HTTPStatus.CREATED
                elif path == "/api/step":
                    result = application.step_session(payload.get("session_token"))
                    status = HTTPStatus.OK
                else:
                    result = application.reveal_session(payload.get("session_token"))
                    status = HTTPStatus.OK
                self._send_json(status, result)
            except DemoAPIError as exc:
                self._send_json(exc.status, {"error": exc.message})
            except Exception:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "The local demo encountered an unexpected server error."},
                )

        def log_message(self, format: str, *args: object) -> None:
            # Do not persist or print request bodies/user messages.
            return

    return DemoRequestHandler


def create_server(
    application: DemoApplication,
    host: str,
    port: int,
    *,
    index_path: str | Path = INDEX_HTML,
) -> HTTPServer:
    index_html = Path(index_path).read_bytes()
    return HTTPServer((host, port), make_handler(application, index_html))


def main() -> None:
    parser = argparse.ArgumentParser(description="Local single-session evaluator journey demo")
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    args = parser.parse_args()

    print("[demo] loading public samples and catalog index...", flush=True)
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)
    application = DemoApplication(agent, samples, catalog_ids, categories, products)
    server = create_server(application, args.host, args.port)
    print(f"[demo] ready: http://{args.host}:{server.server_port}/", flush=True)
    print("[demo] local demonstration server only; press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] stopping...", flush=True)
    finally:
        server.server_close()
        application.close()


if __name__ == "__main__":
    main()
