import time
import os
from urllib.parse import urlparse
from typing import Any

import requests

from pmeval.models import EvalCase, TargetResult
from pmeval.utils import safe_json_loads


class MockTargetClient:
    def __init__(self, api_url: str = "", timeout: int = 30):
        self.api_url = api_url
        self.timeout = timeout

    def recommend(self, case: EvalCase) -> TargetResult:
        start = time.perf_counter()
        constraints = safe_json_loads(case.constraints_json)
        budget = constraints.get("budget")
        categories = constraints.get("categories") or constraints.get("category_keywords") or ["护肤"]
        category_text = "、".join(categories) if isinstance(categories, list) else str(categories)
        avoid = constraints.get("avoid") or constraints.get("risk_keywords") or []
        avoid_text = "、".join(avoid) if isinstance(avoid, list) else str(avoid)
        budget_text = f"预算控制在 {budget} 元以内，" if budget else ""
        response_text = (
            f"推荐优先选择{category_text}方向的产品，{budget_text}"
            f"理由是更贴合你的场景、肤质和使用频率。"
            f"建议先小样或局部试用，避雷点包括{avoid_text or '刺激性成分、过度叠加功效'}。"
        )
        latency_ms = (time.perf_counter() - start) * 1000
        normalized = {
            "answer": response_text,
            "raw_response": {"mock": True},
            "retrieved_items": [],
            "tool_calls": [],
        }
        return TargetResult(True, response_text, latency_ms, raw_response=normalized)


class RealTargetClient:
    """Adapter for a real recommendation/chat endpoint."""

    def __init__(self, api_url: str | None = None, timeout: int = 30):
        self.api_url = self._resolve_api_url(api_url)
        self.timeout = timeout

    def recommend(self, case: EvalCase) -> TargetResult:
        start = time.perf_counter()
        try:
            data = self._call(case)
            latency_ms = (time.perf_counter() - start) * 1000
            normalized = self._normalize_response(data)
            return TargetResult(
                success=True,
                response_text=normalized["answer"],
                latency_ms=latency_ms,
                raw_response=normalized,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return TargetResult(False, "", latency_ms, error=str(exc))

    def _resolve_api_url(self, api_url: str | None) -> str:
        raw = api_url or os.getenv("TARGET_API_URL") or os.getenv("BEAUTYAGENT_API_URL") or "http://localhost:8000/api/agent/chat"
        parsed = urlparse(raw)
        if parsed.path in {"", "/"}:
            return raw.rstrip("/") + "/api/agent/chat"
        return raw

    def _call(self, case: EvalCase) -> dict[str, Any]:
        if self.api_url.rstrip("/").endswith("/api/agent/chat"):
            response = requests.post(
                self.api_url,
                data={
                    "message": case.user_query,
                    "user_id": "pm_eval",
                    "use_ai": "true",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        payload = self._json_payload(case)
        response = requests.post(self.api_url, json=payload, timeout=self.timeout)
        if response.status_code in {400, 415, 422}:
            response = requests.post(
                self.api_url,
                data={
                    "message": case.user_query,
                    "query": case.user_query,
                    "user_query": case.user_query,
                    "case_id": case.case_id,
                    "use_ai": "true",
                },
                timeout=self.timeout,
            )
        response.raise_for_status()
        return response.json()

    def _json_payload(self, case: EvalCase) -> dict[str, Any]:
        constraints = safe_json_loads(case.constraints_json)
        return {
            "query": case.user_query,
            "user_query": case.user_query,
            "message": case.user_query,
            "input": case.user_query,
            "case_id": case.case_id,
            "scenario_type": case.scenario_type,
            "expected_behavior": case.expected_behavior,
            "constraints": constraints,
            "constraints_json": constraints,
            "user_id": "pm_eval",
            "use_ai": True,
        }

    def _normalize_response(self, data: dict[str, Any]) -> dict[str, Any]:
        answer = (
            data.get("answer")
            or data.get("assistant_message")
            or data.get("recommendation")
            or data.get("response")
            or data.get("text")
            or data.get("ai_summary")
            or ""
        )
        retrieved_items = self._extract_retrieved_items(data)
        tool_calls = data.get("tool_calls") or data.get("observations") or []
        return {
            "answer": str(answer),
            "raw_response": data,
            "retrieved_items": retrieved_items,
            "tool_calls": tool_calls if isinstance(tool_calls, list) else [],
        }

    def _extract_retrieved_items(self, data: dict[str, Any]) -> list[Any]:
        items: list[Any] = []
        products = data.get("products")
        if isinstance(products, list):
            items.extend(products)
        matched = data.get("matched_products")
        if isinstance(matched, list):
            for group in matched:
                if isinstance(group, dict) and isinstance(group.get("products"), list):
                    items.extend(group["products"])
        return items


class TargetClient:
    def __init__(self, api_url: str, timeout: int = 30, mock_mode: bool = True):
        self.client = MockTargetClient(api_url, timeout) if mock_mode else RealTargetClient(api_url, timeout)

    def recommend(self, case: EvalCase) -> TargetResult:
        return self.client.recommend(case)


def create_target_client(client_mode: str, api_url: str, timeout: int = 30):
    if client_mode == "real":
        return RealTargetClient(api_url, timeout)
    return MockTargetClient(api_url, timeout)


# Backward-compatible aliases for old imports/config snippets.
MockBeautyAgentClient = MockTargetClient
RealBeautyAgentClient = RealTargetClient
BeautyAgentClient = TargetClient
create_beautyagent_client = create_target_client
