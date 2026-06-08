import json
import os
import time
from typing import Any
from urllib.parse import urlparse

import requests

from pmeval.models import EvalCase, TargetResult
from pmeval.utils import safe_json_loads


class MockTargetClient:
    def __init__(self, api_url: str = "", timeout: int = 30, prompt_name: str = "", prompt_text: str = ""):
        self.api_url = api_url
        self.timeout = timeout
        self.prompt_name = prompt_name
        self.prompt_text = prompt_text

    def recommend(self, case: EvalCase) -> TargetResult:
        start = time.perf_counter()
        constraints = safe_json_loads(case.constraints_json)
        budget = constraints.get("budget")
        categories = (
            constraints.get("categories")
            or constraints.get("category_keywords")
            or constraints.get("category")
            or constraints.get("topics")
            or constraints.get("interests")
            or ["通用方案"]
        )
        category_text = "、".join(categories) if isinstance(categories, list) else str(categories)
        avoid = constraints.get("avoid") or constraints.get("risk_keywords") or constraints.get("must_avoid") or []
        avoid_text = "、".join(avoid) if isinstance(avoid, list) else str(avoid)
        budget_text = f"预算控制在 {budget} 元以内，" if budget else ""
        prompt_hint = f"当前 Prompt 版本为 {self.prompt_name}。" if self.prompt_name else ""
        response_text = (
            f"{prompt_hint}推荐优先选择 {category_text} 方向，{budget_text}"
            "理由是更贴合你的场景、偏好和使用频率。"
            f"建议先小范围试用或验证效果，避雷点包括 {avoid_text or '刺激性成分、过度承诺和无依据结论'}。"
        )
        latency_ms = (time.perf_counter() - start) * 1000
        normalized = {
            "answer": response_text,
            "raw_response": {
                "mock": True,
                "prompt_name": self.prompt_name,
                "prompt_text": self.prompt_text,
            },
            "retrieved_items": [],
            "tool_calls": [],
        }
        return TargetResult(True, response_text, latency_ms, raw_response=normalized)


class RealTargetClient:
    """Adapter for a real recommendation/chat endpoint."""

    def __init__(self, api_url: str | None = None, timeout: int = 30, prompt_name: str = "", prompt_text: str = ""):
        self.api_url = self._resolve_api_url(api_url)
        self.timeout = timeout
        self.prompt_name = prompt_name
        self.prompt_text = prompt_text

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
        constraints = safe_json_loads(case.constraints_json)
        if self.api_url.rstrip("/").endswith("/api/agent/chat"):
            response = requests.post(
                self.api_url,
                data={
                    "message": case.user_query,
                    "user_id": "pm_eval",
                    "use_ai": "true",
                    "turns": json_dumps_compact(constraints.get("turns", [])),
                    "messages": json_dumps_compact(constraints.get("messages", constraints.get("turns", []))),
                    "prompt_name": self.prompt_name,
                    "prompt": self.prompt_text,
                    "system_prompt": self.prompt_text,
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
                    "turns": json_dumps_compact(constraints.get("turns", [])),
                    "messages": json_dumps_compact(constraints.get("messages", constraints.get("turns", []))),
                    "prompt_name": self.prompt_name,
                    "prompt": self.prompt_text,
                    "system_prompt": self.prompt_text,
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
            "turns": constraints.get("turns", []),
            "messages": constraints.get("messages", constraints.get("turns", [])),
            "user_id": "pm_eval",
            "use_ai": True,
            "prompt_name": self.prompt_name,
            "prompt": self.prompt_text,
            "system_prompt": self.prompt_text,
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
        tool_calls = self._extract_tool_calls(data)
        return {
            "answer": str(answer),
            "predicted_intent": data.get("predicted_intent") or data.get("intent") or data.get("route") or "",
            "raw_response": data,
            "retrieved_items": retrieved_items,
            "tool_calls": tool_calls,
        }

    def _extract_retrieved_items(self, data: dict[str, Any]) -> list[Any]:
        items: list[Any] = []
        products = data.get("products")
        if isinstance(products, list):
            items.extend(self._normalize_retrieved_items(products, default_source="products"))
        matched = data.get("matched_products")
        if isinstance(matched, list):
            for group in matched:
                if isinstance(group, dict) and isinstance(group.get("products"), list):
                    items.extend(self._normalize_retrieved_items(group["products"], default_source="matched_products"))
        documents = data.get("retrieved_items") or data.get("documents") or data.get("docs")
        if isinstance(documents, list):
            items.extend(self._normalize_retrieved_items(documents, default_source="documents"))
        return items

    def _normalize_retrieved_items(self, items: list[Any], default_source: str = "") -> list[dict[str, Any]]:
        normalized = []
        for index, item in enumerate(items):
            if isinstance(item, dict):
                item_id = item.get("id") or item.get("doc_id") or item.get("product_id") or item.get("item_id") or item.get("sku_id")
                normalized.append(
                    {
                        **item,
                        "id": str(item_id) if item_id not in (None, "") else str(index),
                        "score": item.get("score") or item.get("similarity") or item.get("rank_score"),
                        "source": item.get("source") or item.get("index") or default_source,
                    }
                )
            else:
                normalized.append({"id": str(item), "score": None, "source": default_source, "text": str(item)})
        return normalized

    def _extract_tool_calls(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        raw_calls = data.get("tool_calls") or data.get("tool_results") or data.get("observations") or data.get("tools") or []
        if isinstance(raw_calls, dict):
            raw_calls = list(raw_calls.values())
        if not isinstance(raw_calls, list):
            return []
        calls = []
        for call in raw_calls:
            if isinstance(call, dict):
                name = call.get("name") or call.get("tool_name") or call.get("tool") or call.get("function")
                calls.append(
                    {
                        **call,
                        "name": str(name or ""),
                        "success": _coerce_tool_success(call),
                        "latency_ms": call.get("latency_ms") or call.get("duration_ms") or call.get("elapsed_ms"),
                        "error": str(call.get("error") or ""),
                    }
                )
            else:
                calls.append({"name": str(call), "success": True, "latency_ms": None, "error": ""})
        return calls


class TargetClient:
    def __init__(self, api_url: str, timeout: int = 30, mock_mode: bool = True, prompt_name: str = "", prompt_text: str = ""):
        self.client = (
            MockTargetClient(api_url, timeout, prompt_name, prompt_text)
            if mock_mode
            else RealTargetClient(api_url, timeout, prompt_name, prompt_text)
        )

    def recommend(self, case: EvalCase) -> TargetResult:
        return self.client.recommend(case)


def create_target_client(
    client_mode: str,
    api_url: str,
    timeout: int = 30,
    prompt_name: str = "",
    prompt_text: str = "",
):
    if client_mode == "real":
        return RealTargetClient(api_url, timeout, prompt_name, prompt_text)
    return MockTargetClient(api_url, timeout, prompt_name, prompt_text)


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value if value is not None else [], ensure_ascii=False, separators=(",", ":"))


def _coerce_tool_success(call: dict[str, Any]) -> bool:
    value = call.get("success")
    if value is None:
        value = call.get("ok")
    if value is None:
        value = call.get("status")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"success", "succeeded", "ok", "true", "1", "yes"}:
            return True
        if lowered in {"failed", "error", "false", "0", "no"}:
            return False
    if value is None:
        return not bool(call.get("error"))
    return bool(value)


# Backward-compatible aliases for old imports/config snippets.
MockBeautyAgentClient = MockTargetClient
RealBeautyAgentClient = RealTargetClient
BeautyAgentClient = TargetClient
create_beautyagent_client = create_target_client
