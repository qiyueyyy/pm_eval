from typing import Any

from pmeval.eval_template import EvalTemplate
from pmeval.models import EvalCase, JudgeScore, MetricScore, RuleScore, TargetResult
from pmeval.utils import safe_json_loads


def evaluate_metrics(
    case: EvalCase,
    target: TargetResult,
    rule_score: RuleScore,
    judge_score: JudgeScore | None,
    template: EvalTemplate | None = None,
) -> MetricScore:
    constraints = safe_json_loads(case.constraints_json)
    raw = target.raw_response if isinstance(target.raw_response, dict) else {}
    judge_raw = judge_score.raw_response if judge_score and isinstance(judge_score.raw_response, dict) else {}
    enabled_metrics = _enabled_metrics(template)

    expected_intent = _first_present(constraints, ["expected_intent", "intent"])
    predicted_intent = _first_present(raw, ["predicted_intent", "intent", "route"])
    if predicted_intent is None:
        predicted_intent = _first_present(raw.get("raw_response"), ["predicted_intent", "intent", "route"])

    intent_accuracy = _intent_accuracy(expected_intent, predicted_intent, judge_raw.get("intent_match"))
    answer_relevance = _judge_score(judge_score, judge_raw, "answer_relevance_score", ["relevance_score", "recall_relevance_score"])
    task_completion_score = _judge_score(judge_score, judge_raw, "task_completion_score", ["resolution_score", "usefulness_score"])
    multi_turn_score = _judge_score(judge_score, judge_raw, "multi_turn_completion_score", [])

    expected_retrieval_ids = _as_list(_first_present(constraints, ["expected_retrieval_ids", "expected_doc_ids", "expected_item_ids"]))
    retrieved_items = raw.get("retrieved_items") if isinstance(raw.get("retrieved_items"), list) else []
    retrieved_ids = _item_ids(retrieved_items)

    expected_tools = _as_list(_first_present(constraints, ["expected_tools", "must_call_tools"]))
    tool_calls = raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []

    has_turns = bool(_as_list(constraints.get("turns")))
    hallucination = _coerce_bool_or_none(judge_raw.get("hallucination"))
    if hallucination is None and judge_score and "faithfulness_score" in judge_score.scores:
        hallucination = judge_score.scores.get("faithfulness_score", 0) < 3

    metric = MetricScore(
        intent_accuracy=intent_accuracy,
        answer_relevance_score=answer_relevance,
        task_completion=_completion_value(target.success, rule_score.score, task_completion_score),
        multi_turn_completion=_completion_value(target.success, rule_score.score, multi_turn_score) if has_turns else None,
        hallucination=hallucination,
        retrieval_recall=_retrieval_recall(expected_retrieval_ids, retrieved_ids),
        tool_success_rate=_tool_success_rate(tool_calls, expected_tools),
        expected_tool_coverage=_expected_tool_coverage(expected_tools, tool_calls),
        details={
            "expected_intent": expected_intent,
            "predicted_intent": predicted_intent,
            "expected_retrieval_ids": expected_retrieval_ids,
            "retrieved_ids": sorted(retrieved_ids),
            "expected_tools": expected_tools,
            "called_tools": _called_tool_names(tool_calls),
            "hallucination_type": judge_raw.get("hallucination_type", ""),
            "template_id": template.id if template else getattr(case, "template_id", ""),
        },
    )
    _apply_enabled_metrics(metric, enabled_metrics)
    metric.details["metric_violations"] = metric_policy_violations(metric, template)
    return metric


def metric_policy_violations(metric: MetricScore | None, template: EvalTemplate | None) -> list[dict[str, Any]]:
    if not metric or not template:
        return []
    thresholds = getattr(template.metric_policy, "thresholds", {}) or {}
    violations = []

    for key, threshold in thresholds.items():
        attr = _metric_attr(key)
        value = getattr(metric, attr, None)
        if value is None:
            continue
        if attr == "hallucination":
            if bool(value):
                violations.append({"metric": key, "value": bool(value), "threshold": threshold, "direction": "must_be_false"})
            continue
        if float(value) < float(threshold):
            violations.append({"metric": key, "value": float(value), "threshold": float(threshold), "direction": "gte"})
    return violations


def _enabled_metrics(template: EvalTemplate | None) -> set[str]:
    if not template or not getattr(template, "metric_policy", None):
        return set()
    return {_metric_attr(item) for item in template.metric_policy.enabled_metrics}


def _apply_enabled_metrics(metric: MetricScore, enabled: set[str]) -> None:
    if not enabled:
        return
    all_attrs = {
        "intent_accuracy",
        "answer_relevance_score",
        "task_completion",
        "multi_turn_completion",
        "hallucination",
        "retrieval_recall",
        "tool_success_rate",
        "expected_tool_coverage",
    }
    for attr in all_attrs - enabled:
        setattr(metric, attr, None)


def _metric_attr(name: str) -> str:
    aliases = {
        "answer_relevance": "answer_relevance_score",
        "answer_relevance_score": "answer_relevance_score",
        "task_completion_rate": "task_completion",
        "task_completion": "task_completion",
        "multi_turn_completion_rate": "multi_turn_completion",
        "multi_turn_completion": "multi_turn_completion",
        "hallucination_rate": "hallucination",
        "hallucination": "hallucination",
        "tool_success": "tool_success_rate",
        "tool_success_rate": "tool_success_rate",
        "expected_tool_coverage": "expected_tool_coverage",
        "retrieval_recall": "retrieval_recall",
        "intent_accuracy": "intent_accuracy",
    }
    return aliases.get(str(name), str(name))


def _first_present(source: Any, keys: list[str]) -> Any:
    if not isinstance(source, dict):
        return None
    for key in keys:
        value = source.get(key)
        if value not in (None, ""):
            return value
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


def _intent_accuracy(expected: Any, predicted: Any, judge_match: Any) -> float | None:
    if expected not in (None, "") and predicted not in (None, ""):
        return 1.0 if str(expected).strip() == str(predicted).strip() else 0.0
    match = _coerce_bool_or_none(judge_match)
    if match is not None:
        return 1.0 if match else 0.0
    return None


def _judge_score(
    judge_score: JudgeScore | None,
    judge_raw: dict[str, Any],
    field: str,
    aliases: list[str],
) -> float | None:
    value = judge_raw.get(field)
    if value is None and judge_score:
        value = judge_score.scores.get(field)
    if value is None and judge_score:
        for alias in aliases:
            if alias in judge_score.scores:
                value = judge_score.scores[alias]
                break
    return _coerce_float_or_none(value)


def _completion_value(success: bool, rule_score: int, judge_value: float | None) -> float | None:
    if not success:
        return 0.0
    if judge_value is not None:
        return 1.0 if judge_value >= 3.5 else 0.0
    return 1.0 if rule_score >= 70 else 0.0


def _retrieval_recall(expected_ids: list[Any], retrieved_ids: set[str]) -> float | None:
    expected = {str(item) for item in expected_ids if str(item)}
    if not expected:
        return None
    return len(expected & retrieved_ids) / len(expected)


def _tool_success_rate(tool_calls: list[Any], expected_tools: list[Any]) -> float | None:
    if not tool_calls:
        return 0.0 if expected_tools else None
    successes = [_tool_success(call) for call in tool_calls]
    return sum(1 for item in successes if item) / len(successes)


def _expected_tool_coverage(expected_tools: list[Any], tool_calls: list[Any]) -> float | None:
    expected = {str(item) for item in expected_tools if str(item)}
    if not expected:
        return None
    called = set(_called_tool_names(tool_calls))
    return len(expected & called) / len(expected)


def _item_ids(items: list[Any]) -> set[str]:
    ids = set()
    for item in items:
        if isinstance(item, dict):
            value = item.get("id") or item.get("doc_id") or item.get("product_id") or item.get("item_id")
            if value not in (None, ""):
                ids.add(str(value))
        elif item not in (None, ""):
            ids.add(str(item))
    return ids


def _called_tool_names(tool_calls: list[Any]) -> list[str]:
    names = []
    for call in tool_calls:
        if isinstance(call, dict):
            name = call.get("name") or call.get("tool_name") or call.get("tool") or call.get("function")
            if name not in (None, ""):
                names.append(str(name))
        elif call not in (None, ""):
            names.append(str(call))
    return names


def _tool_success(call: Any) -> bool:
    if not isinstance(call, dict):
        return True
    value = call.get("success")
    if value is None:
        value = call.get("ok")
    if value is None:
        value = not bool(call.get("error"))
    return bool(value)


def _coerce_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "是"}:
            return True
        if lowered in {"false", "0", "no", "n", "否"}:
            return False
    if value is None:
        return None
    return bool(value)
