import re
from typing import Any

from pmeval.eval_template import EvalTemplate, load_template
from pmeval.models import EvalCase, RuleScore
from pmeval.utils import safe_json_loads


def evaluate_rules(
    case: EvalCase,
    response_text: str,
    success: bool,
    template: EvalTemplate | None = None,
) -> RuleScore:
    selected_template = template or load_template(None)
    constraints = safe_json_loads(case.constraints_json)
    text = response_text or ""
    details: dict[str, object] = {}

    if not success:
        return RuleScore(0, {"接口成功": {"passed": False, "weight": 100, "evidence": "target call failed"}})

    score = 0
    for check in selected_template.rule_checks:
        passed, evidence = _run_check(check.type, text, constraints, check.params)
        details[check.name] = {
            "id": check.id,
            "type": check.type,
            "passed": passed,
            "weight": check.weight,
            "evidence": evidence,
        }
        if passed:
            score += check.weight
    return RuleScore(min(100, int(score)), details)


def _run_check(check_type: str, text: str, constraints: dict[str, Any], params: dict[str, Any]) -> tuple[bool, str]:
    if check_type == "numeric_lte":
        return _check_numeric_lte(text, constraints, params)
    if check_type == "constraint_keywords":
        return _check_constraint_keywords(text, constraints, params)
    if check_type == "contains_any":
        return _check_contains_any(text, params)
    if check_type == "not_contains_any":
        return _check_not_contains_any(text, params)
    if check_type == "min_length_and_avoid_keywords":
        return _check_min_length_and_avoid_keywords(text, params)
    return False, f"unsupported_check_type={check_type}"


def _constraint_values(constraints: dict[str, Any], fields: list[str]) -> list[Any]:
    values: list[Any] = []
    for field in fields:
        value = constraints.get(field)
        if value is None or value == "":
            continue
        if isinstance(value, list):
            values.extend(value)
        else:
            values.append(value)
    return values


def _check_numeric_lte(text: str, constraints: dict[str, Any], params: dict[str, Any]) -> tuple[bool, str]:
    values = _constraint_values(constraints, params.get("constraint_fields", []))
    if not values:
        return bool(params.get("unset_pass", True)), "未设置数值约束"
    try:
        limit = float(values[0])
    except (TypeError, ValueError):
        return False, f"约束不是数字: {values[0]}"
    numbers = [float(num) for num in re.findall(r"(\d+(?:\.\d+)?)\s*(?:元|块|rmb|RMB)?", text)]
    if not numbers:
        return False, "回答未出现可校验数值"
    observed = max(numbers)
    return observed <= limit, f"max_observed={observed}, limit={limit}"


def _check_constraint_keywords(text: str, constraints: dict[str, Any], params: dict[str, Any]) -> tuple[bool, str]:
    keywords = [str(item) for item in _constraint_values(constraints, params.get("constraint_fields", [])) if str(item)]
    if not keywords:
        return bool(params.get("unset_pass", True)), "未设置关键词约束"
    lowered = text.lower()
    matched = [kw for kw in keywords if kw.lower() in lowered]
    return bool(matched), ",".join(matched)


def _check_contains_any(text: str, params: dict[str, Any]) -> tuple[bool, str]:
    keywords = [str(item) for item in params.get("keywords", [])]
    matched = [kw for kw in keywords if kw in text]
    return bool(matched), ",".join(matched)


def _check_not_contains_any(text: str, params: dict[str, Any]) -> tuple[bool, str]:
    keywords = [str(item) for item in params.get("keywords", [])]
    matched = [kw for kw in keywords if kw in text]
    return not matched, ",".join(matched)


def _check_min_length_and_avoid_keywords(text: str, params: dict[str, Any]) -> tuple[bool, str]:
    min_length = int(params.get("min_length", 0))
    if len(text.strip()) < min_length:
        return False, f"回答过短: {len(text.strip())} < {min_length}"
    avoid_keywords = [str(item) for item in params.get("avoid_keywords", [])]
    matched = [kw for kw in avoid_keywords if kw in text]
    return not matched, ",".join(matched)
