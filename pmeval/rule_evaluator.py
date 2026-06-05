import re

from pmeval.models import EvalCase, RuleScore
from pmeval.utils import safe_json_loads


VAGUE_PATTERNS = [
    "看个人",
    "因人而异",
    "都可以",
    "随便",
    "无法判断",
    "建议咨询专业人士",
]

REASON_PATTERNS = ["理由", "原因", "因为", "适合", "优势", "依据"]
RISK_PATTERNS = ["避雷", "风险", "注意", "不建议", "谨慎", "刺激", "过敏", "踩雷"]


def evaluate_rules(case: EvalCase, response_text: str, success: bool) -> RuleScore:
    constraints = safe_json_loads(case.constraints_json)
    text = response_text or ""
    details: dict[str, object] = {}

    if not success:
        return RuleScore(0, {"接口成功": False})

    checks = [
        _check_budget(text, constraints),
        _check_category(text, constraints),
        _check_reason(text),
        _check_risk_tip(text),
        _check_not_vague(text),
    ]
    score = 0
    for name, passed, weight, evidence in checks:
        details[name] = {"passed": passed, "weight": weight, "evidence": evidence}
        if passed:
            score += weight
    return RuleScore(min(100, int(score)), details)


def _check_budget(text: str, constraints: dict) -> tuple[str, bool, int, str]:
    budget = constraints.get("budget")
    if not budget:
        return ("预算约束", True, 20, "未设置预算")
    numbers = [float(num) for num in re.findall(r"(\d+(?:\.\d+)?)\s*(?:元|块|rmb|RMB)?", text)]
    if not numbers:
        return ("预算约束", False, 20, "回答未出现价格")
    passed = max(numbers) <= float(budget)
    return ("预算约束", passed, 20, f"max_price={max(numbers)}, budget={budget}")


def _check_category(text: str, constraints: dict) -> tuple[str, bool, int, str]:
    keywords = constraints.get("categories") or constraints.get("category_keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    if not keywords:
        return ("品类关键词", True, 20, "未设置品类关键词")
    matched = [kw for kw in keywords if str(kw).lower() in text.lower()]
    return ("品类关键词", bool(matched), 20, ",".join(matched))


def _check_reason(text: str) -> tuple[str, bool, int, str]:
    matched = [kw for kw in REASON_PATTERNS if kw in text]
    return ("推荐理由", bool(matched), 20, ",".join(matched))


def _check_risk_tip(text: str) -> tuple[str, bool, int, str]:
    matched = [kw for kw in RISK_PATTERNS if kw in text]
    return ("避雷/风险提示", bool(matched), 20, ",".join(matched))


def _check_not_vague(text: str) -> tuple[str, bool, int, str]:
    if len(text.strip()) < 40:
        return ("非空泛回答", False, 20, "回答过短")
    matched = [kw for kw in VAGUE_PATTERNS if kw in text]
    return ("非空泛回答", not matched, 20, ",".join(matched))
