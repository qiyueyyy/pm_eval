import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pmeval.config import ROOT_DIR


@dataclass
class RuleCheckTemplate:
    id: str
    name: str
    type: str
    weight: int
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeFieldTemplate:
    id: str
    name: str
    description: str


@dataclass
class BadCaseTemplate:
    rule_score_lt: int = 70
    judge_average_lt: float = 3.5
    types: list[str] = field(default_factory=list)
    root_causes: list[str] = field(default_factory=list)
    no_problem_type: str = "无问题"
    default_type: str = "回答太泛"
    default_root_cause: str = "模型问题"


@dataclass
class MetricPolicyTemplate:
    enabled_metrics: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalTemplate:
    id: str
    name: str
    description: str
    rule_checks: list[RuleCheckTemplate]
    judge_fields: list[JudgeFieldTemplate]
    bad_case: BadCaseTemplate
    metric_policy: MetricPolicyTemplate = field(default_factory=MetricPolicyTemplate)


def templates_dir() -> Path:
    return ROOT_DIR / "pmeval" / "templates"


def list_templates() -> list[EvalTemplate]:
    items = []
    for path in sorted(templates_dir().glob("*.json")):
        items.append(load_template(path.stem))
    return items


def load_template(template_id: str | None) -> EvalTemplate:
    selected = template_id or "product_recommendation"
    path = templates_dir() / f"{selected}.json"
    if not path.exists():
        path = templates_dir() / "product_recommendation.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return _parse_template(data)


def _parse_template(data: dict[str, Any]) -> EvalTemplate:
    bad_case = data.get("bad_case", {})
    metric_policy = data.get("metric_policy", {})
    return EvalTemplate(
        id=str(data["id"]),
        name=str(data["name"]),
        description=str(data.get("description", "")),
        rule_checks=[
            RuleCheckTemplate(
                id=str(item["id"]),
                name=str(item["name"]),
                type=str(item["type"]),
                weight=int(item.get("weight", 0)),
                params=dict(item.get("params", {})),
            )
            for item in data.get("rule_checks", [])
        ],
        judge_fields=[
            JudgeFieldTemplate(
                id=str(item["id"]),
                name=str(item["name"]),
                description=str(item.get("description", item["name"])),
            )
            for item in data.get("judge_fields", [])
        ],
        bad_case=BadCaseTemplate(
            rule_score_lt=int(bad_case.get("rule_score_lt", 70)),
            judge_average_lt=float(bad_case.get("judge_average_lt", 3.5)),
            types=[str(item) for item in bad_case.get("types", [])],
            root_causes=[str(item) for item in bad_case.get("root_causes", [])],
            no_problem_type=str(bad_case.get("no_problem_type", "无问题")),
            default_type=str(bad_case.get("default_type", "回答太泛")),
            default_root_cause=str(bad_case.get("default_root_cause", "模型问题")),
        ),
        metric_policy=MetricPolicyTemplate(
            enabled_metrics=[str(item) for item in metric_policy.get("enabled_metrics", [])],
            thresholds={str(key): float(value) for key, value in metric_policy.get("thresholds", {}).items()},
        ),
    )
