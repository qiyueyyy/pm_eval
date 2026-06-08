from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    case_id: str
    user_query: str
    scenario_type: str
    expected_behavior: str
    constraints_json: str
    difficulty: str
    tags: str
    template_id: str = ""


@dataclass
class TargetResult:
    success: bool
    response_text: str
    latency_ms: float
    error: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleScore:
    score: int
    details: dict[str, Any]


@dataclass
class JudgeScore:
    scores: dict[str, float]
    average_score: float
    comment: str = ""
    error: str = ""
    is_bad_case: bool | None = None
    bad_case_type: str = ""
    root_cause: str = ""
    improvement_suggestion: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricScore:
    intent_accuracy: float | None = None
    answer_relevance_score: float | None = None
    task_completion: float | None = None
    multi_turn_completion: float | None = None
    hallucination: bool | None = None
    retrieval_recall: float | None = None
    tool_success_rate: float | None = None
    expected_tool_coverage: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    batch_id: str
    case: EvalCase
    target: TargetResult
    rule_score: RuleScore
    judge_score: JudgeScore | None
    is_bad_case: bool
    bad_case_type: str
    root_cause: str
    improvement_suggestion: str
    metric_score: MetricScore | None = None
