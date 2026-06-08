from pmeval.eval_template import EvalTemplate, load_template
from pmeval.metric_evaluator import metric_policy_violations
from pmeval.models import EvalCase, JudgeScore, MetricScore, RuleScore, TargetResult


def classify_bad_case(
    case: EvalCase,
    target: TargetResult,
    rule_score: RuleScore,
    judge_score: JudgeScore | None,
    template: EvalTemplate | None = None,
    metric_score: MetricScore | None = None,
) -> tuple[bool, str, str, str]:
    selected_template = template or load_template(None)
    bad_case = selected_template.bad_case
    judge_avg = judge_score.average_score if judge_score and not judge_score.error else None
    metric_violations = metric_policy_violations(metric_score, selected_template)
    is_bad = (
        rule_score.score < bad_case.rule_score_lt
        or (judge_avg is not None and judge_avg < bad_case.judge_average_lt)
        or not target.success
        or bool(metric_violations)
    )
    if not is_bad:
        return (
            False,
            bad_case.no_problem_type,
            bad_case.default_root_cause,
            "当前样例表现正常，建议继续扩大覆盖集观察稳定性。",
        )

    if not target.success:
        return (
            True,
            _choice("接口失败", bad_case.types, bad_case.default_type),
            _choice("接口问题", bad_case.root_causes, bad_case.default_root_cause),
            f"检查目标接口地址、请求格式和服务状态。错误: {target.error}",
        )

    if judge_score and not judge_score.error and judge_score.is_bad_case:
        return (
            True,
            judge_score.bad_case_type or bad_case.default_type,
            judge_score.root_cause or bad_case.default_root_cause,
            judge_score.improvement_suggestion or "结合 Judge 低分维度优化回答质量。",
        )

    if metric_violations:
        names = ", ".join(str(item.get("metric")) for item in metric_violations[:3])
        return (
            True,
            _fallback_metric_type(metric_violations, bad_case),
            _choice("业务规则问题", bad_case.root_causes, bad_case.default_root_cause),
            f"优先修复产品指标未达标项：{names}。",
        )

    failed_checks = _failed_checks(rule_score.details)
    if failed_checks:
        failed_names = "、".join(failed_checks[:3])
        return (
            True,
            _fallback_type(failed_names, bad_case),
            bad_case.default_root_cause,
            f"优先修复规则未通过项：{failed_names}。",
        )

    return (
        True,
        bad_case.default_type,
        bad_case.default_root_cause,
        "检查召回、Prompt 和业务规则，定位低分维度对应的链路问题。",
    )


def _failed_checks(details: dict) -> list[str]:
    return [name for name, value in details.items() if isinstance(value, dict) and not value.get("passed", False)]


def _fallback_type(failed_names: str, template) -> str:
    if "约束" in failed_names or "关键词" in failed_names or "预算" in failed_names:
        return _choice("约束不满足", template.types, template.default_type)
    if "风险" in failed_names or "安全" in failed_names:
        return _choice("安全合规风险", template.types, template.default_type)
    if "空泛" in failed_names:
        return _choice("回答太泛", template.types, template.default_type)
    return template.default_type


def _fallback_metric_type(violations: list[dict], template) -> str:
    metrics = {str(item.get("metric", "")) for item in violations}
    if any("hallucination" in item for item in metrics):
        return _choice("幻觉风险", template.types, template.default_type)
    if any("retrieval" in item for item in metrics):
        return _choice("检索/知识库问题", template.types, template.default_type)
    if any("tool" in item for item in metrics):
        return _choice("工具调用问题", template.types, template.default_type)
    if any("intent" in item for item in metrics):
        return _choice("需求理解错误", template.types, template.default_type)
    if any("task" in item or "completion" in item for item in metrics):
        return _choice("未解决问题", template.types, template.default_type)
    return template.default_type


def _choice(value: str, choices: list[str], default: str) -> str:
    return value if value in choices else default
