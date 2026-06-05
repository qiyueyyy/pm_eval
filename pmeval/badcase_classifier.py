from pmeval.models import EvalCase, JudgeScore, RuleScore, TargetResult


BAD_CASE_TYPES = [
    "需求理解错误",
    "约束不满足",
    "幻觉风险",
    "回答太泛",
    "格式错误",
    "检索/知识库问题",
    "接口失败",
    "无问题",
]

ROOT_CAUSES = [
    "Prompt问题",
    "知识库问题",
    "检索问题",
    "工具问题",
    "业务规则问题",
    "模型问题",
    "交互问题",
    "接口问题",
]


def classify_bad_case(
    case: EvalCase,
    target: TargetResult,
    rule_score: RuleScore,
    judge_score: JudgeScore | None,
) -> tuple[bool, str, str, str]:
    judge_avg = judge_score.average_score if judge_score and not judge_score.error else None
    is_bad = rule_score.score < 70 or (judge_avg is not None and judge_avg < 3.5) or not target.success
    if not is_bad:
        return False, "无问题", "模型问题", "当前样例表现正常，建议继续扩大覆盖集。"

    details = rule_score.details
    if not target.success:
        return True, "接口失败", "接口问题", f"检查目标接口地址、请求格式和服务状态。错误: {target.error}"
    if judge_score and not judge_score.error and judge_score.is_bad_case:
        return (
            True,
            judge_score.bad_case_type or "回答太泛",
            judge_score.root_cause or "模型问题",
            judge_score.improvement_suggestion or "结合 Judge 低分维度优化回答质量。",
        )
    if _failed(details, "预算约束") or _failed(details, "品类关键词"):
        return True, "约束不满足", "业务规则问题", "强化 Prompt 中的硬约束解析，并在输出前增加规则校验。"
    if _failed(details, "非空泛回答"):
        return True, "回答太泛", "Prompt问题", "要求输出具体产品/品类、适用原因、使用建议和限制条件。"
    if judge_score and judge_score.scores.get("faithfulness_score", 5) < 3.5:
        return True, "幻觉风险", "知识库问题", "补充可信产品知识库，并要求模型避免编造品牌、成分和功效。"
    if judge_score and judge_score.scores.get("need_understanding_score", 5) < 3.5:
        return True, "需求理解错误", "模型问题", "增加意图分类与关键信息抽取，必要时先追问再推荐。"
    if _failed(details, "推荐理由") or _failed(details, "避雷/风险提示"):
        return True, "格式错误", "Prompt问题", "固定回答结构，确保包含推荐理由和风险提示。"
    return True, "检索/知识库问题", "检索问题", "检查检索召回结果是否覆盖用户场景和约束。"


def _failed(details: dict, name: str) -> bool:
    value = details.get(name)
    return isinstance(value, dict) and not value.get("passed", False)
