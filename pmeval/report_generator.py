from collections import Counter

from pmeval.models import EvalResult


def generate_report(results: list[EvalResult], target_name: str = "目标服务") -> str:
    if not results:
        return "# PM-Eval 评测报告\n\n暂无评测结果。"

    total = len(results)
    success = sum(1 for r in results if r.target.success)
    bad = sum(1 for r in results if r.is_bad_case)
    avg_rule = sum(r.rule_score.score for r in results) / total
    judge_values = [r.judge_score.average_score for r in results if r.judge_score and not r.judge_score.error]
    avg_judge = sum(judge_values) / len(judge_values) if judge_values else None
    avg_latency = sum(r.target.latency_ms for r in results) / total
    bad_results = [r for r in results if r.is_bad_case]

    type_counter = Counter(r.bad_case_type for r in bad_results)
    cause_counter = Counter(r.root_cause for r in bad_results)
    severity_counter = Counter(_severity(r) for r in bad_results)

    lines = [
        "# PM-Eval 评测报告",
        "",
        "## 本轮评测概况",
        "",
        f"- Batch ID: `{results[0].batch_id}`",
        f"- 本轮覆盖 {total} 条 {target_name} 文本推荐 case，聚焦预算、品类、风险提示、回答可用性与事实可信度。",
        f"- 共发现 {bad} 条 Bad Case，Bad Case 率为 {bad / total:.1%}。",
        f"- Judge 状态: {'已启用' if judge_values else '未启用或调用失败，仅使用规则评分'}。",
        "",
        "## 核心指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 总 case 数 | {total} |",
        f"| 成功率 | {success / total:.1%} |",
        f"| Bad Case 率 | {bad / total:.1%} |",
        f"| 平均 rule_score | {avg_rule:.1f} |",
        f"| 平均 judge_score | {avg_judge:.2f} |" if avg_judge is not None else "| 平均 judge_score | N/A |",
        f"| 平均响应时间 | {avg_latency:.0f} ms |",
        "",
        "## Top Bad Case 类型",
        "",
    ]

    lines.extend(_counter_lines(type_counter, "本轮未发现 Bad Case 类型集中问题。"))
    lines.extend(["", "## 主要问题归因", ""])
    lines.extend(_counter_lines(cause_counter, "本轮未发现明确根因集中问题。"))
    lines.extend(["", "## 严重度分布", ""])
    lines.extend(_counter_lines(severity_counter, "本轮无严重度分布。"))

    lines.extend(["", "## 迭代建议", ""])
    suggestions = []
    for result in bad_results:
        if result.improvement_suggestion and result.improvement_suggestion not in suggestions:
            suggestions.append(result.improvement_suggestion)
    if suggestions:
        for suggestion in suggestions[:8]:
            lines.append(f"- {suggestion}")
    else:
        lines.append("- 当前未发现高优先级迭代建议，建议扩充测试集并接入真实接口继续观察。")

    lines.extend(["", "## 下一版本优化优先级", ""])
    lines.extend(_priority_lines(type_counter, cause_counter, severity_counter))
    return "\n".join(lines)


def _counter_lines(counter: Counter, empty_text: str) -> list[str]:
    if not counter:
        return [f"- {empty_text}"]
    return [f"- {name}: {count}" for name, count in counter.most_common(5)]


def _severity(result: EvalResult) -> str:
    judge_avg = result.judge_score.average_score if result.judge_score and not result.judge_score.error else None
    if not result.target.success:
        return "P0"
    if result.rule_score.score < 50 or (judge_avg is not None and judge_avg < 2.5):
        return "P0"
    if result.rule_score.score < 70 or (judge_avg is not None and judge_avg < 3.5):
        return "P1"
    return "P2"


def _priority_lines(type_counter: Counter, cause_counter: Counter, severity_counter: Counter) -> list[str]:
    priorities = []
    if severity_counter.get("P0", 0):
        priorities.append("P0: 优先修复接口失败、严重约束不满足和高幻觉风险问题，确保评测链路可稳定复现。")
    if cause_counter.get("业务规则问题", 0) or type_counter.get("约束不满足", 0):
        priorities.append("P1: 增强预算、品类、禁用项等硬约束解析，并在输出前增加规则校验。")
    if cause_counter.get("知识库问题", 0) or type_counter.get("幻觉风险", 0):
        priorities.append("P1: 补充可信产品知识库与证据字段，降低无依据推荐和功效夸大。")
    if cause_counter.get("Prompt问题", 0) or type_counter.get("回答太泛", 0):
        priorities.append("P2: 固定推荐回答结构，要求包含推荐理由、适用边界和避雷提示。")
    if not priorities:
        priorities.append("P2: 扩充 case 覆盖面，加入更多真实用户 query、边界预算和敏感场景。")
    return [f"- {item}" for item in priorities]
