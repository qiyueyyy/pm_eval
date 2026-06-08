from collections import Counter, defaultdict

import pandas as pd

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
        f"- 本轮覆盖 {total} 条 {target_name} case。",
        f"- 共发现 {bad} 条 Bad Case，Bad Case 率为 {bad / total:.1%}。",
        f"- Judge 状态：{'已启用' if judge_values else '未启用或调用失败，仅使用规则评分'}。",
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

    lines.extend(["", "## Product Metrics", ""])
    lines.extend(_metric_summary_lines(results))
    lines.extend([""])
    lines.extend(_counter_lines(type_counter, "本轮未发现 Bad Case 类型集中问题。"))
    lines.extend(["", "## 主要问题归因", ""])
    lines.extend(_counter_lines(cause_counter, "本轮未发现明确根因集中问题。"))
    lines.extend(["", "## 严重度分布", ""])
    lines.extend(_counter_lines(severity_counter, "本轮无严重度分布。"))

    lines.extend(["", "## 迭代建议", ""])
    lines.extend(["", "## Scenario / Template Summary", ""])
    lines.extend(_scenario_summary_lines(results))

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


def generate_compare_report(base_run: pd.Series, target_run: pd.Series, compare_df: pd.DataFrame) -> str:
    base_name = _run_name(base_run)
    target_name = _run_name(target_run)
    lines = [
        "# PM-Eval 对比报告",
        "",
        "## 对比对象",
        "",
        f"- 基准版本: `{base_run['batch_id']}` ({base_name})",
        f"- 对比版本: `{target_run['batch_id']}` ({target_name})",
        "",
        "## 核心指标变化",
        "",
        "| 指标 | 基准版本 | 对比版本 | 变化 |",
        "| --- | ---: | ---: | ---: |",
        _metric_row("成功率", base_run["success_rate"], target_run["success_rate"], "{:.1%}"),
        _metric_row("Bad Case 率", base_run["bad_case_rate"], target_run["bad_case_rate"], "{:.1%}"),
        _metric_row("平均 rule_score", base_run["avg_rule_score"], target_run["avg_rule_score"], "{:.1f}"),
        _metric_row("平均 judge_score", base_run.get("avg_judge_score", 0), target_run.get("avg_judge_score", 0), "{:.2f}"),
        _metric_row("平均响应时间", base_run["avg_latency_ms"], target_run["avg_latency_ms"], "{:.0f} ms"),
        "",
        "## Product Metrics 变化",
        "",
        "| 指标 | 基准版本 | 对比版本 | 变化 |",
        "| --- | ---: | ---: | ---: |",
        _metric_row("Intent Accuracy", base_run.get("avg_intent_accuracy", 0) or 0, target_run.get("avg_intent_accuracy", 0) or 0, "{:.1%}"),
        _metric_row("Answer Relevance", base_run.get("avg_answer_relevance_score", 0) or 0, target_run.get("avg_answer_relevance_score", 0) or 0, "{:.2f}"),
        _metric_row("Task Completion", base_run.get("avg_task_completion", 0) or 0, target_run.get("avg_task_completion", 0) or 0, "{:.1%}"),
        _metric_row("Multi-turn Completion", base_run.get("avg_multi_turn_completion", 0) or 0, target_run.get("avg_multi_turn_completion", 0) or 0, "{:.1%}"),
        _metric_row("Hallucination Rate", base_run.get("avg_hallucination_rate", 0) or 0, target_run.get("avg_hallucination_rate", 0) or 0, "{:.1%}"),
        _metric_row("Retrieval Recall", base_run.get("avg_retrieval_recall", 0) or 0, target_run.get("avg_retrieval_recall", 0) or 0, "{:.1%}"),
        _metric_row("Tool Success Rate", base_run.get("avg_tool_success_rate", 0) or 0, target_run.get("avg_tool_success_rate", 0) or 0, "{:.1%}"),
        "",
    ]

    if compare_df.empty:
        lines.append("未找到可对比的 case 明细。")
        return "\n".join(lines)

    status_counts = compare_df["change_status"].value_counts()
    lines.extend(
        [
            "## Bad Case 收敛",
            "",
            f"- 收敛: {int(status_counts.get('收敛', 0))}",
            f"- 新增问题: {int(status_counts.get('新增问题', 0))}",
            f"- 未收敛: {int(status_counts.get('未收敛', 0))}",
            f"- 稳定通过: {int(status_counts.get('稳定通过', 0))}",
            "",
            "## 重点 Case 变化",
            "",
        ]
    )
    focus = compare_df[compare_df["change_status"].isin(["新增问题", "未收敛", "收敛"])].head(20)
    if focus.empty:
        lines.append("- 没有需要重点关注的 case 变化。")
    else:
        lines.extend(["| case_id | 状态 | rule 变化 | judge 变化 | 类型变化 | 根因变化 |", "| --- | --- | ---: | ---: | --- | --- |"])
        for _, row in focus.iterrows():
            lines.append(
                "| {case_id} | {status} | {rule_delta:.1f} | {judge_delta:.2f} | {type_base} -> {type_target} | {cause_base} -> {cause_target} |".format(
                    case_id=row.get("case_id", ""),
                    status=row.get("change_status", ""),
                    rule_delta=row.get("rule_score_delta", 0),
                    judge_delta=row.get("judge_average_delta", 0),
                    type_base=row.get("bad_case_type_base", "") or "无",
                    type_target=row.get("bad_case_type_target", "") or "无",
                    cause_base=row.get("root_cause_base", "") or "无",
                    cause_target=row.get("root_cause_target", "") or "无",
                )
            )
    return "\n".join(lines)


def _scenario_summary_lines(results: list[EvalResult]) -> list[str]:
    buckets: dict[tuple[str, str], list[EvalResult]] = defaultdict(list)
    for result in results:
        buckets[(result.case.scenario_type, getattr(result.case, "template_id", ""))].append(result)
    if not buckets:
        return ["- No scenario data."]

    lines = [
        "| scenario_type | template_id | cases | success_rate | bad_case_rate | avg_rule_score | avg_judge_score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (scenario_type, template_id), items in sorted(buckets.items()):
        total = len(items)
        success_rate = sum(1 for item in items if item.target.success) / total if total else 0
        bad_rate = sum(1 for item in items if item.is_bad_case) / total if total else 0
        avg_rule = sum(item.rule_score.score for item in items) / total if total else 0
        judge_values = [item.judge_score.average_score for item in items if item.judge_score and not item.judge_score.error]
        avg_judge = sum(judge_values) / len(judge_values) if judge_values else None
        avg_judge_text = f"{avg_judge:.2f}" if avg_judge is not None else "N/A"
        lines.append(
            f"| {scenario_type or '-'} | {template_id or '-'} | {total} | {success_rate:.1%} | {bad_rate:.1%} | {avg_rule:.1f} | {avg_judge_text} |"
        )
    return lines


def _metric_summary_lines(results: list[EvalResult]) -> list[str]:
    specs = [
        ("intent_accuracy", "Intent Accuracy", True),
        ("answer_relevance_score", "Answer Relevance", False),
        ("task_completion", "Task Completion", True),
        ("multi_turn_completion", "Multi-turn Completion", True),
        ("hallucination", "Hallucination Rate", True),
        ("retrieval_recall", "Retrieval Recall", True),
        ("tool_success_rate", "Tool Success Rate", True),
        ("expected_tool_coverage", "Expected Tool Coverage", True),
    ]
    lines = [
        "| metric | value | covered_cases |",
        "| --- | ---: | ---: |",
    ]
    for attr, label, percent in specs:
        values = []
        for result in results:
            ms = getattr(result, "metric_score", None)
            if ms is not None:
                v = getattr(ms, attr, None)
                if v is not None:
                    values.append(v)
        if not values:
            lines.append(f"| {label} | N/A | 0 |")
            continue
        numeric_values = [float(value) for value in values]
        average = sum(numeric_values) / len(numeric_values)
        value_text = f"{average:.1%}" if percent else f"{average:.2f}"
        lines.append(f"| {label} | {value_text} | {len(values)} |")
    return lines


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
        priorities.append("P0: 优先修复接口失败、严重约束不满足和高风险问题，确保评测链路可稳定复现。")
    if any(name for name in type_counter if "约束" in name or "政策" in name):
        priorities.append("P1: 强化硬约束解析，并在输出前增加规则校验。")
    if any(name for name in cause_counter if "知识库" in name or "检索" in name):
        priorities.append("P1: 检查知识库、索引、召回和证据字段，降低无依据回答。")
    if cause_counter.get("Prompt问题", 0) or any(name for name in type_counter if "泛" in name or "解释" in name):
        priorities.append("P2: 固定回答结构，要求包含理由、边界条件和下一步动作。")
    if not priorities:
        priorities.append("P2: 扩充 case 覆盖面，加入更多真实 query、边界条件和敏感场景。")
    return [f"- {item}" for item in priorities]


def _metric_row(label: str, base: float, target: float, fmt: str) -> str:
    delta = target - base
    return f"| {label} | {fmt.format(base)} | {fmt.format(target)} | {fmt.format(delta)} |"


def _run_name(row: pd.Series) -> str:
    prompt_name = row.get("prompt_name") or "默认 Prompt"
    target_name = row.get("target_name") or "target"
    return f"{target_name} / {prompt_name}"
