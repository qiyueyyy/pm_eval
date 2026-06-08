import json
import threading
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from pmeval.config import ROOT_DIR, Settings, get_settings
from pmeval.db import EvalDatabase
from pmeval.eval_template import EvalTemplate, list_templates, load_template
from pmeval.report_generator import generate_compare_report, generate_report
from pmeval.runner import EvalRunner
from pmeval.scenario_router import known_template_ids, resolve_template_id
from pmeval.utils import dataframe_to_cases, ensure_dirs, read_cases_csv


st.set_page_config(page_title="PM-Eval v0.1", layout="wide")
ensure_dirs(ROOT_DIR)


CASE_COLUMNS = [
    "case_id",
    "user_query",
    "scenario_type",
    "template_id",
    "expected_behavior",
    "constraints_json",
    "difficulty",
    "tags",
]


def init_state() -> None:
    if "cases_df" not in st.session_state:
        st.session_state.cases_df = read_cases_csv(ROOT_DIR / "data" / "sample_cases_30.csv")
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()
    if "bad_cases_df" not in st.session_state:
        st.session_state.bad_cases_df = pd.DataFrame()
    if "report_md" not in st.session_state:
        st.session_state.report_md = ""
    if "compare_report_md" not in st.session_state:
        st.session_state.compare_report_md = ""
    if "active_template_id" not in st.session_state:
        st.session_state.active_template_id = get_settings().template_id
    if "eval_job" not in st.session_state:
        st.session_state.eval_job = None


def results_to_frames(results, template: EvalTemplate | None = None):
    rows = []
    bad_rows = []
    for item in results:
        judge = item.judge_score
        normalized = item.target.raw_response if isinstance(item.target.raw_response, dict) else {}
        retrieved_items = normalized.get("retrieved_items") or []
        tool_calls = normalized.get("tool_calls") or []
        row = {
            "batch_id": item.batch_id,
            "case_id": item.case.case_id,
            "user_query": item.case.user_query,
            "scenario_type": item.case.scenario_type,
            "template_id": item.case.template_id or (template.id if template else ""),
            "success": item.target.success,
            "latency_ms": round(item.target.latency_ms, 1),
            "rule_score": item.rule_score.score,
            "judge_average": judge.average_score if judge else None,
            "judge_is_bad_case": judge.is_bad_case if judge else None,
            "judge_error": judge.error if judge else "",
            "is_bad_case": item.is_bad_case,
            "bad_case_type": item.bad_case_type,
            "root_cause": item.root_cause,
            "response_text": item.target.response_text,
            "error": item.target.error,
            "retrieved_items_count": len(retrieved_items) if isinstance(retrieved_items, list) else 0,
            "tool_calls_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
            "improvement_suggestion": item.improvement_suggestion,
        }
        metric = item.metric_score
        if metric:
            row.update(
                {
                    "intent_accuracy": metric.intent_accuracy,
                    "answer_relevance_score": metric.answer_relevance_score,
                    "task_completion": metric.task_completion,
                    "multi_turn_completion": metric.multi_turn_completion,
                    "hallucination": metric.hallucination,
                    "retrieval_recall": metric.retrieval_recall,
                    "tool_success_rate": metric.tool_success_rate,
                    "expected_tool_coverage": metric.expected_tool_coverage,
                }
            )
        if judge:
            for field_id, score in judge.scores.items():
                row[f"judge_{field_id}"] = score
        row["severity"] = infer_severity(row) if item.is_bad_case else "正常"
        rows.append(row)
        if item.is_bad_case:
            bad_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(bad_rows)


def infer_severity(row: dict) -> str:
    if not row.get("success", True):
        return "P0"
    judge_average = row.get("judge_average")
    rule_score = row.get("rule_score") or 0
    if rule_score < 50 or (pd.notna(judge_average) and judge_average and judge_average < 2.5):
        return "P0"
    if rule_score < 70 or (pd.notna(judge_average) and judge_average and judge_average < 3.5):
        return "P1"
    return "P2"


def save_exports(results_df: pd.DataFrame, bad_cases_df: pd.DataFrame, report_md: str, compare_report_md: str = "") -> None:
    export_dir = ROOT_DIR / "exports"
    results_df.to_csv(export_dir / "results.csv", index=False, encoding="utf-8-sig")
    bad_cases_df.to_csv(export_dir / "bad_cases.csv", index=False, encoding="utf-8-sig")
    (export_dir / "report.md").write_text(report_md, encoding="utf-8")
    if compare_report_md:
        (export_dir / "compare_report.md").write_text(compare_report_md, encoding="utf-8")


def metric_value(df: pd.DataFrame, column: str, default=0):
    if df.empty or column not in df:
        return default
    return df[column].mean()


def metric_display(df: pd.DataFrame, column: str, percent: bool = True) -> str:
    if df.empty or column not in df:
        return "N/A"
    values = df[column].dropna()
    if values.empty:
        return "N/A"
    value = values.mean()
    return f"{value:.1%}" if percent else f"{value:.2f}"


def normalize_cases_df(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for col in CASE_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = ""
    normalized = normalized[CASE_COLUMNS].fillna("")
    normalized = normalized.astype(str)
    normalized["template_id"] = normalized.apply(
        lambda row: resolve_template_id(row["scenario_type"], row["template_id"]),
        axis=1,
    )
    return normalized.reset_index(drop=True)


def validate_cases_df(df: pd.DataFrame) -> list[str]:
    errors = []
    normalized = normalize_cases_df(df)
    if normalized["case_id"].str.strip().eq("").any():
        errors.append("case_id 不能为空。")
    duplicated = normalized["case_id"][normalized["case_id"].duplicated() & normalized["case_id"].str.strip().ne("")]
    if not duplicated.empty:
        errors.append(f"case_id 重复：{', '.join(sorted(duplicated.unique())[:10])}")
    if normalized["user_query"].str.strip().eq("").any():
        errors.append("user_query 不能为空。")
    valid_templates = known_template_ids()
    invalid_template_ids = sorted(
        {
            value
            for value in normalized["template_id"].str.strip().tolist()
            if value and value not in valid_templates
        }
    )
    if invalid_template_ids:
        errors.append(f"template_id 涓嶅湪宸茬煡妯℃澘涓細{', '.join(invalid_template_ids[:10])}")
    invalid_json_ids = []
    for _, row in normalized.iterrows():
        raw = row["constraints_json"].strip()
        if not raw:
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            invalid_json_ids.append(row["case_id"])
            continue
        if not isinstance(value, dict):
            invalid_json_ids.append(row["case_id"])
    if invalid_json_ids:
        errors.append(f"constraints_json 必须是 JSON 对象：{', '.join(invalid_json_ids[:10])}")
    return errors


def split_tags(raw: str) -> list[str]:
    text = str(raw or "").replace("，", ",").replace("、", ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def cases_health_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_cases_df(df)
    if normalized.empty:
        return pd.DataFrame()
    return (
        normalized.groupby(["scenario_type", "template_id"], dropna=False)
        .agg(
            case_count=("case_id", "count"),
            difficulty_count=("difficulty", "nunique"),
            tag_count=("tags", lambda values: len({tag for raw in values for tag in split_tags(raw)})),
        )
        .reset_index()
        .sort_values(["template_id", "scenario_type"])
    )


def scenario_result_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    results_df = results_df.copy()
    for col in [
        "intent_accuracy",
        "answer_relevance_score",
        "task_completion",
        "multi_turn_completion",
        "hallucination",
        "retrieval_recall",
        "tool_success_rate",
        "expected_tool_coverage",
    ]:
        if col not in results_df:
            results_df[col] = None
    grouped = (
        results_df.groupby(["scenario_type", "template_id"], dropna=False)
        .agg(
            case_count=("case_id", "count"),
            success_rate=("success", "mean"),
            bad_case_rate=("is_bad_case", "mean"),
            avg_rule_score=("rule_score", "mean"),
            avg_judge_score=("judge_average", "mean"),
            intent_accuracy=("intent_accuracy", "mean"),
            answer_relevance_score=("answer_relevance_score", "mean"),
            task_completion=("task_completion", "mean"),
            multi_turn_completion=("multi_turn_completion", "mean"),
            hallucination_rate=("hallucination", "mean"),
            retrieval_recall=("retrieval_recall", "mean"),
            tool_success_rate=("tool_success_rate", "mean"),
            expected_tool_coverage=("expected_tool_coverage", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(["bad_case_rate", "avg_rule_score"], ascending=[False, True])


def metric_summary_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    specs = [
        ("intent_accuracy", "Intent Accuracy"),
        ("answer_relevance_score", "Answer Relevance"),
        ("task_completion", "Task Completion"),
        ("multi_turn_completion", "Multi-turn Completion"),
        ("hallucination", "Hallucination Rate"),
        ("retrieval_recall", "Retrieval Recall"),
        ("tool_success_rate", "Tool Success Rate"),
        ("expected_tool_coverage", "Expected Tool Coverage"),
    ]
    rows = []
    for column, label in specs:
        if column not in results_df:
            continue
        values = results_df[column].dropna()
        rows.append(
            {
                "metric": label,
                "covered_cases": int(values.shape[0]),
                "value": values.mean() if not values.empty else None,
            }
        )
    return pd.DataFrame(rows)


def option_values(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df:
        return []
    if column == "tags":
        values = sorted({tag for raw in df[column].tolist() for tag in split_tags(raw)})
    else:
        values = sorted([str(item) for item in df[column].dropna().unique().tolist() if str(item)])
    return values


def filter_cases_df(df: pd.DataFrame, selected_tags: list[str], selected_difficulties: list[str], keyword: str) -> pd.DataFrame:
    filtered = df.copy()
    if selected_difficulties:
        filtered = filtered[filtered["difficulty"].isin(selected_difficulties)]
    if selected_tags:
        filtered = filtered[filtered["tags"].apply(lambda raw: bool(set(split_tags(raw)) & set(selected_tags)))]
    if keyword.strip():
        needle = keyword.strip().lower()
        mask = (
            filtered["case_id"].str.lower().str.contains(needle, na=False)
            | filtered["user_query"].str.lower().str.contains(needle, na=False)
            | filtered["expected_behavior"].str.lower().str.contains(needle, na=False)
            | filtered["tags"].str.lower().str.contains(needle, na=False)
        )
        filtered = filtered[mask]
    return filtered


def save_cases_csv(df: pd.DataFrame, filename: str) -> str:
    safe_name = "".join(ch for ch in filename.strip() if ch.isalnum() or ch in {"-", "_", "."})
    if not safe_name:
        safe_name = f"cases_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    if not safe_name.lower().endswith(".csv"):
        safe_name += ".csv"
    path = ROOT_DIR / "data" / safe_name
    normalize_cases_df(df).to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def render_case_manager() -> None:
    cases_df = normalize_cases_df(st.session_state.cases_df)
    st.session_state.cases_df = cases_df

    control_cols = st.columns([2, 2, 3])
    selected_difficulties = control_cols[0].multiselect(
        "难度筛选",
        option_values(cases_df, "difficulty"),
        key="case_filter_difficulty",
    )
    selected_tags = control_cols[1].multiselect(
        "标签筛选",
        option_values(cases_df, "tags"),
        key="case_filter_tags",
    )
    keyword = control_cols[2].text_input(
        "关键词",
        placeholder="case_id / query / 期望行为 / 标签",
        key="case_filter_keyword",
    )

    filtered = filter_cases_df(cases_df, selected_tags, selected_difficulties, keyword)
    st.caption(f"当前测试集：{len(cases_df)} 条；筛选结果：{len(filtered)} 条")

    health = cases_health_frame(cases_df)
    if not health.empty:
        with st.expander("Case set health check", expanded=True):
            st.dataframe(health, use_container_width=True, height=220)

    action_cols = st.columns([2, 2, 2, 2])
    with action_cols[0]:
        delete_ids = st.multiselect("删除 case", filtered["case_id"].tolist())
        if st.button("删除选中", disabled=not delete_ids):
            st.session_state.cases_df = cases_df[~cases_df["case_id"].isin(delete_ids)].reset_index(drop=True)
            st.success(f"已删除 {len(delete_ids)} 条 case")
            st.rerun()
    with action_cols[1]:
        save_name = st.text_input("保存文件名", value="edited_cases.csv")
        if st.button("保存到 data/"):
            errors = validate_cases_df(cases_df)
            if errors:
                st.error("保存失败：" + " ".join(errors))
            else:
                path = save_cases_csv(cases_df, save_name)
                st.success(f"已保存：{path}")
    with action_cols[2]:
        saved_files = sorted([path.name for path in (ROOT_DIR / "data").glob("*.csv")])
        selected_file = st.selectbox("加载本地测试集", saved_files, index=saved_files.index("sample_cases_30.csv") if "sample_cases_30.csv" in saved_files else 0)
        if st.button("加载选中文件"):
            st.session_state.cases_df = read_cases_csv(ROOT_DIR / "data" / selected_file)
            st.success(f"已加载：{selected_file}")
            st.rerun()
    with action_cols[3]:
        st.download_button(
            "下载当前测试集",
            normalize_cases_df(cases_df).to_csv(index=False).encode("utf-8-sig"),
            "cases.csv",
        )
        if st.button("重置为示例集"):
            st.session_state.cases_df = read_cases_csv(ROOT_DIR / "data" / "sample_cases_30.csv")
            st.rerun()

    with st.expander("新增 case", expanded=False):
        add_cols = st.columns(4)
        new_case_id = add_cols[0].text_input("case_id", key="new_case_id")
        new_difficulty = add_cols[1].text_input("difficulty", key="new_difficulty")
        new_tags = add_cols[2].text_input("tags", key="new_tags")
        new_template_id = add_cols[3].selectbox("template_id", [""] + template_ids, key="new_template_id")
        new_query = st.text_area("user_query", key="new_user_query", height=80)
        new_expected = st.text_area("expected_behavior", key="new_expected_behavior", height=80)
        new_scenario = st.text_input("scenario_type", key="new_scenario_type")
        new_constraints = st.text_area("constraints_json", value="{}", key="new_constraints_json", height=90)
        if st.button("添加 case"):
            new_row = {
                "case_id": new_case_id.strip(),
                "user_query": new_query.strip(),
                "scenario_type": new_scenario.strip(),
                "template_id": new_template_id.strip(),
                "expected_behavior": new_expected.strip(),
                "constraints_json": new_constraints.strip(),
                "difficulty": new_difficulty.strip(),
                "tags": new_tags.strip(),
            }
            candidate = pd.concat([cases_df, pd.DataFrame([new_row])], ignore_index=True)
            errors = validate_cases_df(candidate)
            if errors:
                st.error("添加失败：" + " ".join(errors))
            else:
                st.session_state.cases_df = normalize_cases_df(candidate)
                st.success("已添加 case")
                st.rerun()

    editable = filtered.copy()
    editable.insert(0, "_row_id", editable.index)
    edited_df = st.data_editor(
        editable,
        use_container_width=True,
        height=520,
        num_rows="fixed",
        disabled=["_row_id"],
        column_config={
            "_row_id": st.column_config.NumberColumn("行号", help="原始测试集行号，保存编辑时用于合并回完整测试集。"),
            "constraints_json": st.column_config.TextColumn("constraints_json", width="large"),
            "expected_behavior": st.column_config.TextColumn("expected_behavior", width="large"),
            "user_query": st.column_config.TextColumn("user_query", width="large"),
        },
        key="cases_editor",
    )
    editor_cols = st.columns([1, 1, 4])
    if editor_cols[0].button("应用表格编辑"):
        merged = cases_df.copy()
        edited_df = edited_df.copy()
        for _, row in edited_df.iterrows():
            row_id = int(row["_row_id"])
            for col in CASE_COLUMNS:
                merged.at[row_id, col] = str(row.get(col, ""))
        errors = validate_cases_df(merged)
        if errors:
            st.error("应用失败：" + " ".join(errors))
        else:
            st.session_state.cases_df = normalize_cases_df(merged)
            st.success("已应用表格编辑")
            st.rerun()
    if editor_cols[1].button("清空筛选"):
        st.session_state.case_filter_difficulty = []
        st.session_state.case_filter_tags = []
        st.session_state.case_filter_keyword = ""
        st.rerun()


def run_label(row) -> str:
    template_id = row.get("template_id") or "unknown"
    target_name = row.get("target_name") or "target"
    prompt_name = row.get("prompt_name") or "默认 Prompt"
    return f"{row['started_at']} | {target_name} | {prompt_name} | {template_id} | {row['batch_id']}"


def build_settings(
    base_settings: Settings,
    api_url: str,
    timeout: int,
    client_mode: str,
    openai_base_url: str,
    openai_model: str,
    target_name: str,
    template_id: str,
    prompt_name: str = "",
    prompt_text: str = "",
) -> Settings:
    return Settings(
        api_url=api_url,
        api_timeout=int(timeout),
        mock_mode=(client_mode == "mock"),
        client_mode=client_mode,
        openai_api_key=base_settings.openai_api_key,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
        db_path=base_settings.db_path,
        target_name=target_name,
        template_id=template_id,
        prompt_name=prompt_name,
        prompt_text=prompt_text,
    )


def run_eval_to_frames(settings: Settings, template: EvalTemplate, cases, use_judge: bool):
    results = EvalRunner(settings, template).run(cases, use_judge=use_judge)
    results_df, bad_cases_df = results_to_frames(results, template)
    report_md = generate_report(results, settings.target_name)
    return results, results_df, bad_cases_df, report_md


def start_eval_job(settings: Settings, template: EvalTemplate, cases, use_judge: bool) -> dict:
    job = {
        "status": "running",
        "completed": 0,
        "total": len(cases),
        "current_case_id": "",
        "batch_id": "",
        "started_at": "",
        "results": [],
        "results_df": pd.DataFrame(),
        "bad_cases_df": pd.DataFrame(),
        "report_md": "",
        "error": "",
        "stop_event": threading.Event(),
        "thread": None,
        "applied": False,
        "settings": settings,
        "template": template,
        "cases": cases,
        "use_judge": use_judge,
    }
    thread = threading.Thread(
        target=_eval_job_worker,
        args=(job, settings, template, cases, use_judge),
        daemon=True,
    )
    job["thread"] = thread
    thread.start()
    return job


def resume_eval_job(job: dict) -> dict:
    job["status"] = "running"
    job["stop_event"] = threading.Event()
    job["applied"] = False
    job["error"] = ""
    thread = threading.Thread(
        target=_eval_job_worker,
        args=(job, job["settings"], job["template"], job["cases"], job["use_judge"]),
        daemon=True,
    )
    job["thread"] = thread
    thread.start()
    return job


def _eval_job_worker(job: dict, settings: Settings, template: EvalTemplate, cases, use_judge: bool) -> None:
    def on_progress(completed, total, result, batch_id, started_at, stopped):
        job["completed"] = completed
        job["total"] = total
        job["batch_id"] = batch_id
        job["started_at"] = started_at
        if result:
            job["current_case_id"] = result.case.case_id
            job["results"] = [*job.get("results", []), result]
        if stopped and job["status"] == "running":
            job["status"] = "stopping"

    try:
        runner = EvalRunner(settings, template)
        existing_results = list(job.get("results", []))
        completed_case_ids = {result.case.case_id for result in existing_results}
        remaining_cases = [case for case in cases if case.case_id not in completed_case_ids]
        results = runner.run(
            remaining_cases,
            use_judge=use_judge,
            should_stop=job["stop_event"].is_set,
            on_progress=on_progress,
            batch_id=job.get("batch_id") or None,
            started_at=job.get("started_at") or None,
            initial_results=existing_results,
            total_cases=len(cases),
        )
        job["results"] = results
        if results:
            results_df, bad_cases_df = results_to_frames(results, template)
            report_md = generate_report(results, settings.target_name)
            save_exports(results_df, bad_cases_df, report_md)
            job["results_df"] = results_df
            job["bad_cases_df"] = bad_cases_df
            job["report_md"] = report_md
        job["status"] = "stopped" if job["stop_event"].is_set() else "completed"
    except Exception as exc:
        job["error"] = str(exc)
        job["status"] = "failed"


def render_eval_job(job: dict | None) -> None:
    if not job:
        return
    status = job.get("status", "")
    completed = int(job.get("completed", 0))
    total = int(job.get("total", 0))
    progress = completed / total if total else 0

    st.progress(progress, text=f"评测进度：{completed} / {total}")
    info_cols = st.columns(4)
    info_cols[0].metric("状态", _job_status_text(status))
    info_cols[1].metric("已完成", completed)
    info_cols[2].metric("剩余", max(total - completed, 0))
    info_cols[3].metric("Batch ID", job.get("batch_id") or "-")
    if job.get("current_case_id"):
        st.caption(f"最近完成 case：{job['current_case_id']}")

    if status in {"running", "stopping"}:
        stop_disabled = status == "stopping"
        if st.button("暂停评测并查看已完成结果", type="secondary", disabled=stop_disabled):
            job["stop_event"].set()
            job["status"] = "stopping"
            st.rerun()
        if status == "stopping":
            st.warning("正在暂停：会等待当前 case 结束，然后保存已完成结果。")
        time.sleep(1)
        st.rerun()

    if status in {"completed", "stopped"} and not job.get("applied"):
        st.session_state.results_df = job.get("results_df", pd.DataFrame())
        st.session_state.bad_cases_df = job.get("bad_cases_df", pd.DataFrame())
        st.session_state.report_md = job.get("report_md", "")
        job["applied"] = True
        if status == "stopped":
            st.warning(f"评测已暂停，已保留 {completed} 条完成结果，可先查看结果或继续评测。")
        else:
            st.success("评测完成")
        st.rerun()

    if status == "stopped" and completed < total:
        resume_cols = st.columns([1, 4])
        if resume_cols[0].button("继续评测剩余 case", type="primary"):
            st.session_state.eval_job = resume_eval_job(job)
            st.rerun()
        resume_cols[1].caption(f"将从第 {completed + 1} 条继续，剩余 {total - completed} 条，结果会合并到同一个 Batch ID。")

    if status == "failed":
        st.error(f"评测任务失败: {job.get('error')}")


def _job_status_text(status: str) -> str:
    return {
        "running": "运行中",
        "stopping": "暂停中",
        "stopped": "已暂停",
        "completed": "已完成",
        "failed": "失败",
    }.get(status, status or "-")


def comparison_summary(db: EvalDatabase, runs_df: pd.DataFrame, base_id: str, target_id: str):
    base_run = runs_df[runs_df["batch_id"] == base_id].iloc[0]
    target_run = runs_df[runs_df["batch_id"] == target_id].iloc[0]
    compare_df = db.compare_runs(base_id, target_id)
    return base_run, target_run, compare_df


def delta_metric(col, label: str, base, target, formatter):
    delta = target - base
    col.metric(label, formatter(target), delta=formatter(delta))


def render_version_compare(db: EvalDatabase) -> None:
    runs_df = db.list_runs()
    if len(runs_df) < 2:
        st.info("至少完成两次评测后，才能进行版本对比。")
        if not runs_df.empty:
            st.dataframe(runs_df, use_container_width=True, height=240)
        return

    labels = {run_label(row): row["batch_id"] for _, row in runs_df.iterrows()}
    label_list = list(labels.keys())
    selector_cols = st.columns(2)
    base_label = selector_cols[0].selectbox("基准版本", label_list, index=1, key="compare_base")
    target_label = selector_cols[1].selectbox("对比版本", label_list, index=0, key="compare_target")
    base_id = labels[base_label]
    target_id = labels[target_label]
    if base_id == target_id:
        st.warning("请选择两个不同的评测版本。")
        return

    base_run, target_run, compare_df = comparison_summary(db, runs_df, base_id, target_id)
    if (base_run.get("template_id") or "") != (target_run.get("template_id") or ""):
        st.warning("两次评测使用了不同模板，指标口径可能不完全一致。")

    metric_summary = db.compare_metric_summary(base_id, target_id)
    render_compare_metrics(base_run, target_run, metric_summary)
    render_compare_detail(compare_df)
    compare_report = generate_compare_report(base_run, target_run, compare_df)
    st.download_button("下载对比报告", compare_report.encode("utf-8"), "compare_report.md")


def render_compare_metrics(base_run, target_run, metric_summary: dict | None = None) -> None:
    st.subheader("基础指标变化")
    metric_cols = st.columns(5)
    delta_metric(metric_cols[0], "成功率", base_run["success_rate"], target_run["success_rate"], lambda v: f"{v:.1%}")
    delta_metric(metric_cols[1], "Bad Case 率", base_run["bad_case_rate"], target_run["bad_case_rate"], lambda v: f"{v:.1%}")
    delta_metric(metric_cols[2], "平均 rule_score", base_run["avg_rule_score"], target_run["avg_rule_score"], lambda v: f"{v:.1f}")
    delta_metric(metric_cols[3], "平均 judge_score", base_run.get("avg_judge_score", 0), target_run.get("avg_judge_score", 0), lambda v: f"{v:.2f}")
    metric_cols[4].metric("覆盖 case", f"{int(target_run['total_cases'])}", delta=f"{int(target_run['total_cases'] - base_run['total_cases'])}")

    # Product metric deltas
    if metric_summary:
        st.subheader("Product Metrics 变化")
        product_metric_specs = [
            ("Intent Accuracy", "intent_accuracy", lambda v: f"{v:.1%}" if v is not None else "N/A"),
            ("Answer Relevance", "answer_relevance_score", lambda v: f"{v:.2f}" if v is not None else "N/A"),
            ("Task Completion", "task_completion", lambda v: f"{v:.1%}" if v is not None else "N/A"),
            ("Multi-turn Completion", "multi_turn_completion", lambda v: f"{v:.1%}" if v is not None else "N/A"),
            ("Hallucination Rate", "hallucination", lambda v: f"{v:.1%}" if v is not None else "N/A"),
            ("Retrieval Recall", "retrieval_recall", lambda v: f"{v:.1%}" if v is not None else "N/A"),
            ("Tool Success Rate", "tool_success_rate", lambda v: f"{v:.1%}" if v is not None else "N/A"),
        ]
        # Use the run-level averages for display, fall back to metric_summary
        run_metric_map = {
            "intent_accuracy": ("avg_intent_accuracy", lambda v: f"{v:.1%}"),
            "answer_relevance_score": ("avg_answer_relevance_score", lambda v: f"{v:.2f}"),
            "task_completion": ("avg_task_completion", lambda v: f"{v:.1%}"),
            "multi_turn_completion": ("avg_multi_turn_completion", lambda v: f"{v:.1%}"),
            "hallucination": ("avg_hallucination_rate", lambda v: f"{v:.1%}"),
            "retrieval_recall": ("avg_retrieval_recall", lambda v: f"{v:.1%}"),
            "tool_success_rate": ("avg_tool_success_rate", lambda v: f"{v:.1%}"),
        }
        cols = st.columns(4)
        for i, (label, key, fmt) in enumerate(product_metric_specs):
            col_idx = i % 4
            run_col, run_fmt = run_metric_map.get(key, (None, None))
            base_val = base_run.get(run_col) if run_col else None
            target_val = target_run.get(run_col) if run_col else None
            if base_val is None and target_val is None and metric_summary:
                ms = metric_summary.get(label, {})
                base_val = ms.get("base")
                target_val = ms.get("target")
            if base_val is not None and target_val is not None:
                delta_val = target_val - base_val
                cols[col_idx].metric(
                    label,
                    run_fmt(target_val) if run_fmt else fmt(target_val),
                    delta=run_fmt(delta_val) if run_fmt else fmt(delta_val),
                )
            else:
                base_text = run_fmt(base_val) if run_fmt and base_val is not None else (fmt(base_val) if base_val is not None else "N/A")
                cols[col_idx].metric(label, base_text)


def render_compare_detail(compare_df: pd.DataFrame) -> None:
    if compare_df.empty:
        st.info("未找到可对比的 case 明细。")
        return

    status_counts = compare_df["change_status"].value_counts()
    status_cols = st.columns(4)
    status_cols[0].metric("收敛", int(status_counts.get("收敛", 0)))
    status_cols[1].metric("新增问题", int(status_counts.get("新增问题", 0)))
    status_cols[2].metric("未收敛", int(status_counts.get("未收敛", 0)))
    status_cols[3].metric("稳定通过", int(status_counts.get("稳定通过", 0)))

    trend_cols = st.columns(2)
    with trend_cols[0]:
        st.subheader("Bad Case 类型变化")
        type_trend = _category_trend(compare_df, "bad_case_type_base", "bad_case_type_target")
        st.dataframe(type_trend, use_container_width=True, height=260)
    with trend_cols[1]:
        st.subheader("根因变化")
        cause_trend = _category_trend(compare_df, "root_cause_base", "root_cause_target")
        st.dataframe(cause_trend, use_container_width=True, height=260)

    display_cols = [
        "case_id",
        "change_status",
        "rule_score_base",
        "rule_score_target",
        "rule_score_delta",
        "judge_average_base",
        "judge_average_target",
        "judge_average_delta",
        "bad_case_type_base",
        "bad_case_type_target",
        "root_cause_base",
        "root_cause_target",
        "user_query_target",
    ]
    existing_cols = [col for col in display_cols if col in compare_df.columns]
    st.subheader("Case 级变化")
    st.dataframe(compare_df[existing_cols], use_container_width=True, height=420)


def _category_trend(df: pd.DataFrame, base_col: str, target_col: str) -> pd.DataFrame:
    base = df[df.get("is_bad_case_base", 0).fillna(0).astype(int) == 1][base_col].fillna("").value_counts()
    target = df[df.get("is_bad_case_target", 0).fillna(0).astype(int) == 1][target_col].fillna("").value_counts()
    trend = pd.DataFrame({"基准版本": base, "对比版本": target}).fillna(0).astype(int)
    trend["变化"] = trend["对比版本"] - trend["基准版本"]
    return trend.reset_index().rename(columns={"index": "维度"})


def render_bad_case_visuals(bad_cases_df: pd.DataFrame, db: EvalDatabase) -> None:
    chart_cols = st.columns(3)
    with chart_cols[0]:
        st.subheader("Bad Case 类型")
        if bad_cases_df.empty:
            st.info("暂无 Bad Case 类型分布")
        else:
            type_counts = _counts_df(bad_cases_df, "bad_case_type", "类型")
            st.vega_lite_chart(
                type_counts,
                {
                    "mark": {"type": "arc", "innerRadius": 35},
                    "encoding": {
                        "theta": {"field": "数量", "type": "quantitative"},
                        "color": {"field": "类型", "type": "nominal"},
                        "tooltip": [{"field": "类型"}, {"field": "数量"}],
                    },
                },
                use_container_width=True,
            )
    with chart_cols[1]:
        st.subheader("严重度分布")
        if bad_cases_df.empty:
            st.info("暂无严重度分布")
        else:
            severity_counts = _counts_df(bad_cases_df, "severity", "严重度")
            st.bar_chart(severity_counts.set_index("严重度"))
    with chart_cols[2]:
        st.subheader("根因分布")
        if bad_cases_df.empty:
            st.info("暂无根因分布")
        else:
            cause_counts = _counts_df(bad_cases_df, "root_cause", "根因")
            st.bar_chart(cause_counts.set_index("根因"))

    st.subheader("根因趋势")
    root_trend = build_root_cause_trend(db)
    if root_trend.empty:
        st.info("至少需要有历史 Bad Case，才能展示根因趋势。")
    else:
        st.line_chart(root_trend)


def _counts_df(df: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    return df[column].fillna("未分类").replace("", "未分类").value_counts().rename_axis(label).reset_index(name="数量")


def build_root_cause_trend(db: EvalDatabase) -> pd.DataFrame:
    runs_df = db.list_runs().sort_values("started_at").tail(12)
    rows = []
    for _, run in runs_df.iterrows():
        scores = db.get_scores(run["batch_id"])
        if scores.empty:
            continue
        bad_scores = scores[scores["is_bad_case"].fillna(0).astype(int) == 1]
        counts = bad_scores["root_cause"].fillna("未分类").replace("", "未分类").value_counts()
        version = _short_version(run)
        for cause, count in counts.items():
            rows.append({"version": version, "root_cause": cause, "count": int(count)})
    if not rows:
        return pd.DataFrame()
    trend = pd.DataFrame(rows).pivot_table(index="version", columns="root_cause", values="count", aggfunc="sum", fill_value=0)
    return trend


def render_eval_trends(db: EvalDatabase) -> None:
    runs_df = db.list_runs().sort_values("started_at")
    if runs_df.empty:
        st.info("暂无历史评测。")
        return

    # --- Filter controls ---
    filter_options = db.get_filter_options()
    with st.expander("指标筛选（按场景/难度/标签/模板/Prompt）", expanded=False):
        filter_cols = st.columns(5)
        selected_scenarios = filter_cols[0].multiselect(
            "场景筛选",
            filter_options.get("scenarios", []),
            key="trend_filter_scenarios",
        )
        selected_difficulties = filter_cols[1].multiselect(
            "难度筛选",
            filter_options.get("difficulties", []),
            key="trend_filter_difficulties",
        )
        selected_tags = filter_cols[2].multiselect(
            "标签筛选",
            filter_options.get("tags", []),
            key="trend_filter_tags",
        )
        selected_templates = filter_cols[3].multiselect(
            "模板筛选",
            filter_options.get("templates", []),
            key="trend_filter_templates",
        )
        selected_prompts = filter_cols[4].multiselect(
            "Prompt 筛选",
            filter_options.get("prompt_names", []),
            key="trend_filter_prompts",
        )

    has_filters = any(
        [selected_scenarios, selected_difficulties, selected_tags, selected_templates, selected_prompts]
    )

    # Get filtered metric trends for product metrics
    if has_filters:
        metric_trend_df = db.get_filtered_metric_trends(
            scenarios=selected_scenarios or None,
            difficulties=selected_difficulties or None,
            tags=selected_tags or None,
            templates=selected_templates or None,
            prompt_names=selected_prompts or None,
        )
    else:
        metric_trend_df = db.get_filtered_metric_trends()

    # Basic metrics use runs_df (unfiltered historical view)
    max_count = min(30, len(runs_df))
    selected_count = st.slider(
        "展示最近 N 次评测",
        min_value=1,
        max_value=max_count,
        value=min(10, max_count),
        key="trend_count",
    )
    trend_df = runs_df.tail(selected_count).copy()
    trend_df["version"] = trend_df.apply(_short_version, axis=1)
    trend_df = trend_df.set_index("version")

    st.subheader("基础指标趋势")
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.caption("成功率 / Bad Case 率")
        st.line_chart(trend_df[["success_rate", "bad_case_rate"]])
    with chart_cols[1]:
        st.caption("平均分")
        st.line_chart(trend_df[["avg_rule_score", "avg_judge_score"]])

    # Product metric curves
    if metric_trend_df.empty:
        st.info("暂无 Product Metric 数据。")
    else:
        metric_trend_df = metric_trend_df.sort_values("started_at")
        metric_trend_df["version"] = metric_trend_df.apply(
            lambda row: f"{str(row.get('started_at', ''))[-8:]} {row.get('prompt_name', '') or '默认'} {str(row.get('batch_id', ''))[-8:]}",
            axis=1,
        )
        metric_trend_df = metric_trend_df.set_index("version")

        st.subheader("Product Metrics 趋势")
        if has_filters:
            st.caption("当前指标已按选中的筛选条件过滤（仅统计符合条件的 case）。")

        prod_chart_cols = st.columns(2)
        with prod_chart_cols[0]:
            st.caption("Task Completion Rate")
            if "task_completion" in metric_trend_df.columns:
                st.line_chart(metric_trend_df[["task_completion"]])
            else:
                st.info("暂无数据")
        with prod_chart_cols[1]:
            st.caption("Hallucination Rate")
            if "hallucination_rate" in metric_trend_df.columns:
                st.line_chart(metric_trend_df[["hallucination_rate"]])
            else:
                st.info("暂无数据")

        prod_chart_cols2 = st.columns(2)
        with prod_chart_cols2[0]:
            st.caption("Retrieval Recall")
            if "retrieval_recall" in metric_trend_df.columns:
                st.line_chart(metric_trend_df[["retrieval_recall"]])
            else:
                st.info("暂无数据")
        with prod_chart_cols2[1]:
            st.caption("Tool Success Rate")
            if "tool_success_rate" in metric_trend_df.columns:
                st.line_chart(metric_trend_df[["tool_success_rate"]])
            else:
                st.info("暂无数据")

        # Additional product metrics
        st.subheader("更多 Product Metrics")
        extra_chart_cols = st.columns(3)
        with extra_chart_cols[0]:
            st.caption("Intent Accuracy")
            if "intent_accuracy" in metric_trend_df.columns:
                st.line_chart(metric_trend_df[["intent_accuracy"]])
            else:
                st.info("暂无数据")
        with extra_chart_cols[1]:
            st.caption("Answer Relevance")
            if "answer_relevance_score" in metric_trend_df.columns:
                st.line_chart(metric_trend_df[["answer_relevance_score"]])
            else:
                st.info("暂无数据")
        with extra_chart_cols[2]:
            st.caption("Multi-turn Completion")
            if "multi_turn_completion" in metric_trend_df.columns:
                st.line_chart(metric_trend_df[["multi_turn_completion"]])
            else:
                st.info("暂无数据")

        # Show filtered metric summary table
        with st.expander("Product Metrics 明细表", expanded=False):
            display_cols = [
                "version", "case_count", "intent_accuracy", "answer_relevance_score",
                "task_completion", "multi_turn_completion", "hallucination_rate",
                "retrieval_recall", "tool_success_rate",
            ]
            existing_cols = [c for c in display_cols if c in metric_trend_df.columns]
            st.dataframe(
                metric_trend_df[existing_cols].reset_index(),
                use_container_width=True,
                height=300,
            )

    # Historical version detail (unfiltered)
    st.subheader("历史版本明细")
    display_cols = [
        "started_at",
        "batch_id",
        "target_name",
        "prompt_name",
        "template_id",
        "total_cases",
        "success_rate",
        "avg_rule_score",
        "avg_judge_score",
        "bad_case_rate",
        "avg_latency_ms",
    ]
    existing_cols = [c for c in display_cols if c in runs_df.columns]
    st.dataframe(runs_df[existing_cols], use_container_width=True, height=420)


def _short_version(row) -> str:
    prompt = row.get("prompt_name") or "默认"
    batch = str(row.get("batch_id", ""))[-8:]
    return f"{row.get('started_at', '')[-8:]} {prompt} {batch}"


def render_dual_eval(
    base_settings: Settings,
    template: EvalTemplate,
    cases_df: pd.DataFrame,
    api_url: str,
    timeout: int,
    client_mode: str,
    target_name: str,
    openai_base_url: str,
    openai_model: str,
    judge_enabled: bool,
    db: EvalDatabase,
) -> None:
    st.subheader("同时评测两个版本 / 两个 Prompt")
    st.caption("两轮会使用相同测试集和模板顺序执行，并自动生成对比报告。被测接口会收到 prompt_name、prompt、system_prompt 字段。")

    cols = st.columns(2)
    with cols[0]:
        prompt_a_name = st.text_input("Prompt A 名称", value="Prompt A")
        prompt_a_text = st.text_area("Prompt A 内容", height=180)
    with cols[1]:
        prompt_b_name = st.text_input("Prompt B 名称", value="Prompt B")
        prompt_b_text = st.text_area("Prompt B 内容", height=180)

    if st.button("开始双版本评测", type="primary"):
        cases = dataframe_to_cases(cases_df)
        with st.spinner("正在评测 Prompt A..."):
            settings_a = build_settings(
                base_settings,
                api_url,
                timeout,
                client_mode,
                openai_base_url,
                openai_model,
                target_name,
                template.id,
                prompt_a_name,
                prompt_a_text,
            )
            results_a, results_df_a, bad_cases_df_a, report_a = run_eval_to_frames(settings_a, template, cases, judge_enabled)

        with st.spinner("正在评测 Prompt B..."):
            settings_b = build_settings(
                base_settings,
                api_url,
                timeout,
                client_mode,
                openai_base_url,
                openai_model,
                target_name,
                template.id,
                prompt_b_name,
                prompt_b_text,
            )
            results_b, results_df_b, bad_cases_df_b, report_b = run_eval_to_frames(settings_b, template, cases, judge_enabled)

        runs_df = db.list_runs()
        base_run, target_run, compare_df = comparison_summary(db, runs_df, results_a[0].batch_id, results_b[0].batch_id)
        compare_report = generate_compare_report(base_run, target_run, compare_df)
        save_exports(results_df_b, bad_cases_df_b, report_b, compare_report)
        st.session_state.results_df = results_df_b
        st.session_state.bad_cases_df = bad_cases_df_b
        st.session_state.report_md = report_b
        st.session_state.compare_report_md = compare_report
        st.success(f"双版本评测完成：{results_a[0].batch_id} vs {results_b[0].batch_id}")

    if st.session_state.compare_report_md:
        st.download_button("下载最近一次对比报告", st.session_state.compare_report_md.encode("utf-8"), "compare_report.md")
        with st.expander("查看最近一次对比报告", expanded=False):
            st.markdown(st.session_state.compare_report_md)


init_state()
base_settings = get_settings()
db = EvalDatabase(base_settings.db_path)
templates = list_templates()
template_ids = [item.id for item in templates]
template_names = {item.id: item.name for item in templates}
default_template_id = st.session_state.active_template_id if st.session_state.active_template_id in template_ids else base_settings.template_id
if default_template_id not in template_ids:
    default_template_id = "product_recommendation"

st.title("PM-Eval v0.1")

with st.sidebar:
    st.header("配置")
    if st.button("重新读取 .env"):
        for key in [
            "cfg_target_name",
            "cfg_api_url",
            "cfg_timeout",
            "cfg_client_mode",
            "cfg_template_id",
            "cfg_prompt_name",
            "cfg_prompt_text",
            "cfg_openai_base_url",
            "cfg_openai_model",
        ]:
            st.session_state.pop(key, None)
        st.session_state.active_template_id = get_settings().template_id
        st.rerun()

    target_name = st.text_input("TARGET_NAME", value=base_settings.target_name, key="cfg_target_name")
    api_url = st.text_input("TARGET_API_URL", value=base_settings.api_url, key="cfg_api_url")
    timeout = st.number_input("接口超时秒数", min_value=1, max_value=300, value=base_settings.api_timeout, key="cfg_timeout")
    client_mode = st.selectbox(
        "client_mode",
        options=["mock", "real"],
        index=1 if base_settings.client_mode == "real" else 0,
        key="cfg_client_mode",
    )
    selected_template_id = st.selectbox(
        "评测场景模板",
        options=template_ids,
        index=template_ids.index(default_template_id),
        format_func=lambda item: template_names.get(item, item),
        key="cfg_template_id",
    )
    st.session_state.active_template_id = selected_template_id
    selected_template = load_template(selected_template_id)
    st.caption(selected_template.description)

    default_prompt_name = base_settings.prompt_name or "默认 Prompt"
    prompt_name = st.text_input("Prompt 名称", value=default_prompt_name, key="cfg_prompt_name")
    prompt_text = st.text_area("单版本 Prompt 内容", value=base_settings.prompt_text, height=120, key="cfg_prompt_text")

    judge_configured = bool(base_settings.openai_api_key)
    use_judge = st.toggle("启用 LLM-as-Judge", value=judge_configured)
    if use_judge and judge_configured:
        st.success("Judge 已启用")
    elif use_judge and not judge_configured:
        st.warning("Judge 未启用：OPENAI_API_KEY 未配置，将只使用规则评分")
    else:
        st.info("Judge 未启用：当前只使用规则评分")
    openai_base_url = st.text_input("OPENAI_BASE_URL", value=base_settings.openai_base_url, key="cfg_openai_base_url")
    openai_model = st.text_input("OPENAI_MODEL", value=base_settings.openai_model, key="cfg_openai_model")

uploaded = st.file_uploader("上传 CSV 替换测试集", type=["csv"])
if uploaded:
    try:
        st.session_state.cases_df = read_cases_csv(uploaded)
        st.success("测试集已替换")
    except Exception as exc:
        st.error(f"CSV 加载失败: {exc}")

cases_df = normalize_cases_df(st.session_state.cases_df)
st.session_state.cases_df = cases_df
results_df = st.session_state.results_df
bad_cases_df = st.session_state.bad_cases_df
report_md = st.session_state.report_md

total = len(results_df) if not results_df.empty else len(cases_df)
success_rate = metric_value(results_df, "success")
avg_rule = metric_value(results_df, "rule_score")
avg_judge = results_df["judge_average"].dropna().mean() if "judge_average" in results_df else 0
bad_rate = metric_value(results_df, "is_bad_case")
avg_latency = metric_value(results_df, "latency_ms")

metric_cols = st.columns(6)
metric_cols[0].metric("总 case 数", f"{total}")
metric_cols[1].metric("成功率", f"{success_rate:.1%}")
metric_cols[2].metric("Bad Case 率", f"{bad_rate:.1%}")
metric_cols[3].metric("平均 rule_score", f"{avg_rule:.1f}")
metric_cols[4].metric("平均 judge_score", f"{avg_judge:.2f}" if pd.notna(avg_judge) and avg_judge else "N/A")
metric_cols[5].metric("平均响应时间", f"{avg_latency:.0f} ms")

product_metric_cols = st.columns(6)
product_metric_cols[0].metric("Task Completion", metric_display(results_df, "task_completion"))
product_metric_cols[1].metric("Hallucination Rate", metric_display(results_df, "hallucination"))
product_metric_cols[2].metric("Intent Accuracy", metric_display(results_df, "intent_accuracy"))
product_metric_cols[3].metric("Answer Relevance", metric_display(results_df, "answer_relevance_score", percent=False))
product_metric_cols[4].metric("Retrieval Recall", metric_display(results_df, "retrieval_recall"))
product_metric_cols[5].metric("Tool Success", metric_display(results_df, "tool_success_rate"))

render_eval_job(st.session_state.eval_job)

job_running = bool(st.session_state.eval_job and st.session_state.eval_job.get("status") in {"running", "stopping"})
run_clicked = st.button("开始评测", type="primary", disabled=job_running)
if run_clicked:
    errors = validate_cases_df(cases_df)
    if errors:
        st.error("Eval pre-check failed: " + " ".join(errors))
        st.stop()
    settings = build_settings(
        base_settings,
        api_url,
        timeout,
        client_mode,
        openai_base_url,
        openai_model,
        target_name,
        selected_template.id,
        prompt_name,
        prompt_text,
    )
    cases = dataframe_to_cases(cases_df)
    st.session_state.eval_job = start_eval_job(
        settings,
        selected_template,
        cases,
        use_judge=(use_judge and judge_configured),
    )
    st.rerun()

tabs = st.tabs(["测试集管理", "评测结果", "版本对比", "评测趋势", "双版本评测", "Bad Case", "Markdown 报告", "导出"])

with tabs[0]:
    render_case_manager()

with tabs[1]:
    if results_df.empty:
        st.info("尚未运行评测")
    else:
        metric_summary = metric_summary_frame(results_df)
        if not metric_summary.empty:
            st.subheader("Product Metric Summary")
            st.dataframe(metric_summary, use_container_width=True, height=260)
        scenario_summary = scenario_result_frame(results_df)
        if not scenario_summary.empty:
            st.subheader("Scenario / Template Summary")
            st.dataframe(scenario_summary, use_container_width=True, height=220)
        st.dataframe(results_df, use_container_width=True, height=520)

with tabs[2]:
    render_version_compare(db)

with tabs[3]:
    render_eval_trends(db)

with tabs[4]:
    render_dual_eval(
        base_settings,
        selected_template,
        cases_df,
        api_url,
        int(timeout),
        client_mode,
        target_name,
        openai_base_url,
        openai_model,
        judge_enabled=(use_judge and judge_configured),
        db=db,
    )

with tabs[5]:
    render_bad_case_visuals(bad_cases_df, db)
    if bad_cases_df.empty:
        st.info("暂无 Bad Case")
    else:
        filter_cols = st.columns(3)
        type_options = sorted([v for v in bad_cases_df["bad_case_type"].dropna().unique().tolist() if v])
        cause_options = sorted([v for v in bad_cases_df["root_cause"].dropna().unique().tolist() if v])
        severity_options = sorted([v for v in bad_cases_df["severity"].dropna().unique().tolist() if v])
        selected_types = filter_cols[0].multiselect("bad_case_type", type_options, default=type_options)
        selected_causes = filter_cols[1].multiselect("root_cause", cause_options, default=cause_options)
        selected_severity = filter_cols[2].multiselect("severity", severity_options, default=severity_options)

        filtered_bad_cases = bad_cases_df[
            bad_cases_df["bad_case_type"].isin(selected_types)
            & bad_cases_df["root_cause"].isin(selected_causes)
            & bad_cases_df["severity"].isin(selected_severity)
        ]
        st.caption(f"当前筛选结果：{len(filtered_bad_cases)} / {len(bad_cases_df)}")
        st.dataframe(filtered_bad_cases, use_container_width=True, height=520)

with tabs[6]:
    if not report_md:
        st.info("运行评测后生成报告")
    else:
        st.markdown(report_md)

with tabs[7]:
    export_dir = ROOT_DIR / "exports"
    if results_df.empty:
        st.info("运行评测后可导出 results.csv、bad_cases.csv、report.md")
    else:
        st.download_button("下载 results.csv", results_df.to_csv(index=False).encode("utf-8-sig"), "results.csv")
        st.download_button("下载 bad_cases.csv", bad_cases_df.to_csv(index=False).encode("utf-8-sig"), "bad_cases.csv")
        st.download_button("下载 report.md", report_md.encode("utf-8"), "report.md")
        if st.session_state.compare_report_md:
            st.download_button("下载 compare_report.md", st.session_state.compare_report_md.encode("utf-8"), "compare_report.md")
        st.caption(f"文件也已保存到 {export_dir}")
