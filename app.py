from pathlib import Path

import pandas as pd
import streamlit as st

from pmeval.config import ROOT_DIR, Settings, get_settings
from pmeval.report_generator import generate_report
from pmeval.runner import EvalRunner
from pmeval.utils import dataframe_to_cases, ensure_dirs, read_cases_csv


st.set_page_config(page_title="PM-Eval v0.1", layout="wide")
ensure_dirs(ROOT_DIR)


def init_state() -> None:
    if "cases_df" not in st.session_state:
        st.session_state.cases_df = read_cases_csv(ROOT_DIR / "data" / "sample_cases_30.csv")
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()
    if "bad_cases_df" not in st.session_state:
        st.session_state.bad_cases_df = pd.DataFrame()
    if "report_md" not in st.session_state:
        st.session_state.report_md = ""


def results_to_frames(results):
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
            "success": item.target.success,
            "latency_ms": round(item.target.latency_ms, 1),
            "rule_score": item.rule_score.score,
            "judge_average": judge.average_score if judge else None,
            "judge_need_understanding": judge.scores.get("need_understanding_score") if judge else None,
            "judge_constraint_satisfaction": judge.scores.get("constraint_satisfaction_score") if judge else None,
            "judge_relevance": judge.scores.get("relevance_score") if judge else None,
            "judge_usefulness": judge.scores.get("usefulness_score") if judge else None,
            "judge_faithfulness": judge.scores.get("faithfulness_score") if judge else None,
            "judge_clarity": judge.scores.get("clarity_score") if judge else None,
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


def save_exports(results_df: pd.DataFrame, bad_cases_df: pd.DataFrame, report_md: str) -> None:
    export_dir = ROOT_DIR / "exports"
    results_df.to_csv(export_dir / "results.csv", index=False, encoding="utf-8-sig")
    bad_cases_df.to_csv(export_dir / "bad_cases.csv", index=False, encoding="utf-8-sig")
    (export_dir / "report.md").write_text(report_md, encoding="utf-8")


init_state()
base_settings = get_settings()

st.title("PM-Eval v0.1")

with st.sidebar:
    st.header("配置")
    target_name = st.text_input("TARGET_NAME", value=base_settings.target_name)
    api_url = st.text_input("TARGET_API_URL", value=base_settings.api_url)
    timeout = st.number_input("接口超时秒数", min_value=1, max_value=120, value=base_settings.api_timeout)
    client_mode = st.selectbox(
        "client_mode",
        options=["mock", "real"],
        index=1 if base_settings.client_mode == "real" else 0,
    )
    judge_configured = bool(base_settings.openai_api_key)
    use_judge = st.toggle("启用 LLM-as-Judge", value=judge_configured)
    if use_judge and judge_configured:
        st.success("Judge 已启用")
    elif use_judge and not judge_configured:
        st.warning("Judge 未启用：OPENAI_API_KEY 未配置，将只使用规则评分")
    else:
        st.info("Judge 未启用：当前只使用规则评分")
    openai_base_url = st.text_input("OPENAI_BASE_URL", value=base_settings.openai_base_url)
    openai_model = st.text_input("OPENAI_MODEL", value=base_settings.openai_model)

uploaded = st.file_uploader("上传 CSV 替换测试集", type=["csv"])
if uploaded:
    try:
        st.session_state.cases_df = read_cases_csv(uploaded)
        st.success("测试集已替换")
    except Exception as exc:
        st.error(f"CSV 加载失败: {exc}")

cases_df = st.session_state.cases_df
results_df = st.session_state.results_df
bad_cases_df = st.session_state.bad_cases_df
report_md = st.session_state.report_md

total = len(results_df) if not results_df.empty else len(cases_df)
success_rate = results_df["success"].mean() if not results_df.empty else 0
avg_rule = results_df["rule_score"].mean() if "rule_score" in results_df else 0
avg_judge = results_df["judge_average"].dropna().mean() if "judge_average" in results_df else 0
bad_rate = results_df["is_bad_case"].mean() if "is_bad_case" in results_df else 0
avg_latency = results_df["latency_ms"].mean() if "latency_ms" in results_df else 0

metric_cols = st.columns(6)
metric_cols[0].metric("总 case 数", f"{total}")
metric_cols[1].metric("成功率", f"{success_rate:.1%}")
metric_cols[2].metric("Bad Case 率", f"{bad_rate:.1%}")
metric_cols[3].metric("平均 rule_score", f"{avg_rule:.1f}")
metric_cols[4].metric("平均 judge_score", f"{avg_judge:.2f}" if pd.notna(avg_judge) and avg_judge else "N/A")
metric_cols[5].metric("平均响应时间", f"{avg_latency:.0f} ms")

run_clicked = st.button("开始评测", type="primary")
if run_clicked:
    settings = Settings(
        api_url=api_url,
        api_timeout=int(timeout),
        mock_mode=(client_mode == "mock"),
        client_mode=client_mode,
        openai_api_key=base_settings.openai_api_key,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
        db_path=base_settings.db_path,
        target_name=target_name,
    )
    cases = dataframe_to_cases(cases_df)
    with st.spinner("正在批量评测..."):
        try:
            results = EvalRunner(settings).run(cases, use_judge=(use_judge and judge_configured))
            results_df, bad_cases_df = results_to_frames(results)
            report_md = generate_report(results, settings.target_name)
            save_exports(results_df, bad_cases_df, report_md)
            st.session_state.results_df = results_df
            st.session_state.bad_cases_df = bad_cases_df
            st.session_state.report_md = report_md
            st.success("评测完成")
            st.rerun()
        except Exception as exc:
            st.error(f"评测任务失败: {exc}")

tabs = st.tabs(["测试集预览", "评测结果", "Bad Case", "Markdown 报告", "导出"])

with tabs[0]:
    st.dataframe(cases_df, use_container_width=True, height=420)

with tabs[1]:
    if results_df.empty:
        st.info("尚未运行评测")
    else:
        st.dataframe(results_df, use_container_width=True, height=520)

with tabs[2]:
    if bad_cases_df.empty:
        st.info("尚无 Bad Case")
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

with tabs[3]:
    if not report_md:
        st.info("运行评测后生成报告")
    else:
        st.markdown(report_md)

with tabs[4]:
    export_dir = ROOT_DIR / "exports"
    if results_df.empty:
        st.info("运行评测后可导出 results.csv、bad_cases.csv、report.md")
    else:
        st.download_button("下载 results.csv", results_df.to_csv(index=False).encode("utf-8-sig"), "results.csv")
        st.download_button("下载 bad_cases.csv", bad_cases_df.to_csv(index=False).encode("utf-8-sig"), "bad_cases.csv")
        st.download_button("下载 report.md", report_md.encode("utf-8"), "report.md")
        st.caption(f"文件也已保存到 {export_dir}")
