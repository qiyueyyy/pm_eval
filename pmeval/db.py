import json
import sqlite3
from pathlib import Path

import pandas as pd

from pmeval.models import EvalCase, EvalResult


SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_cases (
  case_id TEXT PRIMARY KEY,
  user_query TEXT,
  scenario_type TEXT,
  template_id TEXT,
  expected_behavior TEXT,
  constraints_json TEXT,
  difficulty TEXT,
  tags TEXT
);
CREATE TABLE IF NOT EXISTS eval_runs (
  batch_id TEXT PRIMARY KEY,
  started_at TEXT,
  total_cases INTEGER,
  success_count INTEGER,
  avg_rule_score REAL,
  avg_judge_score REAL,
  bad_case_count INTEGER,
  avg_latency_ms REAL,
  template_id TEXT,
  target_name TEXT,
  prompt_name TEXT,
  prompt_text TEXT
);
CREATE TABLE IF NOT EXISTS eval_scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id TEXT,
  case_id TEXT,
  user_query TEXT,
  scenario_type TEXT,
  template_id TEXT,
  response_text TEXT,
  success INTEGER,
  error TEXT,
  latency_ms REAL,
  rule_score INTEGER,
  rule_details_json TEXT,
  judge_scores_json TEXT,
  judge_average REAL,
  judge_error TEXT,
  is_bad_case INTEGER,
  bad_case_type TEXT,
  root_cause TEXT,
  improvement_suggestion TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS bad_cases (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id TEXT,
  case_id TEXT,
  bad_case_type TEXT,
  root_cause TEXT,
  improvement_suggestion TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS eval_metric_scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id TEXT,
  case_id TEXT,
  scenario_type TEXT,
  template_id TEXT,
  difficulty TEXT,
  tags TEXT,
  intent_accuracy REAL,
  answer_relevance_score REAL,
  task_completion REAL,
  multi_turn_completion REAL,
  hallucination INTEGER,
  retrieval_recall REAL,
  tool_success_rate REAL,
  expected_tool_coverage REAL,
  details_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class EvalDatabase:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    def connect(self):
        return sqlite3.connect(self.db_path)

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._ensure_column(conn, "eval_runs", "avg_judge_score", "REAL")
            self._ensure_column(conn, "eval_runs", "template_id", "TEXT")
            self._ensure_column(conn, "eval_runs", "target_name", "TEXT")
            self._ensure_column(conn, "eval_runs", "prompt_name", "TEXT")
            self._ensure_column(conn, "eval_runs", "prompt_text", "TEXT")
            self._ensure_column(conn, "eval_runs", "avg_intent_accuracy", "REAL")
            self._ensure_column(conn, "eval_runs", "avg_answer_relevance_score", "REAL")
            self._ensure_column(conn, "eval_runs", "avg_task_completion", "REAL")
            self._ensure_column(conn, "eval_runs", "avg_multi_turn_completion", "REAL")
            self._ensure_column(conn, "eval_runs", "avg_hallucination_rate", "REAL")
            self._ensure_column(conn, "eval_runs", "avg_retrieval_recall", "REAL")
            self._ensure_column(conn, "eval_runs", "avg_tool_success_rate", "REAL")
            self._ensure_column(conn, "eval_cases", "template_id", "TEXT")
            self._ensure_column(conn, "eval_scores", "scenario_type", "TEXT")
            self._ensure_column(conn, "eval_scores", "template_id", "TEXT")
            self._ensure_column(conn, "eval_scores", "bad_case_type", "TEXT")
            self._ensure_column(conn, "eval_scores", "root_cause", "TEXT")
            self._ensure_column(conn, "eval_scores", "improvement_suggestion", "TEXT")
            self._ensure_column(conn, "eval_metric_scores", "difficulty", "TEXT")
            self._ensure_column(conn, "eval_metric_scores", "tags", "TEXT")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def upsert_cases(self, cases: list[EvalCase]) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO eval_cases
                (case_id, user_query, scenario_type, template_id, expected_behavior, constraints_json, difficulty, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.case_id,
                        c.user_query,
                        c.scenario_type,
                        getattr(c, "template_id", ""),
                        c.expected_behavior,
                        c.constraints_json,
                        c.difficulty,
                        c.tags,
                    )
                    for c in cases
                ],
            )

    def save_run(
        self,
        batch_id: str,
        started_at: str,
        results: list[EvalResult],
        template_id: str = "",
        target_name: str = "",
        prompt_name: str = "",
        prompt_text: str = "",
    ) -> None:
        total = len(results)
        success_count = sum(1 for r in results if r.target.success)
        avg_rule = sum(r.rule_score.score for r in results) / total if total else 0
        judge_values = [r.judge_score.average_score for r in results if r.judge_score and not r.judge_score.error]
        avg_judge = sum(judge_values) / len(judge_values) if judge_values else 0
        bad_count = sum(1 for r in results if r.is_bad_case)
        avg_latency = sum(r.target.latency_ms for r in results) / total if total else 0
        metric_avgs = self._compute_metric_averages(results)
        with self.connect() as conn:
            conn.execute("DELETE FROM eval_scores WHERE batch_id = ?", (batch_id,))
            conn.execute("DELETE FROM bad_cases WHERE batch_id = ?", (batch_id,))
            conn.execute("DELETE FROM eval_metric_scores WHERE batch_id = ?", (batch_id,))
            conn.execute(
                """
                INSERT OR REPLACE INTO eval_runs
                (batch_id, started_at, total_cases, success_count, avg_rule_score, avg_judge_score,
                 bad_case_count, avg_latency_ms, template_id, target_name, prompt_name, prompt_text,
                 avg_intent_accuracy, avg_answer_relevance_score, avg_task_completion,
                 avg_multi_turn_completion, avg_hallucination_rate, avg_retrieval_recall,
                 avg_tool_success_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    started_at,
                    total,
                    success_count,
                    avg_rule,
                    avg_judge,
                    bad_count,
                    avg_latency,
                    template_id,
                    target_name,
                    prompt_name,
                    prompt_text,
                    metric_avgs.get("avg_intent_accuracy"),
                    metric_avgs.get("avg_answer_relevance_score"),
                    metric_avgs.get("avg_task_completion"),
                    metric_avgs.get("avg_multi_turn_completion"),
                    metric_avgs.get("avg_hallucination_rate"),
                    metric_avgs.get("avg_retrieval_recall"),
                    metric_avgs.get("avg_tool_success_rate"),
                ),
            )
            conn.executemany(
                """
                INSERT INTO eval_scores
                (batch_id, case_id, user_query, scenario_type, template_id, response_text, success, error, latency_ms, rule_score,
                 rule_details_json, judge_scores_json, judge_average, judge_error, is_bad_case,
                 bad_case_type, root_cause, improvement_suggestion)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [self._score_row(r) for r in results],
            )
            conn.executemany(
                """
                INSERT INTO bad_cases
                (batch_id, case_id, bad_case_type, root_cause, improvement_suggestion)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (r.batch_id, r.case.case_id, r.bad_case_type, r.root_cause, r.improvement_suggestion)
                    for r in results
                    if r.is_bad_case
                ],
            )
            conn.executemany(
                """
                INSERT INTO eval_metric_scores
                (batch_id, case_id, scenario_type, template_id, difficulty, tags,
                 intent_accuracy, answer_relevance_score,
                 task_completion, multi_turn_completion, hallucination, retrieval_recall, tool_success_rate,
                 expected_tool_coverage, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [self._metric_row(r) for r in results if hasattr(r, "metric_score") and r.metric_score],
            )

    def list_runs(self) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                """
                SELECT
                  batch_id,
                  started_at,
                  total_cases,
                  success_count,
                  avg_rule_score,
                  COALESCE(avg_judge_score, 0) AS avg_judge_score,
                  bad_case_count,
                  avg_latency_ms,
                  template_id,
                  target_name,
                  prompt_name,
                  prompt_text,
                  COALESCE(avg_intent_accuracy, 0) AS avg_intent_accuracy,
                  COALESCE(avg_answer_relevance_score, 0) AS avg_answer_relevance_score,
                  COALESCE(avg_task_completion, 0) AS avg_task_completion,
                  COALESCE(avg_multi_turn_completion, 0) AS avg_multi_turn_completion,
                  COALESCE(avg_hallucination_rate, 0) AS avg_hallucination_rate,
                  COALESCE(avg_retrieval_recall, 0) AS avg_retrieval_recall,
                  COALESCE(avg_tool_success_rate, 0) AS avg_tool_success_rate,
                  CASE WHEN total_cases > 0 THEN success_count * 1.0 / total_cases ELSE 0 END AS success_rate,
                  CASE WHEN total_cases > 0 THEN bad_case_count * 1.0 / total_cases ELSE 0 END AS bad_case_rate
                FROM eval_runs
                ORDER BY started_at DESC
                """,
                conn,
            )

    def get_scores(self, batch_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                """
                SELECT
                  batch_id,
                  case_id,
                  user_query,
                  scenario_type,
                  template_id,
                  response_text,
                  success,
                  error,
                  latency_ms,
                  rule_score,
                  rule_details_json,
                  judge_scores_json,
                  judge_average,
                  judge_error,
                  is_bad_case,
                  bad_case_type,
                  root_cause,
                  improvement_suggestion
                FROM eval_scores
                WHERE batch_id = ?
                """,
                conn,
                params=(batch_id,),
            )

    def get_metric_scores(self, batch_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                """
                SELECT
                  batch_id,
                  case_id,
                  scenario_type,
                  template_id,
                  intent_accuracy,
                  answer_relevance_score,
                  task_completion,
                  multi_turn_completion,
                  hallucination,
                  retrieval_recall,
                  tool_success_rate,
                  expected_tool_coverage,
                  details_json
                FROM eval_metric_scores
                WHERE batch_id = ?
                """,
                conn,
                params=(batch_id,),
            )

    def compare_runs(self, base_batch_id: str, target_batch_id: str) -> pd.DataFrame:
        base = self.get_scores(base_batch_id)
        target = self.get_scores(target_batch_id)
        if base.empty and target.empty:
            return pd.DataFrame()
        merged = base.merge(target, on="case_id", how="outer", suffixes=("_base", "_target"))
        merged["change_status"] = merged.apply(_change_status, axis=1)
        merged["rule_score_delta"] = merged["rule_score_target"].fillna(0) - merged["rule_score_base"].fillna(0)
        merged["judge_average_delta"] = merged["judge_average_target"].fillna(0) - merged["judge_average_base"].fillna(0)
        # Merge metric scores for product metric deltas
        base_metrics = self.get_metric_scores(base_batch_id)
        target_metrics = self.get_metric_scores(target_batch_id)
        if not base_metrics.empty or not target_metrics.empty:
            metric_merged = base_metrics.merge(
                target_metrics, on="case_id", how="outer", suffixes=("_base", "_target")
            )
            metric_cols = [
                "intent_accuracy", "answer_relevance_score", "task_completion",
                "multi_turn_completion", "hallucination", "retrieval_recall", "tool_success_rate",
            ]
            for col in metric_cols:
                base_col = f"{col}_base"
                target_col = f"{col}_target"
                if base_col in metric_merged.columns and target_col in metric_merged.columns:
                    metric_merged[f"{col}_delta"] = (
                        metric_merged[target_col].fillna(0) - metric_merged[base_col].fillna(0)
                    )
            # Keep only case_id and delta columns to avoid column name conflicts
            delta_cols = ["case_id"] + [f"{col}_delta" for col in metric_cols]
            existing_delta_cols = [c for c in delta_cols if c in metric_merged.columns]
            merged = merged.merge(
                metric_merged[existing_delta_cols], on="case_id", how="left"
            )
        return merged

    def compare_metric_summary(self, base_batch_id: str, target_batch_id: str) -> dict[str, dict[str, float]]:
        """Return per-metric averages for two runs: {metric_name: {base: val, target: val, delta: val}}."""
        base_metrics = self.get_metric_scores(base_batch_id)
        target_metrics = self.get_metric_scores(target_batch_id)
        metric_cols = [
            ("intent_accuracy", "Intent Accuracy"),
            ("answer_relevance_score", "Answer Relevance"),
            ("task_completion", "Task Completion"),
            ("multi_turn_completion", "Multi-turn Completion"),
            ("hallucination", "Hallucination Rate"),
            ("retrieval_recall", "Retrieval Recall"),
            ("tool_success_rate", "Tool Success Rate"),
        ]
        result = {}
        for col, label in metric_cols:
            base_val = base_metrics[col].dropna().mean() if col in base_metrics.columns and not base_metrics.empty else None
            target_val = target_metrics[col].dropna().mean() if col in target_metrics.columns and not target_metrics.empty else None
            delta = None
            if base_val is not None and target_val is not None:
                delta = target_val - base_val
            result[label] = {"base": base_val, "target": target_val, "delta": delta}
        return result

    def get_filtered_metric_trends(
        self,
        scenarios=None,
        difficulties=None,
        tags=None,
        templates=None,
        prompt_names=None,
    ) -> pd.DataFrame:
        """Return per-batch metric averages, optionally filtered by scenario/difficulty/tags/template/prompt."""
        with self.connect() as conn:
            query = """
                SELECT
                  m.batch_id,
                  r.started_at,
                  r.target_name,
                  r.prompt_name,
                  r.template_id AS run_template_id,
                  COUNT(DISTINCT m.case_id) AS case_count,
                  AVG(m.intent_accuracy) AS intent_accuracy,
                  AVG(m.answer_relevance_score) AS answer_relevance_score,
                  AVG(m.task_completion) AS task_completion,
                  AVG(m.multi_turn_completion) AS multi_turn_completion,
                  AVG(m.hallucination) AS hallucination_rate,
                  AVG(m.retrieval_recall) AS retrieval_recall,
                  AVG(m.tool_success_rate) AS tool_success_rate
                FROM eval_metric_scores m
                JOIN eval_runs r ON m.batch_id = r.batch_id
                WHERE 1=1
            """
            params: list = []
            if scenarios:
                placeholders = ",".join(["?"] * len(scenarios))
                query += f" AND m.scenario_type IN ({placeholders})"
                params.extend(scenarios)
            if difficulties:
                placeholders = ",".join(["?"] * len(difficulties))
                query += f" AND m.difficulty IN ({placeholders})"
                params.extend(difficulties)
            if tags:
                tag_clauses = " OR ".join(["m.tags LIKE ?" for _ in tags])
                query += f" AND ({tag_clauses})"
                params.extend([f"%{tag}%" for tag in tags])
            if templates:
                placeholders = ",".join(["?"] * len(templates))
                query += f" AND m.template_id IN ({placeholders})"
                params.extend(templates)
            if prompt_names:
                placeholders = ",".join(["?"] * len(prompt_names))
                query += f" AND r.prompt_name IN ({placeholders})"
                params.extend(prompt_names)
            query += " GROUP BY m.batch_id ORDER BY r.started_at ASC"
            return pd.read_sql_query(query, conn, params=params)

    def get_filter_options(self) -> dict[str, list[str]]:
        """Return available filter values for trends page."""
        with self.connect() as conn:
            scenarios = pd.read_sql_query(
                "SELECT DISTINCT scenario_type FROM eval_metric_scores WHERE scenario_type IS NOT NULL AND scenario_type != '' ORDER BY scenario_type",
                conn,
            )["scenario_type"].tolist()
            difficulties = pd.read_sql_query(
                "SELECT DISTINCT difficulty FROM eval_metric_scores WHERE difficulty IS NOT NULL AND difficulty != '' ORDER BY difficulty",
                conn,
            )["difficulty"].tolist()
            templates = pd.read_sql_query(
                "SELECT DISTINCT template_id FROM eval_metric_scores WHERE template_id IS NOT NULL AND template_id != '' ORDER BY template_id",
                conn,
            )["template_id"].tolist()
            prompt_names = pd.read_sql_query(
                "SELECT DISTINCT prompt_name FROM eval_runs WHERE prompt_name IS NOT NULL AND prompt_name != '' ORDER BY prompt_name",
                conn,
            )["prompt_name"].tolist()
            # Tags need special handling since they're comma-separated
            tags_df = pd.read_sql_query(
                "SELECT DISTINCT tags FROM eval_metric_scores WHERE tags IS NOT NULL AND tags != ''",
                conn,
            )
            tag_set = set()
            for _, row in tags_df.iterrows():
                raw = str(row["tags"]).replace("，", ",").replace("、", ",")
                for tag in raw.split(","):
                    tag = tag.strip()
                    if tag:
                        tag_set.add(tag)
            return {
                "scenarios": scenarios,
                "difficulties": difficulties,
                "tags": sorted(tag_set),
                "templates": templates,
                "prompt_names": prompt_names,
            }

    def _compute_metric_averages(self, results: list):
        """Compute per-metric averages from result metric_scores."""
        metrics = [r.metric_score for r in results if hasattr(r, "metric_score") and r.metric_score]
        if not metrics:
            return {}
        def _avg(attr, as_float=True):
            vals = [getattr(m, attr) for m in metrics if getattr(m, attr) is not None]
            if not vals:
                return None
            return sum(float(v) for v in vals) / len(vals)
        return {
            "avg_intent_accuracy": _avg("intent_accuracy"),
            "avg_answer_relevance_score": _avg("answer_relevance_score"),
            "avg_task_completion": _avg("task_completion"),
            "avg_multi_turn_completion": _avg("multi_turn_completion"),
            "avg_hallucination_rate": _avg("hallucination"),
            "avg_retrieval_recall": _avg("retrieval_recall"),
            "avg_tool_success_rate": _avg("tool_success_rate"),
        }

    def _score_row(self, r: EvalResult):
        judge = r.judge_score
        return (
            r.batch_id,
            r.case.case_id,
            r.case.user_query,
            r.case.scenario_type,
            getattr(r.case, "template_id", ""),
            r.target.response_text,
            int(r.target.success),
            r.target.error,
            r.target.latency_ms,
            r.rule_score.score,
            json.dumps(r.rule_score.details, ensure_ascii=False),
            json.dumps(judge.scores if judge else {}, ensure_ascii=False),
            judge.average_score if judge else 0,
            judge.error if judge else "",
            int(r.is_bad_case),
            r.bad_case_type,
            r.root_cause,
            r.improvement_suggestion,
        )

    def _metric_row(self, r: EvalResult):
        metric = r.metric_score
        return (
            r.batch_id,
            r.case.case_id,
            r.case.scenario_type,
            getattr(r.case, "template_id", ""),
            getattr(r.case, "difficulty", ""),
            getattr(r.case, "tags", ""),
            metric.intent_accuracy,
            metric.answer_relevance_score,
            metric.task_completion,
            metric.multi_turn_completion,
            int(metric.hallucination) if metric.hallucination is not None else None,
            metric.retrieval_recall,
            metric.tool_success_rate,
            metric.expected_tool_coverage,
            json.dumps(metric.details, ensure_ascii=False),
        )


def _change_status(row) -> str:
    base_bad = _to_bool(row.get("is_bad_case_base"))
    target_bad = _to_bool(row.get("is_bad_case_target"))
    if base_bad and not target_bad:
        return "收敛"
    if not base_bad and target_bad:
        return "新增问题"
    if base_bad and target_bad:
        return "未收敛"
    return "稳定通过"


def _to_bool(value) -> bool:
    if pd.isna(value):
        return False
    return bool(int(value))
