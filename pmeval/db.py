import json
import sqlite3
from pathlib import Path

from pmeval.models import EvalCase, EvalResult


SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_cases (
  case_id TEXT PRIMARY KEY,
  user_query TEXT,
  scenario_type TEXT,
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
  bad_case_count INTEGER,
  avg_latency_ms REAL
);
CREATE TABLE IF NOT EXISTS eval_scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id TEXT,
  case_id TEXT,
  user_query TEXT,
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

    def upsert_cases(self, cases: list[EvalCase]) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO eval_cases
                (case_id, user_query, scenario_type, expected_behavior, constraints_json, difficulty, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.case_id,
                        c.user_query,
                        c.scenario_type,
                        c.expected_behavior,
                        c.constraints_json,
                        c.difficulty,
                        c.tags,
                    )
                    for c in cases
                ],
            )

    def save_run(self, batch_id: str, started_at: str, results: list[EvalResult]) -> None:
        total = len(results)
        success_count = sum(1 for r in results if r.target.success)
        avg_rule = sum(r.rule_score.score for r in results) / total if total else 0
        bad_count = sum(1 for r in results if r.is_bad_case)
        avg_latency = sum(r.target.latency_ms for r in results) / total if total else 0
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO eval_runs
                (batch_id, started_at, total_cases, success_count, avg_rule_score, bad_case_count, avg_latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (batch_id, started_at, total, success_count, avg_rule, bad_count, avg_latency),
            )
            conn.executemany(
                """
                INSERT INTO eval_scores
                (batch_id, case_id, user_query, response_text, success, error, latency_ms, rule_score,
                 rule_details_json, judge_scores_json, judge_average, judge_error, is_bad_case)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def _score_row(self, r: EvalResult):
        judge = r.judge_score
        return (
            r.batch_id,
            r.case.case_id,
            r.case.user_query,
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
        )
