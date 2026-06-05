import time
from datetime import datetime
from uuid import uuid4

from pmeval.badcase_classifier import classify_bad_case
from pmeval.config import Settings
from pmeval.db import EvalDatabase
from pmeval.judge_evaluator import JudgeEvaluator
from pmeval.models import EvalCase, EvalResult
from pmeval.rule_evaluator import evaluate_rules
from pmeval.target_client import create_target_client
from pmeval.utils import setup_logging


class EvalRunner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = setup_logging()
        client_mode = settings.client_mode or ("mock" if settings.mock_mode else "real")
        self.client = create_target_client(client_mode, settings.api_url, settings.api_timeout)
        self.judge = JudgeEvaluator(settings.openai_api_key, settings.openai_base_url, settings.openai_model)
        self.db = EvalDatabase(settings.db_path)

    def run(self, cases: list[EvalCase], use_judge: bool = True) -> list[EvalResult]:
        batch_id = datetime.now().strftime("%Y%m%d%H%M%S") + "-" + uuid4().hex[:8]
        started_at = datetime.now().isoformat(timespec="seconds")
        self.db.upsert_cases(cases)
        results: list[EvalResult] = []

        for i, case in enumerate(cases):
            if i > 0:
                time.sleep(3.0)  # 避免连续请求打满后端导致超时
            self.logger.info("Evaluating case_id=%s", case.case_id)
            target = self.client.recommend(case)
            rule_score = evaluate_rules(case, target.response_text, target.success)
            judge_score = self.judge.evaluate(case, target.response_text) if use_judge and target.success else None
            is_bad, bad_type, root_cause, suggestion = classify_bad_case(case, target, rule_score, judge_score)
            results.append(
                EvalResult(
                    batch_id=batch_id,
                    case=case,
                    target=target,
                    rule_score=rule_score,
                    judge_score=judge_score,
                    is_bad_case=is_bad,
                    bad_case_type=bad_type,
                    root_cause=root_cause,
                    improvement_suggestion=suggestion,
                )
            )

        self.db.save_run(batch_id, started_at, results)
        return results
