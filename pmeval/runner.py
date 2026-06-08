import time
from datetime import datetime
from uuid import uuid4

from pmeval.badcase_classifier import classify_bad_case
from pmeval.config import Settings
from pmeval.db import EvalDatabase
from pmeval.eval_template import EvalTemplate, load_template
from pmeval.judge_evaluator import JudgeEvaluator
from pmeval.metric_evaluator import evaluate_metrics
from pmeval.models import EvalCase, EvalResult, TargetResult
from pmeval.rule_evaluator import evaluate_rules
from pmeval.scenario_router import is_mixed_template, resolve_template_id
from pmeval.target_client import create_target_client
from pmeval.utils import setup_logging


class EvalRunner:
    def __init__(self, settings: Settings, template: EvalTemplate | None = None):
        self.settings = settings
        self.template = template or load_template(settings.template_id)
        self.logger = setup_logging()
        client_mode = settings.client_mode or ("mock" if settings.mock_mode else "real")
        self.client = create_target_client(
            client_mode,
            settings.api_url,
            settings.api_timeout,
            prompt_name=settings.prompt_name,
            prompt_text=settings.prompt_text,
        )
        self.judge = JudgeEvaluator(settings.openai_api_key, settings.openai_base_url, settings.openai_model)
        self.db = EvalDatabase(settings.db_path)

    def run(
        self,
        cases: list[EvalCase],
        use_judge: bool = True,
        should_stop=None,
        on_progress=None,
        batch_id: str | None = None,
        started_at: str | None = None,
        initial_results: list[EvalResult] | None = None,
        total_cases: int | None = None,
    ) -> list[EvalResult]:
        batch_id = batch_id or datetime.now().strftime("%Y%m%d%H%M%S") + "-" + uuid4().hex[:8]
        started_at = started_at or datetime.now().isoformat(timespec="seconds")
        self.db.upsert_cases(cases)
        results: list[EvalResult] = list(initial_results or [])
        total = total_cases or len(cases)

        for i, case in enumerate(cases):
            if should_stop and should_stop():
                break
            if i > 0:
                self._stoppable_sleep(3.0, should_stop)
                if should_stop and should_stop():
                    break
            result = self._evaluate_one(batch_id, case, use_judge)
            results.append(result)
            if on_progress:
                on_progress(len(results), total, result, batch_id, started_at, False)

        if results:
            result_template_ids = [
                resolve_template_id(result.case.scenario_type, getattr(result.case, "template_id", ""), self.template.id)
                for result in results
            ]
            self.db.save_run(
                batch_id,
                started_at,
                results,
                template_id="mixed" if is_mixed_template(result_template_ids) else (result_template_ids[0] if result_template_ids else self.template.id),
                target_name=self.settings.target_name,
                prompt_name=self.settings.prompt_name,
                prompt_text=self.settings.prompt_text,
            )
        if on_progress:
            on_progress(len(results), total, None, batch_id, started_at, bool(should_stop and should_stop()))
        return results

    def _evaluate_one(self, batch_id: str, case: EvalCase, use_judge: bool) -> EvalResult:
        self.logger.info("Evaluating case_id=%s", case.case_id)
        template = self._template_for_case(case)
        case.template_id = template.id
        target: TargetResult = self.client.recommend(case)
        rule_score = evaluate_rules(case, target.response_text, target.success, template)
        judge_score = self.judge.evaluate(case, target.response_text, template) if use_judge and target.success else None
        metric_score = evaluate_metrics(case, target, rule_score, judge_score, template)
        is_bad, bad_type, root_cause, suggestion = classify_bad_case(
            case,
            target,
            rule_score,
            judge_score,
            template,
            metric_score,
        )
        return EvalResult(
            batch_id=batch_id,
            case=case,
            target=target,
            rule_score=rule_score,
            judge_score=judge_score,
            is_bad_case=is_bad,
            bad_case_type=bad_type,
            root_cause=root_cause,
            improvement_suggestion=suggestion,
            metric_score=metric_score,
        )

    def _template_for_case(self, case: EvalCase) -> EvalTemplate:
        template_id = resolve_template_id(case.scenario_type, getattr(case, "template_id", ""), self.template.id)
        if template_id == self.template.id:
            return self.template
        return load_template(template_id)

    def _stoppable_sleep(self, seconds: float, should_stop=None) -> None:
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            if should_stop and should_stop():
                return
            time.sleep(min(0.1, deadline - time.perf_counter()))
