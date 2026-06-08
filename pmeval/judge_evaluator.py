import json
import os
import re
from typing import Any

from openai import OpenAI

from pmeval.eval_template import EvalTemplate, load_template
from pmeval.models import EvalCase, JudgeScore
from pmeval.utils import safe_json_loads


class JudgeEvaluator:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "qwen3.6-plus")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key else None

    def evaluate(self, case: EvalCase, answer: str, template: EvalTemplate | None = None) -> JudgeScore:
        selected_template = template or load_template(None)
        if not self.client:
            return JudgeScore({}, 0, error="OPENAI_API_KEY 未配置，跳过 LLM Judge")

        prompt = self._build_prompt(case, answer, selected_template)
        try:
            completion = self._create_completion(prompt)
            content = completion.choices[0].message.content or "{}"
            data = self._normalize_judge_json(self._parse_json(content), selected_template)
            score_fields = [field.id for field in selected_template.judge_fields]
            scores = {field: float(data[field]) for field in score_fields}
            average = sum(scores.values()) / len(score_fields) if score_fields else 0
            return JudgeScore(
                scores=scores,
                average_score=round(average, 2),
                comment=str(data.get("improvement_suggestion", "")),
                is_bad_case=bool(data["is_bad_case"]),
                bad_case_type=str(data["bad_case_type"]),
                root_cause=str(data["root_cause"]),
                improvement_suggestion=str(data["improvement_suggestion"]),
                raw_response=data,
            )
        except Exception as exc:
            return JudgeScore({}, 0, error=str(exc))

    def _create_completion(self, prompt: str):
        messages = [
            {
                "role": "system",
                "content": "你是严格的 AI 产品评测官。必须只输出一个合法 JSON 对象，不要输出 Markdown、解释文字或代码块。",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            if "response_format" not in str(exc):
                raise
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )

    def _build_prompt(self, case: EvalCase, answer: str, template: EvalTemplate) -> str:
        constraints = safe_json_loads(case.constraints_json)
        score_fields = {
            field.id: f"{field.name}，{field.description}，1-5 number" for field in template.judge_fields
        }
        return json.dumps(
            {
                "任务": "评估目标服务的文本回答质量。",
                "评测场景": template.name,
                "场景说明": template.description,
                "硬性要求": [
                    "只输出 JSON 对象。",
                    "不要输出 Markdown。",
                    "所有 score 字段必须是 1-5 的数字。",
                    "is_bad_case 必须是 boolean。",
                    "bad_case_type 必须从给定枚举中选择。",
                    "root_cause 必须从给定枚举中选择。",
                ],
                "输出字段": {
                    **score_fields,
                    "intent_match": "boolean or null",
                    "answer_relevance_score": "1-5 number",
                    "task_completion_score": "1-5 number",
                    "multi_turn_completion_score": "1-5 number or null",
                    "hallucination": "boolean",
                    "hallucination_type": "none | fabricated_price | fabricated_policy | fabricated_product | unsupported_claim | other",
                    "is_bad_case": "boolean",
                    "bad_case_type": template.bad_case.types,
                    "root_cause": template.bad_case.root_causes,
                    "improvement_suggestion": "中文短句，说明最优先改进动作",
                },
                "Bad Case 判定": (
                    f"任一关键维度低于 {template.bad_case.judge_average_lt}，或回答不满足用户明确约束，"
                    "或存在幻觉、空泛、格式、合规、工具调用等问题，则 is_bad_case=true。"
                ),
                "case_id": case.case_id,
                "用户问题": case.user_query,
                "场景": case.scenario_type,
                "期望行为": case.expected_behavior,
                "约束": constraints,
                "被测回答": answer,
            },
            ensure_ascii=False,
        )

    def _parse_json(self, content: str) -> dict[str, Any]:
        try:
            value = json.loads(content)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Judge 输出不是 JSON: {content[:300]}")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Judge JSON 解析失败: {exc}; raw={content[:300]}") from exc
        if not isinstance(value, dict):
            raise ValueError("Judge JSON 顶层必须是对象")
        return value

    def _normalize_judge_json(self, data: dict[str, Any], template: EvalTemplate) -> dict[str, Any]:
        normalized = dict(data)
        score_fields = [field.id for field in template.judge_fields]
        required_fields = score_fields + ["is_bad_case", "bad_case_type", "root_cause", "improvement_suggestion"]
        for field in score_fields:
            normalized[field] = self._coerce_score(normalized.get(field))
        normalized["intent_match"] = self._coerce_optional_bool(normalized.get("intent_match"))
        normalized["answer_relevance_score"] = self._coerce_optional_score(normalized.get("answer_relevance_score"))
        normalized["task_completion_score"] = self._coerce_optional_score(normalized.get("task_completion_score"))
        normalized["multi_turn_completion_score"] = self._coerce_optional_score(normalized.get("multi_turn_completion_score"))
        normalized["hallucination"] = self._coerce_optional_bool(normalized.get("hallucination"))
        normalized["hallucination_type"] = str(normalized.get("hallucination_type") or "none")
        normalized["is_bad_case"] = self._coerce_bool(normalized.get("is_bad_case"))
        normalized["bad_case_type"] = self._choice_or_default(
            normalized.get("bad_case_type"),
            template.bad_case.types,
            template.bad_case.no_problem_type if not normalized["is_bad_case"] else template.bad_case.default_type,
        )
        normalized["root_cause"] = self._choice_or_default(
            normalized.get("root_cause"),
            template.bad_case.root_causes,
            template.bad_case.default_root_cause,
        )
        normalized["improvement_suggestion"] = str(
            normalized.get("improvement_suggestion") or "结合低分维度优化 Prompt、检索和业务规则。"
        )
        missing = [field for field in required_fields if field not in normalized]
        if missing:
            raise ValueError(f"Judge JSON 缺少字段: {', '.join(missing)}")
        return normalized

    def _coerce_score(self, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = 0.0
        return max(1.0, min(5.0, score)) if score else 0.0

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y", "是"}
        return bool(value)

    def _coerce_optional_score(self, value: Any) -> float | None:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        return max(1.0, min(5.0, score)) if score else None

    def _coerce_optional_bool(self, value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "是", "鏄?"}:
                return True
            if lowered in {"false", "0", "no", "n", "否"}:
                return False
            return None
        return bool(value)

    def _choice_or_default(self, value: Any, choices: list[str], default: str) -> str:
        text = str(value or "").strip()
        return text if text in choices else default
