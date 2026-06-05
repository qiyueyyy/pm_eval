import json
import os
import re
from typing import Any

from openai import OpenAI

from pmeval.models import EvalCase, JudgeScore
from pmeval.utils import safe_json_loads


SCORE_FIELDS = [
    "need_understanding_score",
    "constraint_satisfaction_score",
    "relevance_score",
    "usefulness_score",
    "faithfulness_score",
    "clarity_score",
]

REQUIRED_FIELDS = SCORE_FIELDS + [
    "is_bad_case",
    "bad_case_type",
    "root_cause",
    "improvement_suggestion",
]

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


class JudgeEvaluator:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key else None

    def evaluate(self, case: EvalCase, answer: str) -> JudgeScore:
        if not self.client:
            return JudgeScore({}, 0, error="OPENAI_API_KEY 未配置，跳过 LLM Judge")

        prompt = self._build_prompt(case, answer)
        try:
            completion = self._create_completion(prompt)
            content = completion.choices[0].message.content or "{}"
            data = self._normalize_judge_json(self._parse_json(content))
            scores = {field: float(data[field]) for field in SCORE_FIELDS}
            average = sum(scores.values()) / len(SCORE_FIELDS)
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
                "content": (
                    "你是严格的 AI 产品推荐评测官。必须只输出一个合法 JSON 对象，"
                    "不要输出 Markdown、解释文字或代码块。"
                ),
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

    def _build_prompt(self, case: EvalCase, answer: str) -> str:
        constraints = safe_json_loads(case.constraints_json)
        return json.dumps(
            {
                "任务": "评估目标服务的文本推荐回答质量。",
                "硬性要求": [
                    "只输出 JSON 对象。",
                    "不要输出 Markdown。",
                    "所有 score 字段必须是 1-5 的数字。",
                    "is_bad_case 必须是 boolean。",
                    "bad_case_type 必须从给定枚举中选择。",
                    "root_cause 必须从给定枚举中选择。",
                ],
                "输出字段": {
                    "need_understanding_score": "需求理解，1-5 number",
                    "constraint_satisfaction_score": "约束满足，1-5 number",
                    "relevance_score": "推荐相关性，1-5 number",
                    "usefulness_score": "回答可用性，1-5 number",
                    "faithfulness_score": "事实可信度，1-5 number",
                    "clarity_score": "表达清晰度，1-5 number",
                    "is_bad_case": "boolean",
                    "bad_case_type": BAD_CASE_TYPES,
                    "root_cause": ROOT_CAUSES,
                    "improvement_suggestion": "中文短句，说明最优先改进动作",
                },
                "Bad Case 判定": "任一关键维度低于 3.5，或回答不满足用户明确约束，或存在幻觉/空泛/格式问题，则 is_bad_case=true。",
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

    def _normalize_judge_json(self, data: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(data)
        for field in SCORE_FIELDS:
            normalized[field] = self._coerce_score(normalized.get(field))
        normalized["is_bad_case"] = self._coerce_bool(normalized.get("is_bad_case"))
        normalized["bad_case_type"] = self._choice_or_default(
            normalized.get("bad_case_type"),
            BAD_CASE_TYPES,
            "无问题" if not normalized["is_bad_case"] else "回答太泛",
        )
        normalized["root_cause"] = self._choice_or_default(
            normalized.get("root_cause"),
            ROOT_CAUSES,
            "模型问题",
        )
        normalized["improvement_suggestion"] = str(
            normalized.get("improvement_suggestion") or "结合低分维度优化 Prompt、检索和业务规则。"
        )
        missing = [field for field in REQUIRED_FIELDS if field not in normalized]
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

    def _choice_or_default(self, value: Any, choices: list[str], default: str) -> str:
        text = str(value or "").strip()
        return text if text in choices else default
