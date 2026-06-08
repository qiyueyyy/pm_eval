import json
import logging
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd

from pmeval.scenario_router import resolve_template_id


LOGGER_NAME = "pm_eval"


def setup_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def safe_json_loads(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def ensure_dirs(root: Path) -> None:
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "exports").mkdir(parents=True, exist_ok=True)


def read_cases_csv(file_or_path: Any) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)
    required = [
        "case_id",
        "user_query",
        "scenario_type",
        "expected_behavior",
        "constraints_json",
        "difficulty",
        "tags",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要字段: {', '.join(missing)}")
    if "template_id" not in df.columns:
        df["template_id"] = ""
    cols = [
        "case_id",
        "user_query",
        "scenario_type",
        "template_id",
        "expected_behavior",
        "constraints_json",
        "difficulty",
        "tags",
    ]
    normalized = df[cols].fillna("").astype(str)
    normalized["template_id"] = normalized.apply(
        lambda row: resolve_template_id(row["scenario_type"], row["template_id"]),
        axis=1,
    )
    return normalized


def dataframe_to_cases(df: pd.DataFrame):
    from pmeval.models import EvalCase

    init_fields = {field.name for field in fields(EvalCase) if field.init}
    cases = []
    for _, row in df.iterrows():
        raw = row.to_dict()
        case = EvalCase(**{key: value for key, value in raw.items() if key in init_fields})
        # Streamlit may keep an older imported EvalCase class alive across edits.
        # Attach template_id after construction so mixed-template CSVs still work.
        if "template_id" in raw:
            case.template_id = str(raw.get("template_id") or "")
        cases.append(case)
    return cases
