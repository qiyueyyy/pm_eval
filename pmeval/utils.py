import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


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
    return df[required].fillna("")


def dataframe_to_cases(df: pd.DataFrame):
    from pmeval.models import EvalCase

    return [EvalCase(**row.to_dict()) for _, row in df.iterrows()]
