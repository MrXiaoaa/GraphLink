from __future__ import annotations

import json
import re
from typing import Any


def _extract_json(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _normalize_table(value: Any) -> str:
    return str(value).strip().strip('`"\'').lower()


def _table_f1(predicted: list[Any], gold: list[Any]) -> tuple[float, float, float]:
    pred = {_normalize_table(x) for x in predicted if str(x).strip()}
    target = {_normalize_table(x) for x in gold if str(x).strip()}
    if not pred and not target:
        return 1.0, 1.0, 1.0
    if not pred or not target:
        return 0.0, 0.0, 0.0
    inter = len(pred & target)
    precision = inter / max(1, len(pred))
    recall = inter / max(1, len(target))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def _format_reward(obj: dict[str, Any]) -> float:
    if not isinstance(obj.get("selected_tables"), list):
        return -0.2
    if "sql" not in obj:
        return -0.1
    return 0.05


def compute_score(solution_str: str, ground_truth: dict[str, Any] | None = None, **kwargs: Any) -> float:
    """VERL custom reward for GraphLink policy training.

    Expected model output:
      {"selected_tables": [...], "sql": "..."}

    Reward emphasizes table-level policy quality and gives a small format/SQL
    presence bonus. It intentionally avoids requiring live DB credentials in the
    release package. Projects can extend this function with execution matching.
    """
    ground_truth = ground_truth or {}
    obj = _extract_json(solution_str)
    if not obj:
        return -0.5
    selected = obj.get("selected_tables") or []
    gold = ground_truth.get("gold_tables") or ground_truth.get("tables") or []
    precision, recall, f1 = _table_f1(selected, gold)
    sql = str(obj.get("sql") or "").strip()
    sql_bonus = 0.10 if sql else -0.10
    exact_bonus = 0.15 if precision == 1.0 and recall == 1.0 and gold else 0.0
    reward = 0.70 * f1 + 0.10 * precision + 0.10 * recall + sql_bonus + exact_bonus + _format_reward(obj)
    return float(max(-1.0, min(1.0, reward)))


# Some VERL versions call reward functions with `data_source`/`extra_info`.
def score(solution_str: str, ground_truth: dict[str, Any] | None = None, **kwargs: Any) -> float:
    return compute_score(solution_str, ground_truth=ground_truth, **kwargs)
