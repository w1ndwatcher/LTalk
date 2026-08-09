from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

EVAL_EXCEL_FILE = os.getenv("EVAL_EXCEL_FILE", "learntrail_eval_dataset.xlsx")
EVAL_SHEET_NAME = os.getenv("EVAL_SHEET_NAME", "Eval_Set")

GAP_SENTINEL = "GAP"


@dataclass
class EvalRow:
    eval_id: str
    question: str
    ground_truth_answer: str
    expected_qa_ids: list[int] | None  # None means this is a kb_gap row
    category: str
    test_type: str
    site_source: str
    notes: str = ""

    @property
    def is_gap(self) -> bool:
        return self.expected_qa_ids is None


def _parse_expected_ids(raw) -> list[int] | None:
    text = str(raw).strip()
    if text.upper() == GAP_SENTINEL:
        return None
    ids = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    if not ids:
        raise ValueError(f"Could not parse expected_source_qa_ids value: {raw!r}")
    return ids


def load_eval_rows(path: str = EVAL_EXCEL_FILE, sheet_name: str = EVAL_SHEET_NAME) -> list[EvalRow]:
    df = pd.read_excel(path, sheet_name=sheet_name)
    required = {"eval_id", "question", "ground_truth_answer", "expected_source_qa_ids",
                "category", "test_type", "site_source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Eval sheet is missing expected columns: {sorted(missing)}")

    rows: list[EvalRow] = []
    for _, r in df.iterrows():
        rows.append(EvalRow(
            eval_id=str(r["eval_id"]).strip(),
            question=str(r["question"]).strip(),
            ground_truth_answer=str(r["ground_truth_answer"]).strip(),
            expected_qa_ids=_parse_expected_ids(r["expected_source_qa_ids"]),
            category=str(r.get("category", "")).strip(),
            test_type=str(r.get("test_type", "")).strip(),
            site_source=str(r.get("site_source", "")).strip(),
            notes=str(r.get("notes", "") or "").strip(),
        ))
    return rows


if __name__ == "__main__":
    rows = load_eval_rows()
    gap = [r for r in rows if r.is_gap]
    print(f"Loaded {len(rows)} eval rows ({len(gap)} kb_gap, {len(rows) - len(gap)} standard).")