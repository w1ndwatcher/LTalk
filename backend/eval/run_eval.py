# backend/eval/run_eval.py
"""
Runs every question in the eval dataset through each configured model and
produces a side-by-side comparison table.

Usage:
    cd backend
    python -m eval.run_eval
    python -m eval.run_eval --sample 8          # smoke-test on 8 rows first
    python -m eval.run_eval --output-dir results/2026-08-09

What gets scored, and how:
    - Rows with a real `expected_source_qa_ids` (paraphrase / multi_hop /
      distractor_precision / consistency_check): RAGAS faithfulness,
      answer_relevancy, context_precision, context_recall, PLUS the
      deterministic id_context_precision / id_context_recall from metrics.py.
    - kb_gap rows: none of the above apply (there's no reference chunk to
      grade against) — instead, abstention_accuracy via metrics.judge_abstention.
    These two families are reported separately and never averaged together;
    see metrics.py's module docstring for why.

Requires (backend/requirements.txt additions):
    ragas>=0.2,<0.3
    datasets
    langchain-openai
The exact `ragas.metrics` import names have moved around between versions;
this script uses the long-stable lowercase singleton imports
(faithfulness, answer_relevancy, context_precision, context_recall). If your
installed ragas version has renamed these, check `ragas.metrics.__all__`
and adjust the import at the top of run_ragas().
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo's backend/ on path

from eval.eval_dataset import load_eval_rows, EvalRow
from eval.pipeline import build_shared_vectorstore, build_eval_pipeline, PipelineResult
from eval.models_config import MODEL_CONFIGS, build_judge_llm
from eval.metrics import id_context_precision, id_context_recall, judge_abstention

load_dotenv()

_ragas_embedding_model = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"),
    model_kwargs={"device": "cpu"},
)

def run_model_over_eval_set(model_name: str, llm, vectorstore, rows: list[EvalRow]) -> list[dict]:
    """Runs one model over every eval row; returns one result dict per row."""
    run_fn = build_eval_pipeline(llm, vectorstore)
    records = []
    for row in rows:
        result: PipelineResult = run_fn(row.question)
        records.append({
            "model": model_name,
            "eval_id": row.eval_id,
            "question": row.question,
            "category": row.category,
            "test_type": row.test_type,
            "ground_truth_answer": row.ground_truth_answer,
            "expected_qa_ids": row.expected_qa_ids,  # None for gap rows
            "is_gap": row.is_gap,
            "generated_answer": result.answer,
            "retrieved_contexts": result.contexts,
            "retrieved_qa_ids": result.qa_ids,
            "gate_abstained": result.gate_abstained,
        })
        print(f"  [{model_name}] {row.eval_id} ({row.test_type}) done")
    return records


def add_id_based_metrics(records: list[dict]) -> None:
    """Mutates records in place, adding id_context_precision/id_context_recall
    for non-gap rows (left as None for gap rows, where they're undefined)."""
    for rec in records:
        if rec["is_gap"]:
            rec["id_context_precision"] = None
            rec["id_context_recall"] = None
        else:
            rec["id_context_precision"] = id_context_precision(rec["retrieved_qa_ids"], rec["expected_qa_ids"])
            rec["id_context_recall"] = id_context_recall(rec["retrieved_qa_ids"], rec["expected_qa_ids"])


def add_abstention_scores(records: list[dict], judge_llm) -> None:
    """Mutates records in place; only meaningful (and only computed) for gap rows."""
    for rec in records:
        if not rec["is_gap"]:
            rec["abstained"] = None
            rec["abstention_method"] = None
            continue
        verdict = judge_abstention(
            judge_llm,
            question=rec["question"],
            contexts=rec["retrieved_contexts"],
            answer=rec["generated_answer"],
            gate_abstained=rec["gate_abstained"],
        )
        rec["abstained"] = verdict.abstained
        rec["abstention_method"] = verdict.method


def run_ragas(records: list[dict], judge_llm, embedding_model) -> pd.DataFrame | None:
    """Scores the non-gap rows with RAGAS. Returns a per-row score DataFrame
    (one row per input record, same order) or None if there are no non-gap
    rows to score."""
    standard_records = [r for r in records if not r["is_gap"]]
    if not standard_records:
        return None

    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

    try:
        # ragas >= 0.2 schema
        from ragas import EvaluationDataset
        dataset = EvaluationDataset.from_list([
            {
                "user_input": r["question"],
                "response": r["generated_answer"],
                "retrieved_contexts": r["retrieved_contexts"] or ["(nothing retrieved)"],
                "reference": r["ground_truth_answer"],
            }
            for r in standard_records
        ])
    except ImportError:
        # legacy (<0.2) schema
        from datasets import Dataset as HFDataset
        dataset = HFDataset.from_dict({
            "question": [r["question"] for r in standard_records],
            "answer": [r["generated_answer"] for r in standard_records],
            "contexts": [r["retrieved_contexts"] or ["(nothing retrieved)"] for r in standard_records],
            "ground_truth": [r["ground_truth_answer"] for r in standard_records],
        })

    from ragas import evaluate
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    scores_df = result.to_pandas()
    scores_df.insert(0, "eval_id", [r["eval_id"] for r in standard_records])
    return scores_df


def build_comparison_tables(all_records: list[dict], ragas_scores_by_model: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (overall_table, by_test_type_table), both indexed by model."""
    df = pd.DataFrame(all_records)

    overall_rows = []
    by_type_rows = []

    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]
        ragas_df = ragas_scores_by_model.get(model_name)

        gap_df = model_df[model_df["is_gap"]]
        std_df = model_df[~model_df["is_gap"]]

        row = {
            "model": model_name,
            "id_context_precision": std_df["id_context_precision"].mean() if len(std_df) else None,
            "id_context_recall": std_df["id_context_recall"].mean() if len(std_df) else None,
            "abstention_accuracy": gap_df["abstained"].mean() if len(gap_df) else None,
            "n_standard_rows": len(std_df),
            "n_gap_rows": len(gap_df),
        }
        if ragas_df is not None:
            for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
                if metric in ragas_df.columns:
                    row[f"ragas_{metric}"] = ragas_df[metric].mean()
        overall_rows.append(row)

        # Per test_type breakdown (id-based metrics only, for a quick slice
        # without re-merging the RAGAS per-row scores — see the saved CSV
        # for the full per-row RAGAS breakdown by test_type).
        for test_type in model_df["test_type"].unique():
            tt_df = model_df[model_df["test_type"] == test_type]
            tt_std = tt_df[~tt_df["is_gap"]]
            tt_gap = tt_df[tt_df["is_gap"]]
            by_type_rows.append({
                "model": model_name,
                "test_type": test_type,
                "n_rows": len(tt_df),
                "id_context_precision": tt_std["id_context_precision"].mean() if len(tt_std) else None,
                "id_context_recall": tt_std["id_context_recall"].mean() if len(tt_std) else None,
                "abstention_accuracy": tt_gap["abstained"].mean() if len(tt_gap) else None,
            })

    overall_table = pd.DataFrame(overall_rows).set_index("model")
    by_type_table = pd.DataFrame(by_type_rows).set_index(["model", "test_type"])
    return overall_table, by_type_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None,
                         help="Only run the first N eval rows (smoke-test before a full, costlier run).")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Where to save per-row CSV + comparison tables. "
                              "Defaults to eval/results/<timestamp>/.")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(output_dir, exist_ok=True)

    rows = load_eval_rows()
    if args.sample:
        rows = rows[: args.sample]
    print(f"Loaded {len(rows)} eval rows ({sum(r.is_gap for r in rows)} kb_gap).")

    print("Connecting to shared retrieval index...")
    vectorstore = build_shared_vectorstore()

    judge_llm = build_judge_llm()

    all_records: list[dict] = []
    ragas_scores_by_model: dict[str, pd.DataFrame] = {}

    for config in MODEL_CONFIGS:
        model_name = config["name"]
        print(f"\n=== Running model: {model_name} ===")
        llm = config["build_llm"]()

        records = run_model_over_eval_set(model_name, llm, vectorstore, rows)
        add_id_based_metrics(records)
        add_abstention_scores(records, judge_llm)

        print(f"  Scoring {model_name} with RAGAS...")
        ragas_df = run_ragas(records, judge_llm, embedding_model=_ragas_embedding_model)
        # ragas_df = run_ragas(records, judge_llm, embedding_model=vectorstore.embeddings
        #                       if hasattr(vectorstore, "embeddings") else None)
        if ragas_df is not None:
            ragas_scores_by_model[model_name] = ragas_df
            # merge per-row ragas scores back onto the matching records
            ragas_by_id = ragas_df.set_index("eval_id")
            for rec in records:
                if rec["eval_id"] in ragas_by_id.index:
                    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
                        if metric in ragas_by_id.columns:
                            rec[f"ragas_{metric}"] = ragas_by_id.loc[rec["eval_id"], metric]

        all_records.extend(records)

    detail_df = pd.DataFrame(all_records)
    detail_path = os.path.join(output_dir, "per_row_results.csv")
    detail_df.to_csv(detail_path, index=False)
    print(f"\nSaved per-row results: {detail_path}")

    overall_table, by_type_table = build_comparison_tables(all_records, ragas_scores_by_model)

    overall_path = os.path.join(output_dir, "comparison_overall.csv")
    by_type_path = os.path.join(output_dir, "comparison_by_test_type.csv")
    overall_table.to_csv(overall_path)
    by_type_table.to_csv(by_type_path)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n=== Overall comparison ===")
    print(overall_table.round(3))
    print(f"\nSaved: {overall_path}")

    print("\n=== By test_type ===")
    print(by_type_table.round(3))
    print(f"\nSaved: {by_type_path}")


if __name__ == "__main__":
    main()