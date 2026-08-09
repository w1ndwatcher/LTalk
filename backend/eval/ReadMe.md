# LearnTrail RAG eval harness

Compares two generation LLMs on the same retrieval index against
`learntrail_eval_dataset.xlsx`, using RAGAS for rows that have a real
source in the 70-pair KB, and a separate abstention-accuracy check for the
`kb_gap` rows that don't.

## Before running

1. **Re-ingest with the updated `ingest.py`** so every chunk carries a
   `qa_id` in its metadata (needed for the id-based context metrics):
   ```bash
   cd backend
   python ingest.py --recreate-index
   ```
   `--recreate-index` drops and rebuilds the Azure AI Search index so the
   new field is actually part of the schema — see the warning in
   `ingest.py`'s docstring for why re-running without it would duplicate
   every row instead of updating it.

2. **Put `learntrail_eval_dataset.xlsx` next to `learntrail_qa.xlsx`**
   (or set `EVAL_EXCEL_FILE` to its path).

3. **Add to `.env`** (on top of the existing Azure Search / Groq / HF vars):
   ```
   AZURE_OPENAI_ENDPOINT=...
   AZURE_OPENAI_API_KEY=...
   AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
   AZURE_OPENAI_API_VERSION=2024-10-21
   # Optional — only if you want the judge on a different deployment
   # than the azure-gpt-4o-mini model under test:
   # JUDGE_AZURE_OPENAI_ENDPOINT=...
   # JUDGE_AZURE_OPENAI_API_KEY=...
   # JUDGE_AZURE_OPENAI_DEPLOYMENT=...
   ```

4. **Install the extra dependencies:**
   ```bash
   pip install ragas datasets langchain-openai
   ```

## Running

```bash
cd backend
python -m eval.run_eval --sample 8      # smoke-test on 8 rows first — RAGAS calls add up
python -m eval.run_eval                 # full run over all 35 eval rows
```

Results land in `eval/results/<timestamp>/`:
- `per_row_results.csv` — every question, both models' answers, retrieved
  qa_ids, and every metric, one row per (model, question) pair
- `comparison_overall.csv` — the aggregate comparison table
- `comparison_by_test_type.csv` — broken down by paraphrase / multi_hop /
  distractor_precision / consistency_check / kb_gap

## Reading the results

- `ragas_faithfulness`, `ragas_answer_relevancy`, `ragas_context_precision`,
  `ragas_context_recall` — standard RAGAS, computed only over rows with a
  real expected source (i.e. not `kb_gap`).
- `id_context_precision`, `id_context_recall` — deterministic row-ID overlap
  between what was retrieved and `expected_source_qa_ids`. This is the
  literal "did it return row #13" check; it can disagree with RAGAS's
  semantic `context_precision`/`context_recall` (e.g. the retriever pulls a
  *different* row that happens to contain similar wording) — that
  disagreement is itself informative, not a bug.
- `abstention_accuracy` — fraction of `kb_gap` rows where the system
  correctly avoided asserting unsupported specifics (via the relevance gate
  or, failing that, the LLM judge). Only meaningful on the `kb_gap` subset;
  it's `None`/blank everywhere else by design, not averaged into the
  RAGAS columns.