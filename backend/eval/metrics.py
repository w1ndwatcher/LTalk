# backend/eval/metrics.py
"""
Two metric families that RAGAS doesn't cover:

1. ID-based context precision/recall — a direct, deterministic check of
   "did the retriever return QA row #13" against `expected_source_qa_ids`,
   using the qa_id metadata tagged in ingest.py. This is a literal row-ID
   overlap, not a semantic judgment — a useful cross-check alongside
   RAGAS's LLM-judged context_precision/context_recall, which compare
   *text similarity* to the reference answer rather than exact source rows.

2. Abstention accuracy — for kb_gap rows only. There is no reference chunk
   for these (nothing in the 70-pair KB answers them), so RAGAS's
   context_precision/context_recall/faithfulness don't apply — grading them
   against a nonexistent reference wouldn't measure anything real. What *is*
   measurable per row: did the system correctly avoid asserting confident,
   specific claims it can't support, given what it actually retrieved?

   Two tiers, cheapest first:
     - gate_abstained: the relevance gate rejected before generation ever
       ran (retrieval score below RELEVANCE_THRESHOLD) — free, deterministic,
       no LLM call needed.
     - LLM judge: only invoked when generation *did* run. Grades the answer
       against the context that was actually retrieved (not against the
       eval row's ground_truth_answer) — the question is "is this answer
       grounded in what got retrieved", not "does it happen to match the
       real answer". A model that (accidentally, from parametric knowledge
       rather than retrieval) states the right fact anyway still counts as
       HALLUCINATED here, because that ungroundedness is exactly the
       failure mode this check exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def id_context_precision(retrieved_ids: list[int], expected_ids: list[int]) -> float:
    if not retrieved_ids:
        return 0.0
    hits = len(set(retrieved_ids) & set(expected_ids))
    return hits / len(retrieved_ids)


def id_context_recall(retrieved_ids: list[int], expected_ids: list[int]) -> float:
    if not expected_ids:
        return None  # undefined; shouldn't be called on gap rows
    hits = len(set(retrieved_ids) & set(expected_ids))
    return hits / len(expected_ids)


_ABSTENTION_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are grading whether a RAG chatbot's answer stayed grounded in the "
     "context it was actually given, for a question that context does NOT "
     "actually answer. Judge ONLY against the provided CONTEXT — not against "
     "your own knowledge of the real answer.\n\n"
     "Respond with exactly one word: ABSTAINED or HALLUCINATED.\n\n"
     "ABSTAINED: the answer acknowledges it doesn't have specific information "
     "on this, declines to guess, and/or redirects the user (e.g. to contact "
     "support) — even if phrased confidently or without literally saying "
     "'I don't know'.\n"
     "HALLUCINATED: the answer states specific, confident facts, figures, or "
     "claims that are not present in the CONTEXT below."),
    ("human",
     "QUESTION:\n{question}\n\nCONTEXT RETRIEVED BY THE SYSTEM:\n{context}\n\n"
     "SYSTEM'S ANSWER:\n{answer}\n\nVerdict (ABSTAINED or HALLUCINATED):"),
])


@dataclass
class AbstentionVerdict:
    abstained: bool
    method: str  # "gate" or "llm_judge"
    raw_verdict: str | None = None  # the judge's literal output, when used


def judge_abstention(judge_llm, question: str, contexts: list[str], answer: str,
                      gate_abstained: bool) -> AbstentionVerdict:
    if gate_abstained:
        return AbstentionVerdict(abstained=True, method="gate")

    context_text = "\n\n".join(contexts) if contexts else "(no context was retrieved)"
    chain = _ABSTENTION_JUDGE_PROMPT | judge_llm | StrOutputParser()
    verdict = chain.invoke({"question": question, "context": context_text, "answer": answer}).strip().upper()

    if "ABSTAIN" in verdict:
        abstained = True
    elif "HALLUCINAT" in verdict:
        abstained = False
    else:
        # Judge didn't follow the format — fail closed (count as hallucinated)
        # rather than silently mis-scoring a row as a pass.
        abstained = False

    return AbstentionVerdict(abstained=abstained, method="llm_judge", raw_verdict=verdict)