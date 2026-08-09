# backend/eval/pipeline.py
"""
Builds a synchronous, single-turn version of app.py's /ask logic for eval.

Mirrors app.py rather than reusing its Flask route directly:
eval needs a plain function that returns the retrieved docs *and* the answer
in one call (app.py's /ask only streams the answer text to the client), and
it needs to run the same question through two different LLMs back to back.

If app.py's retrieval/generation logic changes, update this file (and
prompts.py) to match — otherwise eval silently stops measuring what's
actually deployed. Both files import SYSTEM_PROMPT/prompt from prompts.py
for exactly this reason; app.py should too.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_huggingface import HuggingFaceEmbeddings

from prompts import prompt

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 3))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", 0.3))
NO_CONTEXT_ANSWER = "Sorry, I couldn't find relevant information for your question."


@dataclass
class PipelineResult:
    answer: str
    contexts: list[str] = field(default_factory=list)
    qa_ids: list[int] = field(default_factory=list)
    gate_abstained: bool = False  # True if the relevance gate short-circuited before generation


def build_shared_vectorstore(embedding_model: HuggingFaceEmbeddings | None = None) -> AzureSearch:
    """One retrieval index, shared across every model under comparison.

    The eval compares generation LLMs on top of *the same* retrieval setup —
    if you actually want to compare retrieval configs (different embeddings,
    different k, a different index), build a separate vectorstore per config
    instead of sharing this one.
    """
    embedding_model = embedding_model or HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"),
        model_kwargs={"device": "cpu"},
    )
    return AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "learntrail-qa-index"),
        embedding_function=embedding_model.embed_query,
        search_type="hybrid",
    )


def build_eval_pipeline(llm, vectorstore: AzureSearch, k: int = RETRIEVAL_K,
                         relevance_threshold: float = RELEVANCE_THRESHOLD):
    """Returns a `run(question) -> PipelineResult` closure for one LLM.

    Mirrors app.py's /ask: same relevance gate (skip generation, return the
    canned "couldn't find relevant information" message, when nothing clears
    the threshold), same prompt, same context formatting. Intentionally
    skips the `general_queries` small-talk fast path and chat history — eval
    questions are all substantive single-turn queries, so neither applies.
    """
    chain = prompt | llm | StrOutputParser()

    def run(question: str) -> PipelineResult:
        results = vectorstore.similarity_search_with_relevance_scores(question, k=k)
        relevant_docs = [doc for doc, score in results if score >= relevance_threshold]

        if not relevant_docs:
            return PipelineResult(answer=NO_CONTEXT_ANSWER, gate_abstained=True)

        context_text = "\n\n".join(d.page_content for d in relevant_docs)
        answer = chain.invoke({"context": context_text, "history": [], "question": question})

        return PipelineResult(
            answer=answer,
            contexts=[d.page_content for d in relevant_docs],
            qa_ids=[d.metadata.get("qa_id") for d in relevant_docs if d.metadata.get("qa_id") is not None],
            gate_abstained=False,
        )

    return run