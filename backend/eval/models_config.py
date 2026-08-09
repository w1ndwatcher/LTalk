# backend/eval/models_config.py
"""
Defines what "both models" means for this comparison.

Candidates:
  - groq-<LLM_MODEL>: whatever model LLM_MODEL is currently set to in .env
    (the label is derived from the env var, not hardcoded, so it can't go
    stale again the way "groq-llama-4-scout" did after that model was
    deprecated by Groq and LLM_MODEL was updated but this label wasn't).
  - azure-<AZURE_OPENAI_DEPLOYMENT>: same idea for the Azure candidate.

Judge: deliberately a THIRD, independent model — llama-3.3-70b-versatile
on Groq. Not the Azure candidate (would make it partially grade its own
output) and not the Groq candidate (same problem). Using a different Groq
model rather than a third Azure deployment also avoids the Foundry
serverless-availability friction that blocked the original 3-model plan —
no new provisioning required, since the Groq API key is already in use.
"""

import os

from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI

_GROQ_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
_AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

MODEL_CONFIGS = [
    {
        "name": f"groq-{_GROQ_MODEL}",
        "build_llm": lambda: ChatGroq(
            model_name=_GROQ_MODEL,
            api_key=os.getenv("LLAMA_GROQ_KEY"),
            temperature=0.7,
        ),
    },
    {
        "name": f"azure-{_AZURE_DEPLOYMENT}",
        "build_llm": lambda: ChatOpenAI(
            model=_AZURE_DEPLOYMENT,
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            temperature=0.7,
        ),
    },
]

_CANDIDATE_DEPLOYMENT_NAMES = {_GROQ_MODEL, _AZURE_DEPLOYMENT}


def build_judge_llm():
    """Fixed evaluator LLM for RAGAS metrics and the abstention judge.

    Independent of both candidates by construction (see module docstring).
    Raises loudly rather than silently biasing results if a JUDGE_GROQ_MODEL
    override happens to collide with a candidate's model name.
    """
    judge_model = os.getenv("JUDGE_GROQ_MODEL", "llama-3.3-70b-versatile")
    if judge_model in _CANDIDATE_DEPLOYMENT_NAMES:
        raise ValueError(
            f"JUDGE_GROQ_MODEL ('{judge_model}') matches a candidate model — "
            "the judge must be independent of every model under test. "
            "Pick a different model or override via JUDGE_GROQ_MODEL."
        )
    return ChatGroq(
        model_name=judge_model,
        api_key=os.getenv("LLAMA_GROQ_KEY"),
        temperature=0,
    )