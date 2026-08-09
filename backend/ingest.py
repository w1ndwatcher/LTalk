# backend/ingest.py
"""
Reads Q&A pairs from an Excel file and indexes them into Azure AI Search.
The Azure AI Search index is created programmatically on first run.
No need to define a schema by hand in the portal.

Every chunk is tagged with `metadata["qa_id"]`, the 1-based row number of
its source pair in the Excel file. This is what lets the eval harness check
"did the retriever actually return QA row #13" against the
`expected_source_qa_ids` column in the eval dataset.

Every chunk is also given a stable document id (`qa-<qa_id>`), so re-running
this script without --recreate-index UPSERTS existing rows instead of
duplicating them — routine re-ingests after editing a few Q&A pairs don't
need a full index rebuild.

Env vars required (see .env):
    AZURE_SEARCH_ENDPOINT
    AZURE_SEARCH_KEY
    AZURE_SEARCH_INDEX_NAME
    HUGGINGFACEHUB_API_TOKEN   (for the embedding model download)
"""

import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes import SearchIndexClient

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Config ---
load_dotenv()

EXCEL_FILE = os.getenv("QA_EXCEL_FILE", "learntrail_qa.xlsx")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "learntrail-qa-index")


if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY]):
    raise EnvironmentError(
        "AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY must be set in .env. "
        "See README for how to provision an Azure AI Search resource."
    )


def recreate_index_if_requested(recreate: bool) -> None:
    """Drop the existing index so it gets recreated with the current schema.

    Safe to call when the index doesn't exist yet (first-ever run) — the
    404 from Azure is swallowed since there's nothing to drop.
    """
    if not recreate:
        return
    print(f"⚠️  --recreate-index passed: dropping index '{AZURE_SEARCH_INDEX_NAME}' if it exists...")
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )
    try:
        index_client.delete_index(AZURE_SEARCH_INDEX_NAME)
        print(f"Dropped index '{AZURE_SEARCH_INDEX_NAME}'.")
    except ResourceNotFoundError:
        print(f"Index '{AZURE_SEARCH_INDEX_NAME}' didn't exist yet — nothing to drop.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Drop the existing Azure AI Search index before ingesting, so it's "
             "rebuilt with the current schema (e.g. after adding a new metadata "
             "field). Omit for routine re-ingests — stable per-row ids mean those "
             "upsert safely without a full rebuild.",
    )
    args = parser.parse_args()

    recreate_index_if_requested(args.recreate_index)

    login(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

    # --- Load Excel ---
    df = pd.read_excel(EXCEL_FILE)
    assert "question" in df.columns and "answer" in df.columns, \
        "Excel must have 'question' and 'answer' columns"

    # --- Prepare Documents ---
    # Kept as a splitter-ready pipeline: today the Q&A pairs are short enough
    # that the splitter is a no-op, but this lets you ingest longer-form docs
    # (course descriptions, policy PDFs, etc.) through the same pipeline later
    # without changing the ingestion contract.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    raw_documents = []
    for idx, row in df.iterrows():
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        # 1-based to match how the eval dataset's `expected_source_qa_ids`
        # column refers to rows (row 1 = the first Q&A pair, etc.).
        qa_id = idx + 1
        content = f"Q: {question}\nA: {answer}"
        raw_documents.append(
            Document(
                page_content=content,
                metadata={"source": "learntrail", "question": question, "qa_id": qa_id},
            )
        )

    documents = splitter.split_documents(raw_documents)
    # RecursiveCharacterTextSplitter.split_documents copies each source
    # document's metadata onto every chunk it produces, so qa_id survives
    # even if a future longer document gets split into multiple chunks.
    print(f"Prepared {len(documents)} chunks from {len(raw_documents)} Q&A pairs.")

    # Stable IDs (matching qa_id) mean re-running this script without
    # --recreate-index upserts existing rows instead of duplicating them.
    doc_ids = [f"qa-{d.metadata['qa_id']}" for d in documents]

    # --- Load Embeddings ---
    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print("Embedding model loaded.")

    # --- Create / Update Azure AI Search index ---
    print(f"Connecting to Azure AI Search index '{AZURE_SEARCH_INDEX_NAME}'...")
    vectorstore = AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=AZURE_SEARCH_INDEX_NAME,
        embedding_function=embedding_model.embed_query,
        # Free (F0) tier does not support the semantic ranker; leave this off
        # for now and revisit if/when the resource is upgraded to Basic+.
        search_type="hybrid",
    )

    vectorstore.add_documents(documents=documents, ids=doc_ids)
    print(f"Indexed {len(documents)} chunks into Azure AI Search index '{AZURE_SEARCH_INDEX_NAME}'.")


if __name__ == "__main__":
    main()