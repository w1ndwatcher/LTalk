import uuid
import asyncio
import threading
import concurrent.futures
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from huggingface_hub import login

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.globals import set_llm_cache
from langsmith import Client as LangSmithClient

from config import config
from logging_config import setup_logging, get_logger
from guardrails import classify_question
from redis_cache import build_redis_client, build_redis_url, RedisSemanticCache
from prompts import prompt
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

setup_logging()
log = get_logger(__name__)

# Small pool for running the guardrail classification and the relevance-gate
# retrieval concurrently per request (see /ask) — both are blocking I/O
# calls with no dependency on each other.
_io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# Persistent background event loop 
# Async clients used here (notably azure-search-documents' async transport,
# if ever reintroduced) cache a session tied to whichever event loop was
# running when the client was first used. Creating a NEW event loop per
# Flask request (and closing it afterward) breaks any such cached session
# on the very next request — "RuntimeError: Event loop is closed". Running
# one event loop for the entire app's lifetime, in a background thread,
# and submitting each request's async work to it via run_coroutine_threadsafe
# avoids this.
_background_loop = asyncio.new_event_loop()


def _run_background_loop():
    asyncio.set_event_loop(_background_loop)
    _background_loop.run_forever()


threading.Thread(target=_run_background_loop, daemon=True).start()

# App setup
app = Flask(__name__)
# Scoped to the actual frontend origin, not left open to any origin —
# an open CORS policy on an app with an LLM behind it is a standing
# invitation for other sites to burn your API budget from their users'
# browsers.
CORS(app, origins=[config.FRONTEND_ORIGIN], expose_headers=["X-Run-Id"])

login(config.HUGGINGFACEHUB_API_TOKEN)

# Embeddings + Vectorstore 
log.info("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
)
log.info("Embedding model loaded.")

log.info("Connecting to Azure AI Search...")
vectorstore = AzureSearch(
    azure_search_endpoint=config.AZURE_SEARCH_ENDPOINT,
    azure_search_key=config.AZURE_SEARCH_KEY,
    index_name=config.AZURE_SEARCH_INDEX_NAME,
    embedding_function=embedding_model.embed_query,
    search_type="hybrid",
)
log.info("Vectorstore ready.")

# Redis config
if not config.redis_configured:
    raise EnvironmentError(
        "REDIS_HOST and REDIS_PASSWORD must be set — chat history and the "
        "semantic cache require Redis (in-memory versions don't survive "
        "multiple workers/replicas). See README for provisioning steps."
    )

log.info("Connecting to Redis...", extra={"host": config.REDIS_HOST})
_redis_client = build_redis_client()
_redis_client.ping()  # fail fast at startup, not on the first request
_redis_url = build_redis_url()
log.info("Redis connected.")

# Rate limiting 
# Backed by Redis, not in-memory — an in-memory limiter's counters would
# reset per gunicorn worker/replica, meaning the effective limit multiplies
# by however many are running instead of applying globally per client.
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri=_redis_url,
    default_limits=[],  # only the explicitly decorated routes below are limited
)
log.info("Rate limiter enabled.", extra={"ask_limit": config.RATE_LIMIT_ASK})

# LLM response cache (semantic, Redis-backed)
llm_cache = RedisSemanticCache(
    embedding_model, _redis_client,
    similarity_threshold=config.CACHE_SIMILARITY_THRESHOLD,
)
set_llm_cache(llm_cache)
log.info("Semantic LLM cache enabled.", extra={"threshold": config.CACHE_SIMILARITY_THRESHOLD})

# LLM setup
llm = ChatGroq(
    model_name=config.LLM_MODEL,
    api_key=config.LLAMA_GROQ_KEY,
    temperature=0.7,
    streaming=True,
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# Core LCEL chain 
# {question, context, history} -> prompt -> llm -> string
# Context is always passed in directly (computed once, up front, in /ask)
# rather than derived by the chain via its own retrieval call — one
# retrieval now serves both the relevance gate and generation.
rag_chain = prompt | llm | StrOutputParser()

# Chat history store (Redis-backed) 
_CHAT_HISTORY_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days — avoid unbounded growth


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id, url=_redis_url, ttl=_CHAT_HISTORY_TTL_SECONDS,
    )


conversational_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


# --------- Routes -------------
@app.route("/ask", methods=["POST"])
@limiter.limit(config.RATE_LIMIT_ASK)
def ask():
    data = request.get_json()
    query = (data.get("question") or "").strip()
    session_id = data.get("session_id", "default")

    if not query:
        return jsonify({"error": "Question is required"}), 400
    if len(query) > config.MAX_QUESTION_LENGTH:
        return jsonify({"error": f"Question too long (max {config.MAX_QUESTION_LENGTH} characters)."}), 400

    # Guardrail (safety + scope + retrieval-need) and the relevance-gate
    # retrieval run CONCURRENTLY — neither depends on the other's result,
    # so running them sequentially was pure wasted latency. Trade-off:
    # retrieval runs (and its Azure Search call is spent) even for
    # messages that turn out to be greetings/off-topic/unsafe, where the
    # result gets discarded. Accepted deliberately: that's a cheap, fast
    # call, and overlapping it with the guardrail call is worth more than
    # avoiding it entirely.
    guardrail_future = _io_executor.submit(classify_question, query)
    retrieval_future = _io_executor.submit(
        vectorstore.similarity_search_with_relevance_scores, query, k=config.RETRIEVAL_K
    )

    verdict = guardrail_future.result()
    log.info("Guardrail verdict", extra={
        "safe": verdict.safe, "in_scope": verdict.in_scope,
        "needs_retrieval": verdict.needs_retrieval, "session_id": session_id,
    })

    if not verdict.safe:
        def refuse():
            yield "I'm not able to help with that request. I'm here to help with questions about LearnTrail's courses and services."
        return Response(stream_with_context(refuse()), content_type="text/plain")

    if not verdict.in_scope:
        def redirect_response():
            yield "I'm LTalk, and I'm focused on helping with LearnTrail's courses, programs, and career services. Let me know if there's something about LearnTrail I can help with!"
        return Response(stream_with_context(redirect_response()), content_type="text/plain")

    if not verdict.needs_retrieval:
        # Plain greeting/thanks/goodbye — retrieval_future's result is
        # simply discarded here; it ran concurrently for latency, not
        # because we knew in advance we'd need it.
        context_text = ""
    else:
        results = retrieval_future.result()
        relevant = [doc for doc, score in results if score >= config.RELEVANCE_THRESHOLD]
        if not relevant:
            return jsonify({
                "answer": "Sorry, I couldn't find relevant information for your question. "
                          "Please visit https://learntrail.co.in/ for the latest details, "
                          "or reach out to the LearnTrail team directly.",
                "sources": []
            })
        context_text = format_docs(relevant)

    # run_id is generated here (not by LangChain) so we can hand it to the
    # frontend via a header before generation even starts, and so /feedback
    # can reference the exact run once LangSmith tracing is on.
    run_id = uuid.uuid4()

    def generate():
        # Bridge LCEL's async .astream() into a sync Flask generator by
        # running the coroutine on the app's single persistent background
        # loop (see module-level setup above), not a throwaway per-request
        # loop — that pattern breaks async clients' cached sessions on the
        # second request onward.
        async def _stream():
            async for chunk in conversational_chain.astream(
                {"question": query, "context": context_text},
                config={
                    "configurable": {"session_id": session_id},
                    "run_id": run_id,
                },
            ):
                yield chunk

        agen = _stream()
        try:
            while True:
                future = asyncio.run_coroutine_threadsafe(agen.__anext__(), _background_loop)
                try:
                    chunk = future.result()
                except StopAsyncIteration:
                    break
                yield chunk
        except GeneratorExit:
            log.info("Client disconnected mid-stream.", extra={"run_id": str(run_id)})
            # Best-effort cleanup of the async generator on its owning loop.
            close_future = asyncio.run_coroutine_threadsafe(agen.aclose(), _background_loop)
            try:
                close_future.result(timeout=2)
            except Exception:
                pass

    resp = Response(stream_with_context(generate()), content_type="text/plain")
    resp.headers["X-Run-Id"] = str(run_id)
    return resp


@app.route("/feedback", methods=["POST"])
@limiter.limit(config.RATE_LIMIT_FEEDBACK)
def feedback():
    """Log a thumbs up/down against the run that produced a given answer.

    Requires LangSmith tracing to be enabled (LANGCHAIN_TRACING_V2=true,
    LANGCHAIN_API_KEY set) so run_id resolves to a real trace — otherwise
    this will fail with a clear error rather than silently no-op.
    """
    data = request.get_json()
    run_id = data.get("run_id")
    score = data.get("score")  # 1 = thumbs up, 0 = thumbs down
    comment = data.get("comment", "")

    if not run_id or score is None:
        return jsonify({"error": "run_id and score are required"}), 400

    try:
        client = LangSmithClient()
        client.create_feedback(
            run_id=run_id,
            key="user_score",
            score=score,
            comment=comment,
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        log.warning("Feedback logging failed (is LangSmith tracing enabled?)", extra={"error": str(e)})
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    redis_ok = True
    try:
        _redis_client.ping()
    except Exception:
        redis_ok = False

    return jsonify({
        "status": "healthy" if redis_ok else "degraded",
        "components": {
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_model": config.LLM_MODEL,
            "vectorstore": "azure_ai_search" if vectorstore else "unavailable",
            "index_name": config.AZURE_SEARCH_INDEX_NAME,
            "redis": "connected" if redis_ok else "unreachable",
        },
        "cache": {
            "hits": llm_cache.hits,
            "misses": llm_cache.misses,
            "hit_rate": llm_cache.hit_rate,
            "exact_hits": llm_cache.exact_hits,
            "semantic_hits": llm_cache.semantic_hits,
        }
    })


if __name__ == "__main__":
    app.run(
        debug=not config.is_production,
        use_reloader=False,
    )