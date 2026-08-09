# Build stage: install dependencies into a venv at a fixed path
FROM python:3.13-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# A venv at a fixed path (not tied to any user's home directory) instead of
# `pip install --user` — the previous version copied --user-installed
# packages from root's home (/root/.local) into appuser's home
# (/home/appuser/.local), but the model-baking step below runs BEFORE the
# USER switch, so it executes as root and looks in /root/.local — where
# nothing was copied. A venv at /opt/venv works identically regardless of
# which user is active, sidestepping that mismatch entirely.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# --extra-index-url pulls the CPU-only torch wheel instead of the default
# CUDA-enabled build — this app only ever runs on CPU (see EMBEDDING_MODEL
# config: device="cpu"), so the GPU build is pure dead weight.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Runtime stage 
FROM python:3.13-slim

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Fixed, explicit cache path — NOT the default ~/.cache/huggingface, which
# would resolve to /root/.cache at bake time (root) but /home/appuser/.cache
# at runtime (appuser), making the baked-in model invisible to the app that
# actually needs it. A fixed path sidesteps that mismatch the same way the
# venv above does.
ENV HF_HOME=/opt/hf_cache
RUN mkdir -p /opt/hf_cache

COPY . .

# Bake the embedding model into the image at build time. Without this, the
# first request after every cold start / scale-up depends on a live
# download from Hugging Face's servers — a real reliability risk (network
# hiccup, HF outage, rate limiting) that shouldn't gate whether your app
# can serve traffic.
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; \
    HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2', model_kwargs={'device': 'cpu'})"

RUN chown -R appuser:appuser /app /opt/hf_cache
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Sync workers (not --preload): app.py starts a background asyncio event
# loop in its own thread at import time, and fork() only duplicates the
# calling thread — a preloaded app's background loop thread would not
# exist in forked worker processes. Each worker instead does its own full
# startup (embedding model load, Redis/Search connections) independently.
# --graceful-timeout gives in-flight streaming responses time to finish
# on SIGTERM (Container Apps sends this before killing a replica) instead
# of being cut off mid-stream.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", \
     "--workers", "2", "--threads", "4", \
     "--timeout", "120", "--graceful-timeout", "30", \
     "app:app"]