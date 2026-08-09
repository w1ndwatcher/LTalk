# backend/redis_cache.py
import hashlib
import json
import uuid
import numpy as np
import redis as redis_lib
from langchain_core.caches import BaseCache
from langchain_core.load import dumps, loads

from config import config


def build_redis_client() -> redis_lib.Redis:
    return redis_lib.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        password=config.REDIS_PASSWORD,
        ssl=config.REDIS_SSL,
        decode_responses=True,
        socket_connect_timeout=5,
    )


def build_redis_url() -> str:
    """For RedisChatMessageHistory, which takes a connection URL rather
    than a client instance. rediss:// (double s) is the TLS scheme Azure
    Cache for Redis requires on its default port."""
    scheme = "rediss" if config.REDIS_SSL else "redis"
    return f"{scheme}://:{config.REDIS_PASSWORD}@{config.REDIS_HOST}:{config.REDIS_PORT}"


class RedisSemanticCache(BaseCache):
    """
    Same embedding-similarity approach as the earlier in-memory version,
    now backed by Redis so hit/miss state and cached responses are shared
    across every worker/replica instead of living in one process's memory.

    Deliberately hand-rolled with brute-force cosine similarity in Python,
    rather than langchain_community's RedisSemanticCache (which relies on
    Redis's vector search via redisvl). At this app's scale — tens to low
    hundreds of cached entries — brute force is simpler, needs no separate
    vector index/schema to manage, and keeps the exact same similarity
    semantics (cosine similarity, threshold=higher-is-closer) as the
    version this replaces, instead of adopting redisvl's differently-
    scaled distance metric.
    """

    def __init__(self, embedding_model, redis_client: redis_lib.Redis,
                 similarity_threshold: float = 0.92, key_prefix: str = "llm_cache",
                 ttl_seconds: int = 60 * 60 * 24 * 7):
        self.embedding_model = embedding_model
        self.redis = redis_client
        self.threshold = similarity_threshold
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        self.exact_hits = 0
        self.semantic_hits = 0

    def _bucket_key(self, llm_string: str) -> str:
        # llm_string (model name + params) can be long; hash it to keep the
        # Redis key short and predictable.
        h = hashlib.sha256(llm_string.encode()).hexdigest()[:16]
        return f"{self.key_prefix}:{h}"

    def lookup(self, prompt: str, llm_string: str):
        key = self._bucket_key(llm_string)
        raw_entries = self.redis.hgetall(key)
        if not raw_entries:
            self.misses += 1
            return None

        entries = [json.loads(v) for v in raw_entries.values()]
        query_vec = np.array(self.embedding_model.embed_query(prompt))
        vecs = np.array([e["vector"] for e in entries])
        sims = vecs @ query_vec / (
            np.linalg.norm(vecs, axis=1) * np.linalg.norm(query_vec) + 1e-8
        )
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.threshold:
            self.hits += 1
            if entries[best_idx]["prompt"] == prompt:
                self.exact_hits += 1
            else:
                self.semantic_hits += 1
            # dumps()/loads() (langchain_core.load) handle full, correct
            # serialization of Generation/ChatGeneration objects — safer
            # than hand-rolling reconstruction of chat message objects.
            return loads(entries[best_idx]["response"])

        self.misses += 1
        return None

    def update(self, prompt: str, llm_string: str, return_val) -> None:
        key = self._bucket_key(llm_string)
        vec = self.embedding_model.embed_query(prompt)
        entry_id = str(uuid.uuid4())
        entry = {
            "prompt": prompt,
            "vector": vec,
            "response": dumps(return_val),
        }
        self.redis.hset(key, entry_id, json.dumps(entry))
        self.redis.expire(key, self.ttl_seconds)  # avoid unbounded growth

    def clear(self, **kwargs) -> None:
        for k in self.redis.scan_iter(f"{self.key_prefix}:*"):
            self.redis.delete(k)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else 0.0