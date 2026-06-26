"""Rate limiter pour les APIs LLM (thread-safe).

Fenêtres glissantes séparées :
  - requêtes : _REQ_WINDOW = 1 s  (RPS)
  - tokens   : _TOK_WINDOW = 60 s (TPM)

Usage:
    from agentic_ml.utils.rate_limiter import get_rate_limiter, RateLimitCallback
    callback = RateLimitCallback(get_rate_limiter())
    llm = ChatMistralAI(..., callbacks=[callback])
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from agentic_ml.config import (
    AGENT_PROVIDER,
    ANTHROPIC_RPM,
    ANTHROPIC_TPM,
    MISTRAL_RPS,
    MISTRAL_TPM,
)

logger = logging.getLogger("agentic_ml.rate_limiter")

_REQ_WINDOW = 1.0   # secondes — fenêtre glissante pour RPS
_TOK_WINDOW = 60.0  # secondes — fenêtre glissante pour TPM


class RateLimiter:
    """Sliding-window rate limiter pour RPS et TPM.

    Thread-safe : le lock n'est tenu que pendant les opérations courtes sur les
    deques, jamais pendant un sleep.
    """

    def __init__(self, rps: int, tpm: int) -> None:
        self._rps = rps
        self._tpm = tpm
        self._lock = threading.Lock()
        self._req_times: deque[float] = deque()
        self._tok_log: deque[tuple[float, int]] = deque()

    def wait_if_needed(self) -> None:
        """Bloque jusqu'à ce que les deux limites autorisent une nouvelle requête."""
        while True:
            sleep_for = self._time_until_allowed()
            if sleep_for <= 0:
                return
            logger.debug("rate limit: sleeping %.2f s", sleep_for)
            time.sleep(sleep_for)

    def record_usage(self, tokens: int) -> None:
        """Enregistre une requête terminée et sa consommation de tokens."""
        now = time.monotonic()
        with self._lock:
            self._req_times.append(now)
            self._tok_log.append((now, tokens))

    def _time_until_allowed(self) -> float:
        """Retourne le nombre de secondes à attendre (0 si immédiat).

        Le lock est tenu uniquement le temps du snapshot ; le sleep se fait hors lock.
        """
        now = time.monotonic()
        req_cutoff = now - _REQ_WINDOW
        tok_cutoff = now - _TOK_WINDOW
        with self._lock:
            while self._req_times and self._req_times[0] < req_cutoff:
                self._req_times.popleft()
            while self._tok_log and self._tok_log[0][0] < tok_cutoff:
                self._tok_log.popleft()

            req_count = len(self._req_times)
            tok_count = sum(t for _, t in self._tok_log)

            if req_count < self._rps and tok_count < self._tpm:
                return 0.0

            waits: list[float] = []
            if req_count >= self._rps and self._req_times:
                waits.append(_REQ_WINDOW - (now - self._req_times[0]))
            if tok_count >= self._tpm and self._tok_log:
                waits.append(_TOK_WINDOW - (now - self._tok_log[0][0]))

        return max(waits) if waits else 0.0


class RateLimitCallback(BaseCallbackHandler):
    """Callback LangChain qui applique le rate limiting autour de chaque appel LLM.

    - on_llm_start → wait_if_needed()  (bloque avant la requête HTTP)
    - on_llm_end   → record_usage()   (enregistre les tokens après la réponse)
    """

    def __init__(self, limiter: RateLimiter) -> None:
        super().__init__()
        self._limiter = limiter

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        self._limiter.wait_if_needed()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        llm_output = response.llm_output or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        tokens = int(usage.get("total_tokens", 0)) if isinstance(usage, dict) else 0
        self._limiter.record_usage(tokens)
        logger.debug("rate limiter | tokens enregistrés : %d", tokens)


# Aliases pour compatibilité avec les imports existants
MistralRateLimiter = RateLimiter
MistralRateLimitCallback = RateLimitCallback


_limiter_instance: RateLimiter | None = None
_singleton_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    """Retourne (ou crée) le singleton RateLimiter partagé par tous les agents."""
    global _limiter_instance
    if _limiter_instance is None:
        with _singleton_lock:
            if _limiter_instance is None:
                if AGENT_PROVIDER == "anthropic":
                    rps = max(1, ANTHROPIC_RPM // 60)
                    tpm = ANTHROPIC_TPM
                else:
                    rps = MISTRAL_RPS
                    tpm = MISTRAL_TPM
                _limiter_instance = RateLimiter(rps=rps, tpm=tpm)
                logger.info(
                    "rate limiter initialisé — provider=%s RPS=%d TPM=%d",
                    AGENT_PROVIDER,
                    rps,
                    tpm,
                )
    return _limiter_instance
