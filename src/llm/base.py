"""Provider-agnostic LLM interface (Day 9, PROJECT_SPEC.md Section 10).

`LLMProvider` is an ABC (not a bare `Protocol`) so every provider shares
one concrete base and the same error types, rather than each
reimplementing its own timeout/error handling shape. The `generate()`
signature matches Section 10's required interface exactly.
"""
from abc import ABC, abstractmethod


class LLMConfigurationError(ValueError):
    """Raised when required provider configuration (API key, model name,
    an unknown provider name) is missing or invalid -- a setup problem,
    distinct from a runtime call failure (`LLMError`)."""


class LLMError(RuntimeError):
    """Raised for any provider call failure: timeout, HTTP/API error, or
    an unexpected response shape. Callers catch this one type regardless
    of which provider is configured -- that's the point of the gateway
    (PROJECT_SPEC.md Section 10: business logic must not know which
    provider is active)."""


class LLMProvider(ABC):
    """PROJECT_SPEC.md Section 10's required interface, plus `model`/
    `provider_name` so `src/llm/gateway.py` can build a structured
    `LLMResult` (Section 31) without knowing each provider's internals."""

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    async def generate(self, messages: list[dict], *, temperature: float = 0.2) -> str:
        """`messages` is OpenAI-style: a list of
        `{"role": "system"|"user"|"assistant", "content": str}` dicts.

        Returns the model's text response. Raises `LLMError` on any
        failure (timeout, HTTP/API error, malformed response) -- never
        returns a partial or fabricated string silently.
        """
        ...
