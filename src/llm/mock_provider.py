"""Deterministic mock provider (Day 9) -- never makes a network call.

An explicit Day 9 deliverable ("mock provider for CI"), not just a test
fixture: `LLM_PROVIDER=mock` is a valid, selectable config value (see
`gateway.py`) for CI, demos, and offline development with no real
provider configured.
"""
from collections.abc import Callable

from src.llm.base import LLMProvider


class MockLLMProvider(LLMProvider):
    """`response` is either a fixed string (always returned) or a
    `(messages, temperature) -> str` callable, for tests that need to
    inspect what was actually sent to `generate()`."""

    def __init__(
        self,
        response: str | Callable[[list[dict], float], str] = "This is a mock LLM response.",
        model: str = "mock-model",
    ):
        self._response = response
        self._model = model

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, messages: list[dict], *, temperature: float = 0.2) -> str:
        if callable(self._response):
            return self._response(messages, temperature)
        return self._response
