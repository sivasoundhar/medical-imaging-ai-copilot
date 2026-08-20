"""Ollama provider (Day 9, PROJECT_SPEC.md Section 12) -- local
dev/testing provider. No API key needed; talks to a local Ollama server.
"""
import httpx

from src.llm.base import LLMError, LLMProvider
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Local models on modest hardware can be slow -- more generous than
# Groq/Claude's cloud timeout.
DEFAULT_TIMEOUT_SECONDS = 60.0


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        """`transport` lets tests inject an `httpx.MockTransport` instead
        of requiring a real local Ollama server (Day 9: "Do not call real
        external providers from automated CI tests")."""
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._transport = transport

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, messages: list[dict], *, temperature: float = 0.2) -> str:
        logger.info("Calling Ollama (model=%s, base_url=%s)", self._model, self._base_url)
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            try:
                response = await client.post(
                    f"{self._base_url}/api/chat",
                    json={
                        "model": self._model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": temperature},
                    },
                )
                response.raise_for_status()
            except httpx.TimeoutException as exc:
                raise LLMError(f"Ollama request timed out after {self._timeout}s") from exc
            except httpx.HTTPStatusError as exc:
                raise LLMError(
                    f"Ollama API error {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except httpx.HTTPError as exc:
                raise LLMError(
                    f"Could not reach Ollama at {self._base_url} -- is it running? ({exc})"
                ) from exc

        data = response.json()
        try:
            return data["message"]["content"]
        except KeyError as exc:
            raise LLMError(f"Unexpected Ollama response shape: {data}") from exc
