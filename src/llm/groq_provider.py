"""Groq provider (Day 9, PROJECT_SPEC.md Section 11) -- primary cloud
provider, fast inference.

Groq exposes an OpenAI-compatible chat completions API. Calls it
directly via `httpx` (already a project dependency for the FastAPI test
client) rather than adding the `openai` SDK as a dependency for one
provider.
"""
import httpx

from src.llm.base import LLMError, LLMProvider
from src.utils.logging import get_logger

logger = get_logger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 30.0


class GroqProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        """`transport` lets tests inject an `httpx.MockTransport` instead
        of hitting the real Groq API (Day 9: "Do not call real external
        providers from automated CI tests")."""
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._transport = transport

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, messages: list[dict], *, temperature: float = 0.2) -> str:
        logger.info("Calling Groq (model=%s)", self._model)
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            try:
                response = await client.post(
                    GROQ_API_URL,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "model": self._model,
                        "messages": messages,
                        "temperature": temperature,
                    },
                )
                response.raise_for_status()
            except httpx.TimeoutException as exc:
                raise LLMError(f"Groq request timed out after {self._timeout}s") from exc
            except httpx.HTTPStatusError as exc:
                raise LLMError(
                    f"Groq API error {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except httpx.HTTPError as exc:
                raise LLMError(f"Groq request failed: {exc}") from exc

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Groq response shape: {data}") from exc
