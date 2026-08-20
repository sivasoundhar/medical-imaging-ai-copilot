"""Claude provider (Day 9, PROJECT_SPEC.md Section 19) -- optional
high-quality cloud provider for comparison/evaluation. The app must not
depend on it (Section 19: "Do not make the application dependent on
Claude") -- Groq/Ollama cover the required paths.

Uses the official `anthropic` SDK's async client, per project convention
for calling Claude (never raw HTTP when an SDK exists).
"""
import anthropic

from src.llm.base import LLMError, LLMProvider
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Current recommended default -- PROJECT_SPEC.md Section 19 explicitly
# warns provider model availability changes, so CLAUDE_MODEL should
# normally be set explicitly; this is only a fallback when it isn't.
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 1024


class ClaudeProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: anthropic.AsyncAnthropic | None = None,
    ):
        """`client` lets tests inject a stub instead of hitting the real
        Claude API (Day 9: "Do not call real external providers from
        automated CI tests")."""
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model or DEFAULT_MODEL
        self._max_tokens = max_tokens

    @property
    def provider_name(self) -> str:
        return "claude"

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, messages: list[dict], *, temperature: float = 0.2) -> str:
        # Anthropic's Messages API takes "system" as a separate top-level
        # field, not a message with role "system" (unlike Groq/Ollama's
        # OpenAI-style APIs) -- translate rather than forwarding the
        # messages list unchanged.
        system_prompt: str | None = None
        chat_messages = list(messages)
        if chat_messages and chat_messages[0].get("role") == "system":
            system_prompt = chat_messages[0]["content"]
            chat_messages = chat_messages[1:]

        logger.info("Calling Claude (model=%s)", self._model)

        # `temperature` is accepted only to satisfy the shared
        # LLMProvider interface -- it is deliberately NOT forwarded.
        # Current-generation Claude models (Opus 5, Opus 4.7+) reject
        # non-default sampling parameters outright with a 400.
        request: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": chat_messages,
        }
        if system_prompt is not None:
            request["system"] = system_prompt

        try:
            response = await self._client.messages.create(**request)
        except anthropic.APITimeoutError as exc:
            raise LLMError("Claude request timed out") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError(f"Claude rate limit hit: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach Claude API: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMError("Claude declined to respond (safety refusal).")

        text_blocks = [block.text for block in response.content if block.type == "text"]
        if not text_blocks:
            raise LLMError(f"Claude returned no text content: {response.content}")
        return "".join(text_blocks)
