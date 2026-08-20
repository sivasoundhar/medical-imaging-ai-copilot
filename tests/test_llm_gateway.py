"""Tests for the Day 9 LLM gateway (src/llm/).

All HTTP/SDK calls are mocked -- httpx.MockTransport for Groq/Ollama
(both plain HTTP APIs), a stub client for Claude (the anthropic SDK) --
per Day 9's "Do not call real external providers from automated CI
tests. Use mocked providers for CI."
"""
import httpx
import pytest

from src.config import Settings
from src.llm.base import LLMConfigurationError, LLMError, LLMProvider
from src.llm.claude_provider import ClaudeProvider
from src.llm.gateway import generate_result, get_llm_provider, get_llm_providers
from src.llm.groq_provider import GroqProvider
from src.llm.mock_provider import MockLLMProvider
from src.llm.ollama_provider import OllamaProvider

_MESSAGES = [
    {"role": "system", "content": "You explain grounded vision-model findings."},
    {"role": "user", "content": "Explain this finding."},
]


# --- MockLLMProvider ---------------------------------------------------


@pytest.mark.asyncio
async def test_mock_provider_returns_fixed_string_by_default():
    provider = MockLLMProvider()
    result = await provider.generate(_MESSAGES)
    assert result == "This is a mock LLM response."
    assert provider.provider_name == "mock"


@pytest.mark.asyncio
async def test_mock_provider_callable_response_sees_actual_inputs():
    provider = MockLLMProvider(response=lambda messages, temperature: f"{len(messages)}:{temperature}")
    result = await provider.generate(_MESSAGES, temperature=0.7)
    assert result == "2:0.7"


# --- GroqProvider (mocked HTTP) ----------------------------------------


@pytest.mark.asyncio
async def test_groq_provider_returns_message_content_on_success():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello from groq"}}]})

    provider = GroqProvider(
        api_key="fake-key", model="openai/gpt-oss-120b", transport=httpx.MockTransport(handler)
    )
    result = await provider.generate(_MESSAGES)
    assert result == "hello from groq"
    assert provider.provider_name == "groq"
    assert provider.model == "openai/gpt-oss-120b"


@pytest.mark.asyncio
async def test_groq_provider_raises_llm_error_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid api key"})

    provider = GroqProvider(
        api_key="bad-key", model="openai/gpt-oss-120b", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMError, match="401"):
        await provider.generate(_MESSAGES)


@pytest.mark.asyncio
async def test_groq_provider_raises_llm_error_on_unexpected_response_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    provider = GroqProvider(
        api_key="fake-key", model="openai/gpt-oss-120b", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMError, match="Unexpected Groq response shape"):
        await provider.generate(_MESSAGES)


@pytest.mark.asyncio
async def test_groq_provider_raises_llm_error_on_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    provider = GroqProvider(
        api_key="fake-key", model="openai/gpt-oss-120b", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMError, match="timed out"):
        await provider.generate(_MESSAGES)


# --- OllamaProvider (mocked HTTP) ---------------------------------------


@pytest.mark.asyncio
async def test_ollama_provider_returns_message_content_on_success():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hello from ollama"}})

    provider = OllamaProvider(
        base_url="http://localhost:11434", model="gemma3", transport=httpx.MockTransport(handler)
    )
    result = await provider.generate(_MESSAGES)
    assert result == "hello from ollama"
    assert provider.provider_name == "ollama"


@pytest.mark.asyncio
async def test_ollama_provider_raises_llm_error_when_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = OllamaProvider(
        base_url="http://localhost:11434", model="gemma3", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMError, match="is it running"):
        await provider.generate(_MESSAGES)


# --- ClaudeProvider (stubbed SDK client) --------------------------------


class _FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, text: str, stop_reason: str = "end_turn"):
        self.content = [_FakeTextBlock(text)]
        self.stop_reason = stop_reason


class _FakeMessagesResource:
    def __init__(self, response: _FakeAnthropicResponse | Exception):
        self._response = response
        self.last_request: dict | None = None

    async def create(self, **kwargs):
        self.last_request = kwargs
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response: _FakeAnthropicResponse | Exception):
        self.messages = _FakeMessagesResource(response)


@pytest.mark.asyncio
async def test_claude_provider_returns_text_and_splits_system_message():
    fake_client = _FakeAnthropicClient(_FakeAnthropicResponse("hello from claude"))
    provider = ClaudeProvider(api_key="fake-key", model="claude-opus-5", client=fake_client)

    result = await provider.generate(_MESSAGES)

    assert result == "hello from claude"
    # system role must NOT be forwarded inside `messages` -- Anthropic's
    # API takes it as a separate top-level field.
    sent = fake_client.messages.last_request
    assert sent["system"] == _MESSAGES[0]["content"]
    assert sent["messages"] == _MESSAGES[1:]
    assert "temperature" not in sent  # deliberately never forwarded


@pytest.mark.asyncio
async def test_claude_provider_raises_llm_error_on_refusal():
    fake_client = _FakeAnthropicClient(_FakeAnthropicResponse("", stop_reason="refusal"))
    provider = ClaudeProvider(api_key="fake-key", client=fake_client)

    with pytest.raises(LLMError, match="refusal"):
        await provider.generate(_MESSAGES)


@pytest.mark.asyncio
async def test_claude_provider_defaults_model_when_not_configured():
    fake_client = _FakeAnthropicClient(_FakeAnthropicResponse("ok"))
    provider = ClaudeProvider(api_key="fake-key", model=None, client=fake_client)
    assert provider.model == "claude-opus-5"


# --- Gateway factory -----------------------------------------------------


def _settings(**overrides) -> Settings:
    base = dict(
        llm_provider="mock",
        llm_fallback_provider=None,
        groq_api_key=None,
        groq_model=None,
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.2",
        anthropic_api_key=None,
        claude_model=None,
    )
    base.update(overrides)
    return Settings(**base)


def test_get_llm_provider_returns_mock_for_mock_config():
    provider = get_llm_provider(_settings(llm_provider="mock"))
    assert isinstance(provider, MockLLMProvider)


def test_get_llm_provider_returns_ollama_with_no_api_key_needed():
    provider = get_llm_provider(_settings(llm_provider="ollama"))
    assert isinstance(provider, OllamaProvider)


def test_get_llm_provider_raises_when_groq_key_missing():
    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
        get_llm_provider(_settings(llm_provider="groq", groq_api_key=None, groq_model="openai/gpt-oss-120b"))


def test_get_llm_provider_raises_when_claude_key_missing():
    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        get_llm_provider(_settings(llm_provider="claude", anthropic_api_key=None))


def test_get_llm_provider_raises_on_unknown_provider_name():
    with pytest.raises(LLMConfigurationError, match="Unknown LLM provider"):
        get_llm_provider(_settings(llm_provider="not-a-real-provider"))


def test_get_llm_provider_returns_groq_when_configured():
    provider = get_llm_provider(
        _settings(llm_provider="groq", groq_api_key="fake-key", groq_model="openai/gpt-oss-120b")
    )
    assert isinstance(provider, GroqProvider)


def test_get_llm_provider_returns_claude_when_configured():
    provider = get_llm_provider(_settings(llm_provider="claude", anthropic_api_key="fake-key"))
    assert isinstance(provider, ClaudeProvider)


# --- Gateway factory: primary + fallback (Day 11) -------------------------


def test_get_llm_providers_returns_single_item_list_with_no_fallback_configured():
    providers = get_llm_providers(_settings(llm_provider="mock"))
    assert len(providers) == 1
    assert isinstance(providers[0], MockLLMProvider)


def test_get_llm_providers_returns_primary_then_fallback_in_order():
    providers = get_llm_providers(
        _settings(llm_provider="ollama", llm_fallback_provider="mock")
    )
    assert len(providers) == 2
    assert isinstance(providers[0], OllamaProvider)
    assert isinstance(providers[1], MockLLMProvider)


def test_get_llm_providers_raises_when_fallback_is_misconfigured():
    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
        get_llm_providers(
            _settings(llm_provider="ollama", llm_fallback_provider="groq", groq_api_key=None)
        )


# --- Provider switching without changing business logic (Day 9's
# explicit test requirement) --------------------------------------------


async def _business_logic(provider: LLMProvider) -> str:
    """Stand-in for real application code: depends only on the
    `LLMProvider` interface, never on a concrete provider class."""
    result = await generate_result(provider, _MESSAGES)
    return result.report


@pytest.mark.asyncio
async def test_business_logic_is_unchanged_across_provider_configs():
    mock_provider = get_llm_provider(_settings(llm_provider="mock"))
    assert await _business_logic(mock_provider) == "This is a mock LLM response."

    fake_client = _FakeAnthropicClient(_FakeAnthropicResponse("claude says hi"))
    claude_provider = ClaudeProvider(api_key="fake-key", client=fake_client)
    assert await _business_logic(claude_provider) == "claude says hi"


# --- Structured output (LLMResult) --------------------------------------


@pytest.mark.asyncio
async def test_generate_result_wraps_provider_output_in_llm_result_schema():
    provider = MockLLMProvider(response="the finding is grounded", model="mock-model-v1")
    result = await generate_result(provider, _MESSAGES, temperature=0.3)

    assert result.provider == "mock"
    assert result.model == "mock-model-v1"
    assert result.report == "the finding is grounded"
    assert result.grounded is False
