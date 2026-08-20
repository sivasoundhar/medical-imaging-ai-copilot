"""Provider factory (Day 9, PROJECT_SPEC.md Section 10): "The application
chooses the provider using configuration" -- this module is the ONLY
place that branches on provider name. Everything downstream (services,
tests) depends only on the `LLMProvider` interface, so switching
`LLM_PROVIDER` never touches business logic.
"""
from src.config import Settings, get_settings
from src.llm.base import LLMConfigurationError, LLMProvider
from src.llm.claude_provider import ClaudeProvider
from src.llm.groq_provider import GroqProvider
from src.llm.mock_provider import MockLLMProvider
from src.llm.ollama_provider import OllamaProvider
from src.schemas.llm import LLMResult


def _build_provider(provider_name: str, settings: Settings) -> LLMProvider:
    """The one place that branches on provider name (module docstring's
    own rule) -- shared by both `get_llm_provider` and
    `get_llm_providers` so primary and fallback are built identically."""
    provider_name = provider_name.lower()

    if provider_name == "groq":
        if not settings.groq_api_key:
            raise LLMConfigurationError("LLM_PROVIDER=groq requires GROQ_API_KEY to be set.")
        if not settings.groq_model:
            raise LLMConfigurationError("LLM_PROVIDER=groq requires GROQ_MODEL to be set.")
        return GroqProvider(api_key=settings.groq_api_key, model=settings.groq_model)

    if provider_name == "ollama":
        return OllamaProvider(base_url=settings.ollama_base_url, model=settings.ollama_model)

    if provider_name == "claude":
        if not settings.anthropic_api_key:
            raise LLMConfigurationError("LLM_PROVIDER=claude requires ANTHROPIC_API_KEY to be set.")
        return ClaudeProvider(api_key=settings.anthropic_api_key, model=settings.claude_model)

    if provider_name == "mock":
        return MockLLMProvider()

    raise LLMConfigurationError(
        f"Unknown LLM provider: {provider_name!r}. Must be 'groq', 'ollama', 'claude', or 'mock'."
    )


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Builds the configured *primary* provider from `LLM_PROVIDER` (env
    var, via `Settings`). Raises `LLMConfigurationError` for a missing
    required key/model or an unrecognized provider name -- fails fast
    rather than at the first real request. Kept for callers that only
    ever want one provider (e.g. this module's own gateway tests); see
    `get_llm_providers` for the primary+fallback list."""
    settings = settings or get_settings()
    return _build_provider(settings.llm_provider, settings)


def get_llm_providers(settings: Settings | None = None) -> list[LLMProvider]:
    """Day 11: primary provider, plus `LLM_FALLBACK_PROVIDER` if one is
    configured (Section 21: "Invalid -> retry / fallback" -- Day 10 only
    built the retry half; this adds the actual cross-provider fallback).
    A misconfigured fallback (e.g. LLM_FALLBACK_PROVIDER=groq with no
    GROQ_API_KEY) raises immediately too, same as a misconfigured
    primary -- a fallback nobody can actually reach should be reported,
    not silently dropped."""
    settings = settings or get_settings()
    providers = [_build_provider(settings.llm_provider, settings)]
    if settings.llm_fallback_provider:
        providers.append(_build_provider(settings.llm_fallback_provider, settings))
    return providers


async def generate_result(
    provider: LLMProvider, messages: list[dict], *, temperature: float = 0.2
) -> LLMResult:
    """Wraps a provider's raw text response in the Section 31 response
    contract's structured `LLMResult` (Day 9's "structured Pydantic
    output"). `grounded` is always `False` here -- Day 9 is gateway
    plumbing only; actually grounding a report in vision findings (and
    setting this flag for real) is Day 10 scope, not implemented yet."""
    report = await provider.generate(messages, temperature=temperature)
    return LLMResult(
        provider=provider.provider_name,
        model=provider.model,
        report=report,
        grounded=False,
    )
