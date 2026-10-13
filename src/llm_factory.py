import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROVIDER_DEFAULTS = {
    "openai": {
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "api_key_field": "api_key",
    },
    "gemini": {
        "model": "gemini-3.8-flash",
        "api_key_env": "GEMINI_API_KEY",
        "api_key_field": "google_api_key",
    },
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
        "api_key_field": "api_key",
    },
    "grok": {
        "model": "grok-2-1212",
        "api_key_env": "GROK_API_KEY",
        "api_key_field": "api_key",
    },
    "claude": {
        "model": "claude-3-5-sonnet-20241022",
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_key_field": "anthropic_api_key",
    },
    "ollama": {
        "model": "llama3.2",
        "api_key_env": None,
        "api_key_field": None,
    },
}

SUPPORTED_PROVIDERS = tuple(PROVIDER_DEFAULTS.keys())


def normalize_provider(provider: str | None) -> str:
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai")

    normalized = str(provider).strip().lower()
    aliases = {
        "openai": "openai",
        "gpt": "openai",
        "gemini": "gemini",
        "google": "gemini",
        "groq": "groq",
        "grok": "grok",
        "xai": "grok",
        "claude": "claude",
        "anthropic": "claude",
        "ollama": "ollama",
        "local": "ollama",
    }

    resolved = aliases.get(normalized)
    if resolved is None:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. "
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    return resolved


def get_llm_config(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> dict[str, Any]:
    resolved_provider = normalize_provider(provider)
    defaults = PROVIDER_DEFAULTS[resolved_provider]

    provider_model_env = f"{resolved_provider.upper()}_MODEL"
    provider_model_name_env = f"{resolved_provider.upper()}_MODEL_NAME"

    resolved_model = (
        model
        or os.getenv(provider_model_env)
        or os.getenv(provider_model_name_env)
        or os.getenv("LLM_MODEL")
        or defaults["model"]
    )

    api_key_env = defaults["api_key_env"]
    api_key = os.getenv(api_key_env) if api_key_env else None

    return {
        "provider": resolved_provider,
        "model": resolved_model,
        "api_key_env": api_key_env,
        "api_key_field": defaults["api_key_field"],
        "api_key": api_key,
        "temperature": temperature,
        "extra_kwargs": dict(kwargs),
    }


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
):
    config = get_llm_config(
        provider=provider,
        model=model,
        temperature=temperature,
        **kwargs,
    )

    provider = config["provider"]
    extra_kwargs = dict(config["extra_kwargs"])

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "Install langchain-ollama to use the Ollama provider."
            ) from exc

        base_url = os.getenv("OLLAMA_BASE_URL")
        if base_url:
            extra_kwargs.setdefault("base_url", base_url)

        return ChatOllama(
            model=config["model"],
            temperature=config["temperature"],
            **extra_kwargs,
        )

    if not config["api_key"]:
        raise ValueError(
            f"Missing API key for provider '{provider}'. "
            f"Set {config['api_key_env']} in your environment or .env file."
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config["model"],
            api_key=config["api_key"],
            temperature=config["temperature"],
            **extra_kwargs,
        )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "Install langchain-google-genai to use the Gemini provider."
            ) from exc

        thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "low")
        extra_kwargs.setdefault("thinking_level", thinking_level)

        gemini_kwargs = {
            "model": config["model"],
            "google_api_key": config["api_key"],
            "temperature": config["temperature"],
        }
        gemini_kwargs.update(extra_kwargs)

        return ChatGoogleGenerativeAI(**gemini_kwargs)

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=config["model"],
            api_key=config["api_key"],
            temperature=config["temperature"],
            **extra_kwargs,
        )

    if provider == "grok":
        from langchain_openai import ChatOpenAI

        extra_kwargs.setdefault("base_url", "https://api.x.ai/v1")
        return ChatOpenAI(
            model=config["model"],
            api_key=config["api_key"],
            temperature=config["temperature"],
            **extra_kwargs,
        )

    if provider == "claude":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "Install langchain-anthropic to use the Claude provider."
            ) from exc

        return ChatAnthropic(
            model=config["model"],
            anthropic_api_key=config["api_key"],
            temperature=config["temperature"],
            **extra_kwargs,
        )

    raise ValueError(f"Unsupported provider: {provider}")
