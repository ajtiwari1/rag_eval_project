import os

from src.llm_factory import get_llm_config


def test_get_llm_config_uses_provider_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    cfg = get_llm_config()

    assert cfg["provider"] == "gemini"
    assert cfg["model"] == "gemini-2.0-flash"
    assert cfg["api_key_env"] == "GEMINI_API_KEY"


def test_get_llm_config_prefers_explicit_arguments(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    cfg = get_llm_config(provider="groq", model="llama-3.3-70b-versatile")

    assert cfg["provider"] == "groq"
    assert cfg["model"] == "llama-3.3-70b-versatile"
    assert cfg["api_key_env"] == "GROQ_API_KEY"


def test_get_llm_config_uses_provider_specific_model_first(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")

    cfg = get_llm_config(provider="gemini")

    assert cfg["provider"] == "gemini"
    assert cfg["model"] == "gemini-2.0-flash"
