import os

from evals.eval_retriever import LLMJudge
from src.llm_factory import get_llm_config


def test_get_llm_config_uses_provider_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    cfg = get_llm_config()

    assert cfg["provider"] == "gemini"
    assert cfg["model"] == "gemini-3.8-flash"
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
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")

    cfg = get_llm_config(provider="gemini")

    assert cfg["provider"] == "gemini"
    assert cfg["model"] == "gemini-3.8-flash"


def test_get_llm_factory_uses_low_thinking_by_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")

    llm = __import__("src.llm_factory", fromlist=["get_llm"]).get_llm()

    assert llm.model == "gemini-3.8-flash"
    assert llm.thinking_level == "low"


def test_llm_judge_generate_normalizes_list_content(monkeypatch):
    class DummyResponse:
        content = [{"text": "hello"}, {"text": " world"}]

    class DummyLLM:
        def invoke(self, prompt):
            return DummyResponse()

    judge = LLMJudge.__new__(LLMJudge)
    judge.llm = DummyLLM()

    assert judge.generate("ping") == "hello world"
