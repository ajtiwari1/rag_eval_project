import importlib.util
from pathlib import Path

import pytest

from scripts.generate_env_config import generate_config, read_env_keys


def test_read_env_keys_ignores_comments_and_deduplicates(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nFIRST=value\nexport SECOND=another\nFIRST=overridden\n",
        encoding="utf-8",
    )

    assert read_env_keys(env_file) == ["FIRST", "SECOND"]


def test_read_env_keys_rejects_malformed_assignment(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("not an assignment\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.env:1: expected"):
        read_env_keys(env_file)


def test_generated_module_reads_values_at_runtime_without_embedding_them(
    tmp_path, monkeypatch
):
    secret = "test-secret-value"
    env_file = tmp_path / ".env.schema"
    env_file.write_text(f"API_TOKEN={secret}\nAPP_MODE=development\n", encoding="utf-8")
    output_file = tmp_path / "generated_config.py"

    assert generate_config(env_file, output_file) == ["API_TOKEN", "APP_MODE"]
    generated_source = output_file.read_text(encoding="utf-8")
    assert secret not in generated_source

    (tmp_path / ".env").write_text(
        f"API_TOKEN={secret}\nAPP_MODE=development\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("generated_config", output_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.get_config() == {
        "API_TOKEN": secret,
        "APP_MODE": "development",
    }


def test_generate_config_rejects_same_input_and_output(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("KEY=value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be different"):
        generate_config(env_file, env_file)
