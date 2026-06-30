import os
from unittest import mock

from rag_eval.core.config import EvalConfig, LLMConfig


def test_llm_config_defaults():
    config = LLMConfig()
    assert config.provider == "ollama"
    assert config.model == "llama3.2"
    assert config.temperature == 0.0


def test_llm_config_get_api_key_explicit():
    config = LLMConfig(api_key="secret")
    assert config.get_api_key() == "secret"


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-secret"})
def test_llm_config_get_api_key_env():
    config = LLMConfig(provider="openai")
    assert config.get_api_key() == "env-secret"


@mock.patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://my-ollama:11434"})
def test_llm_config_get_api_base_env():
    config = LLMConfig(provider="ollama")
    assert config.get_api_base() == "http://my-ollama:11434"


def test_llm_config_get_api_base_explicit():
    config = LLMConfig(provider="ollama", api_base="http://custom:11434")
    assert config.get_api_base() == "http://custom:11434"


def test_eval_config_from_yaml(tmp_path):
    yaml_content = """
    llm:
      provider: openai
      model: gpt-4
    metrics:
      - exact_match
    """
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml_content)

    config = EvalConfig.from_yaml(yaml_file)
    assert config.llm.provider == "openai"
    assert config.llm.model == "gpt-4"
    assert config.metrics == ["exact_match"]
