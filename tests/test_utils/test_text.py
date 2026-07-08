"""Tests for text processing utilities."""

from __future__ import annotations

import pytest

from rag_eval.utils.text import (
    count_tokens,
    extract_json_from_text,
    normalize_text,
    split_into_claims,
    split_into_sentences,
)


class TestCountTokens:
    """Test token counting across providers."""

    def test_openai_model(self):
        tokens = count_tokens("Hello, world!", model="gpt-4o")
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_local_model_heuristic(self):
        text = "This is a simple test sentence with several words."
        tokens = count_tokens(text, model="llama3.2")
        assert isinstance(tokens, int)
        # Heuristic: ~1.3 tokens per word, 9 words → ~12 tokens
        assert 8 <= tokens <= 20

    def test_empty_string(self):
        tokens = count_tokens("", model="gpt-4o")
        assert tokens == 0


class TestSplitIntoSentences:
    """Test sentence splitting with NLTK."""

    def test_basic_splitting(self):
        text = "This is sentence one. This is sentence two. And a third."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3

    def test_abbreviations(self):
        text = "Dr. Smith went to Washington. He met with Mrs. Jones."
        sentences = split_into_sentences(text)
        # NLTK should handle "Dr." and "Mrs." correctly
        assert len(sentences) == 2

    def test_empty_text(self):
        assert split_into_sentences("") == []
        assert split_into_sentences("   ") == []

    def test_single_sentence(self):
        sentences = split_into_sentences("Just one sentence here.")
        assert len(sentences) == 1
        assert sentences[0] == "Just one sentence here."

    def test_question_and_exclamation(self):
        text = "Is this a question? Yes it is! And here's a statement."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3


class TestSplitIntoClaims:
    """Test claim splitting (currently delegates to sentence splitting)."""

    def test_basic(self):
        text = "Paris is the capital of France. It has the Eiffel Tower."
        claims = split_into_claims(text)
        assert len(claims) == 2

    def test_empty(self):
        assert split_into_claims("") == []


class TestNormalizeText:
    """Test text normalization."""

    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_remove_articles(self):
        assert normalize_text("The cat sat on a mat") == "cat sat on mat"

    def test_strip_whitespace(self):
        assert normalize_text("  hello  world  ") == "hello world"

    def test_combined(self):
        assert normalize_text("  The Quick Brown Fox  ") == "quick brown fox"


class TestExtractJsonFromText:
    """Test JSON extraction from LLM output text."""

    def test_pure_json(self):
        result = extract_json_from_text('{"score": 0.85}')
        assert result == {"score": 0.85}

    def test_json_with_surrounding_text(self):
        text = 'Here is my analysis:\n{"score": 0.85, "reason": "good"}\nThat is all.'
        result = extract_json_from_text(text)
        assert result["score"] == 0.85
        assert result["reason"] == "good"

    def test_nested_json(self):
        text = '{"outer": {"inner": 42}}'
        result = extract_json_from_text(text)
        assert result["outer"]["inner"] == 42

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No valid JSON"):
            extract_json_from_text("This has no JSON at all")

    def test_json_with_markdown_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = extract_json_from_text(text)
        assert result["key"] == "value"
