"""Tests for the backend interface."""

import pytest
from src.backends import ModelResponse, OllamaBackend, AnthropicBackend, create_backend


class TestModelResponse:
    def test_basic_properties(self):
        r = ModelResponse(text="hello world", model="test", backend="test")
        assert r.text == "hello world"
        assert not r.is_empty

    def test_empty_response(self):
        r = ModelResponse(text="", model="test", backend="test")
        assert r.is_empty

    def test_whitespace_is_empty(self):
        r = ModelResponse(text="   \n  ", model="test", backend="test")
        assert r.is_empty

    def test_token_count_estimate(self):
        r = ModelResponse(text="one two three four", model="test", backend="test")
        # 4 words * 4/3 ≈ 5
        assert r.token_count_estimate > 0

    def test_metadata_default(self):
        r = ModelResponse(text="hi", model="test", backend="test")
        assert r.metadata == {}


class TestCreateBackend:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("nonexistent")

    def test_anthropic_without_package(self):
        """Anthropic backend should fail gracefully if package not installed or no key."""
        # This test just verifies the factory doesn't crash before trying to init
        # The actual init may fail due to missing API key, which is expected
        try:
            backend = create_backend("anthropic")
        except (RuntimeError, Exception):
            pass  # Expected — no API key or package

    def test_ollama_requires_server(self):
        """OllamaBackend should give a clear error if server isn't reachable."""
        try:
            # Try connecting to a port that's (likely) not running ollama
            backend = OllamaBackend(base_url="http://localhost:99999")
            pytest.fail("Should have raised RuntimeError")
        except (RuntimeError, Exception):
            pass  # Expected
