"""
Backend interface for language model interaction.

The machine needs a subject to probe. These backends are the
strapped-down patient on the operating table — we ask the questions,
they answer, and we study the squirming.
"""

import json
import requests
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional


@dataclass
class ModelResponse:
    """What came back from the void."""
    text: str
    model: str
    backend: str
    metadata: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def token_count_estimate(self) -> int:
        """Rough token estimate. Good enough for our purposes."""
        return len(self.text.split()) * 4 // 3


class Backend(ABC):
    """A thing that answers questions. We want to find where it can't."""

    @abstractmethod
    def query(self, prompt: str, system: str = "", temperature: float = 0.7) -> ModelResponse:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class OllamaBackend(Backend):
    """Local model via Ollama. Free. Private. Caged on your own hardware."""

    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._verify_connection()

    def _verify_connection(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if self.model not in models:
                available = ", ".join(models) or "none"
                raise RuntimeError(
                    f"Model '{self.model}' not found in Ollama. Available: {available}"
                )
        except requests.ConnectionError:
            raise RuntimeError(
                "Cannot reach Ollama. Run 'ollama serve' first."
            )

    def query(self, prompt: str, system: str = "", temperature: float = 0.7) -> ModelResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature},
        }
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return ModelResponse(
            text=data.get("response", ""),
            model=self.model,
            backend="ollama",
            metadata={
                "total_duration_ns": data.get("total_duration"),
                "eval_count": data.get("eval_count"),
            },
        )

    def name(self) -> str:
        return f"ollama:{self.model}"


class AnthropicBackend(Backend):
    """Claude via the Anthropic API. Costs money. Arguably more interesting to probe."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("pip install anthropic")
        self.model = model
        self.client = anthropic.Anthropic()

    def query(self, prompt: str, system: str = "", temperature: float = 0.7) -> ModelResponse:
        import anthropic
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        msg = self.client.messages.create(**kwargs)
        text = msg.content[0].text if msg.content else ""
        return ModelResponse(
            text=text,
            model=self.model,
            backend="anthropic",
            metadata={
                "input_tokens": msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
                "stop_reason": msg.stop_reason,
            },
        )

    def name(self) -> str:
        return f"anthropic:{self.model}"


def create_backend(backend_type: str = "ollama", **kwargs) -> Backend:
    """Factory. Pick your subject."""
    if backend_type == "ollama":
        return OllamaBackend(**kwargs)
    elif backend_type == "anthropic":
        return AnthropicBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend_type}. Try 'ollama' or 'anthropic'.")
