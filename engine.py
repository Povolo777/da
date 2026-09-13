"""
engine.py — pluggable inference backends for Alex.

Inspired by OpenJarvis's (https://github.com/open-jarvis/OpenJarvis)
registry pattern: each backend is a small class registered under a
name, and brain.py only ever talks to the abstract Engine interface.
Swapping or adding a backend later (a cloud API, a different local
runtime) means writing one new class here — brain.py doesn't change.

Currently only "ollama" is implemented, since that's what Alex runs on:
a single local process, no external server to manage.
"""

from abc import ABC, abstractmethod

import ollama

ENGINE_REGISTRY = {}


def register_engine(name: str):
    def decorator(cls):
        ENGINE_REGISTRY[name] = cls
        return cls
    return decorator


class Engine(ABC):
    """Minimal contract every backend must satisfy."""

    @abstractmethod
    def check_available(self):
        """Raise a clear RuntimeError if the backend can't be reached."""

    @abstractmethod
    def chat(self, messages: list, tools: list, max_tokens: int) -> dict:
        """Returns a dict: {'content': str, 'tool_calls': [{'name': str, 'input': dict}, ...]}"""


@register_engine("ollama")
class OllamaEngine(Engine):
    def __init__(self, model: str):
        self.model = model

    def check_available(self):
        try:
            ollama.list()
        except Exception as e:
            raise RuntimeError(
                "Couldn't reach Ollama. Make sure it's installed and running "
                "(https://ollama.com), and that you've pulled a model, e.g. "
                f"'ollama pull {self.model}'. Original error: {e}"
            )

    def chat(self, messages: list, tools: list, max_tokens: int) -> dict:
        response = ollama.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            think=False,  # Qwen3 and other reasoning-capable models auto-enable
                          # "thinking" by default — off here so replies stay
                          # fast and spoken text never mixes with raw reasoning.
            options={"num_predict": max_tokens},
        )
        message = response["message"]
        tool_calls = []
        for call in (message.get("tool_calls") or []):
            tool_calls.append({
                "name": call["function"]["name"],
                "input": call["function"]["arguments"],
            })
        return {"content": message.get("content", "") or "", "tool_calls": tool_calls}


def build_engine(config: dict) -> Engine:
    """config is the 'brain' section of config.json."""
    name = config.get("engine", "ollama")
    if name not in ENGINE_REGISTRY:
        raise ValueError(f"Unknown engine '{name}'. Available: {list(ENGINE_REGISTRY)}")
    engine_cfg = config.get(name, {})
    if name == "ollama":
        return OllamaEngine(model=engine_cfg.get("model", "qwen3:14b"))
    raise ValueError(f"No constructor wired up for engine '{name}'")