"""Minimal backend abstraction for chat models with capability hints.

Allows switching between transformers (existing QwenChatAPI) and llama.cpp (GGUF) backends
using environment variable MODEL_BACKEND=transformers|llama.cpp

This is intentionally lightweight and avoids any heavy optional import if not needed.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import Any, Protocol


# Protocol definition -------------------------------------------------------
class ChatBackend(Protocol):
    capabilities: dict[str, bool]

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9, **kwargs: Any) -> dict[str, Any]:
        """Generate completion from list of messages.
        Return dict with keys: text, prompt_tokens, completion_tokens, total_tokens.
        """
        ...

# Transformers (existing) ---------------------------------------------------
@dataclass
class TransformersBackend:
    qwen_api: Any
    capabilities: dict[str, bool] = field(default_factory=lambda: {
        "tools": True,
        "tool_choice": True,
        "max_input_tokens": True,
        "repetition_penalty": True,
        "stream": False,
    })

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9, **kwargs: Any) -> dict[str, Any]:
        # Accept extra kwargs and pass along when applicable.
        repetition_penalty = kwargs.get("repetition_penalty", 1.1)
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        max_input_tokens = kwargs.get("max_input_tokens")
        result = self.qwen_api.chat_messages(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            tools=tools,
            tool_choice=tool_choice,
            max_input_tokens=max_input_tokens,
            return_usage=True,
        )
        return {
            "text": result.text,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        }

# llama.cpp backend ---------------------------------------------------------
@dataclass
class LlamaCppBackend:
    llm: Any
    capabilities: dict[str, bool] = field(default_factory=lambda: {
        "tools": False,  # basic chat only by default
        "tool_choice": False,
        "max_input_tokens": False,  # controlled by context; we expose max_tokens per call
        "repetition_penalty": True,  # map to repeat_penalty when provided
        "stream": False,
    })

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9, **kwargs: Any) -> dict[str, Any]:
        # Convert to llama.cpp chat format (already similar: list of {role, content})
        try:
            # Optional knobs
            repeat_penalty = kwargs.get("repetition_penalty")
            create_kwargs: dict[str, Any] = {
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if repeat_penalty is not None:
                # llama.cpp uses repeat_penalty
                create_kwargs["repeat_penalty"] = float(repeat_penalty)
            output = self.llm.create_chat_completion(
                **create_kwargs,
            )
            choice = output["choices"][0]["message"]
            text = choice.get("content", "")
            usage = output.get("usage", {})
            return {
                "text": text,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)),
            }
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"llama.cpp generation failed: {e}") from e

# Factory -------------------------------------------------------------------

def load_backend(
    model_path: str,
    backend: str = "transformers",
    **kwargs: Any,
) -> ChatBackend:
    backend = backend.lower()
    if backend == "transformers":
        from .qwenchat import QwenChatAPI  # local import to avoid heavy cost  # noqa: PLC0415
        qwen = QwenChatAPI(model_path=model_path, **kwargs)
        return TransformersBackend(qwen_api=qwen)
    elif backend in {"llama.cpp", "llamacpp", "llama"}:
        try:
            from llama_cpp import Llama  # type: ignore  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("llama_cpp not installed. pip install llama-cpp-python") from e
        # Minimal sane defaults; allow override via kwargs
        n_ctx = int(os.getenv("LLAMA_CTX", os.getenv("MAX_INPUT_TOKENS", "1024")))
        n_threads = int(os.getenv("LLAMA_THREADS", "4"))
        n_batch_env = os.getenv("LLAMA_BATCH") or os.getenv("LLAMA_N_BATCH")
        n_ubatch_env = os.getenv("LLAMA_UBATCH") or os.getenv("LLAMA_N_UBATCH")
        llm_kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_threads": n_threads,
        }
        if n_batch_env:
            with contextlib.suppress(ValueError):
                n_batch_val = int(n_batch_env)
                if n_batch_val > 0:
                    llm_kwargs["n_batch"] = n_batch_val
        if n_ubatch_env:
            with contextlib.suppress(ValueError):
                n_ubatch_val = int(n_ubatch_env)
                if n_ubatch_val > 0:
                    llm_kwargs["n_ubatch"] = n_ubatch_val
        llm_kwargs.update(kwargs)
        llm = Llama(**llm_kwargs)
        return LlamaCppBackend(llm=llm)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported MODEL_BACKEND: {backend}")
