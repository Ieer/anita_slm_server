import contextlib
import inspect
import os
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_threads = os.getenv("TORCH_NUM_THREADS")
if _threads and _threads.isdigit():
    with contextlib.suppress(Exception):
        torch.set_num_threads(int(_threads))
@dataclass
class GenerationOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class QwenChatConfig:
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "./models/qwen/Qwen2.5-0.5B-Instruct"))
    device_map: str | None = "auto"
    dtype: str | None = field(default_factory=lambda: os.getenv("TORCH_DTYPE", "auto"))
    compile_model: bool = field(default_factory=lambda: os.getenv("COMPILE_MODEL", "0") in {"1", "true", "TRUE", "True"})
    quantization: str | None = field(default_factory=lambda: os.getenv("MODEL_QUANTIZATION", None))
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = "auto"
    max_input_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_INPUT_TOKENS", "2048")))
    default_system_prompt: str | None = None
    low_cpu_mem_usage: bool = field(default_factory=lambda: os.getenv("LOW_CPU_MEM_USAGE", "1") in {"1", "true", "True"})
    global_seed: int | None = field(default_factory=lambda: (int(os.getenv("GLOBAL_SEED", "0")) if os.getenv("GLOBAL_SEED") else None))


class QwenChatAPI:
    def __init__(self, config: QwenChatConfig | None = None, **legacy_kwargs) -> None:  # noqa: PLR0912
        # Backward compatibility: allow passing old kwargs directly.
        if config is None:
            # Map legacy kwargs to config fields if provided, else environment defaults via QwenChatConfig().
            base = QwenChatConfig()
            for k, v in legacy_kwargs.items():
                if hasattr(base, k):
                    setattr(base, k, v)
            config = base
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
        torch_dtype: Any = "auto"
        if isinstance(self.config.dtype, str) and self.config.dtype != "auto":
            dtype_map = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            torch_dtype = dtype_map.get(self.config.dtype.lower(), "auto")
    # transformers 新版已將 torch_dtype 參數標記為 deprecated，改用 dtype；為了向後相容，
    # 我們動態檢查 from_pretrained 簽名是否包含 dtype / torch_dtype 其一，優先使用 dtype。
        _sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
        param_names = set(_sig.parameters.keys())
        dtype_key = "dtype" if "dtype" in param_names else ("torch_dtype" if "torch_dtype" in param_names else None)
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": self.config.device_map,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
        }
        if dtype_key:
            model_kwargs[dtype_key] = torch_dtype
        if self.config.quantization:
            try:
                from transformers import BitsAndBytesConfig  # noqa: PLC0415
                if self.config.quantization == "4bit":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                elif self.config.quantization == "8bit":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
            except ImportError:  # pragma: no cover
                print("警告：需要安裝 bitsandbytes 才能啟用量化")
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path, **model_kwargs)
        if not torch.cuda.is_available() and hasattr(self.model, "config") and hasattr(self.model.config, "attn_implementation"):
            with contextlib.suppress(Exception):
                self.model.config.attn_implementation = "eager" 
        self.model.eval()
        if self.config.compile_model and hasattr(torch, "compile"):
            with contextlib.suppress(Exception):
                self.model = torch.compile(self.model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generation_defaults = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        self.tools = self.config.tools
        self.tool_choice = self.config.tool_choice
        self.max_input_tokens = self.config.max_input_tokens
        self.default_system_prompt = self.config.default_system_prompt
        self.global_seed = self.config.global_seed
        if self.global_seed is not None:
            torch.manual_seed(self.global_seed)

    def build_generation_kwargs(self,  # noqa: PLR0913
                                max_new_tokens: int,
                                temperature: float,
                                top_p: float,
                                repetition_penalty: float,
                                do_sample: bool,
                                extra: dict[str, Any] | None = None) -> dict[str, Any]:
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "do_sample": do_sample,
        }
        if extra:
            kwargs |= extra
        return kwargs

    def _truncate_inputs(self, inputs: dict[str, torch.Tensor], max_input_tokens: int) -> dict[str, torch.Tensor]:
        if "input_ids" not in inputs:
            return inputs
        seq_len = inputs["input_ids"].shape[1]
        if seq_len <= max_input_tokens:
            return inputs
        k = max_input_tokens
        out: dict[str, torch.Tensor] = {
            name: (
                tensor[:, -k:]
                if tensor.dim() == 2 and tensor.shape[1] >= seq_len
                else tensor
            )
            for name, tensor in inputs.items()
        }
        return out

    def _build_inputs(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, tool_choice: Any | None = None) -> dict[str, torch.Tensor]:
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer([text], return_tensors="pt")

    def _truncate_messages_role_aware(self, messages: list[dict[str, Any]], max_input_tokens: int, reserve_system: bool = True) -> list[dict[str, Any]]:
        if max_input_tokens <= 0:
            return messages[-1:] if messages else []
        if not messages:
            return []
        system_msg = None
        body_msgs: list[dict[str, Any]] = []
        for m in messages:
            if reserve_system and m.get("role") == "system" and system_msg is None:
                system_msg = m
            else:
                body_msgs.append(m)
        kept_rev: list[dict[str, Any]] = []
        total_tokens = 0
        for m in reversed(body_msgs):
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                token_ids = self.tokenizer(content, add_special_tokens=False).input_ids
                need = len(token_ids)
            else:
                need = 0
            if total_tokens + need <= max_input_tokens:
                kept_rev.append(m)
                total_tokens += need
            else:
                if isinstance(content, str) and need > 0:
                    remain = max(0, max_input_tokens - total_tokens)
                    if remain > 16:
                        ratio = remain / need
                        cut_chars = max(1, int(len(content) * ratio))
                        m2 = dict(m)
                        m2["content"] = content[-cut_chars:]
                        kept_rev.append(m2)
                        total_tokens = max_input_tokens
                break
        kept = list(reversed(kept_rev))
        return [system_msg] + kept if system_msg is not None else kept

    def chat_messages(self,  # noqa: PLR0913
                    messages: list[dict[str, Any]],
                    max_new_tokens: int = 512,
                    temperature: float = 0.7,
                    top_p: float = 0.9,
                    repetition_penalty: float = 1.1,
                    seed: int | None = None,
                    tools: list[dict[str, Any]] | None = None,
                    tool_choice: Any | None = None,
                    max_input_tokens: int | None = None,
                    return_usage: bool = False,
                    stop: list[str] | None = None) -> Any:
        use_tools = tools if tools is not None else self.tools
        use_tool_choice = tool_choice if tool_choice is not None else self.tool_choice
        if self.default_system_prompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": self.default_system_prompt}] + messages
        effective_token_limit = max_input_tokens if max_input_tokens is not None else self.max_input_tokens
        if effective_token_limit and effective_token_limit > 0:
            messages = self._truncate_messages_role_aware(messages, effective_token_limit)
        inputs = self._build_inputs(messages, tools=use_tools, tool_choice=use_tool_choice)
        cap_tokens = effective_token_limit if effective_token_limit and effective_token_limit > 0 else None
        if cap_tokens is None and self.max_input_tokens and self.max_input_tokens > 0:
            cap_tokens = self.max_input_tokens
        if cap_tokens:
            inputs = self._truncate_inputs(inputs, cap_tokens)
        if seed is not None:
            torch.manual_seed(seed)
        do_sample = temperature != 0
        with torch.inference_mode():
            extra = {"input_ids": inputs["input_ids"], "attention_mask": inputs.get("attention_mask")}
            generated_ids = self.model.generate(**self.build_generation_kwargs(max_new_tokens, temperature, top_p, repetition_penalty, do_sample, extra))  # type: ignore[attr-defined]
        new_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
                    break
        if not return_usage:
            return text
        prompt_tokens = int(inputs["input_ids"].shape[1])
        completion_tokens = int(new_tokens.shape[0])
        total_tokens = prompt_tokens + completion_tokens
        return GenerationOutput(text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)

    def chat(self, prompt: str, **gen_kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.chat_messages(messages, **gen_kwargs)
