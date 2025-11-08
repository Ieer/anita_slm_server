"""Full server implementation for SLM API compatible with QwenChat.

建議啟動命令（避免 Windows 啟動器錯誤）：
	python -m uvicorn src.anita_slm_server.slm_server:app --host 127.0.0.1 --port 8000

或使用 uvicorn（需確認 uvicorn 安裝在當前 venv 並指向正確 Python）：
	uvicorn src.anita_slm_server.slm_server:app --host 127.0.0.1 --port 8000
"""


from __future__ import annotations

# isort: skip_file

import asyncio
import base64
import contextlib
import gc
import importlib.util
import logging
import math
import os
import re
import threading
import time
import uuid
import warnings
import json
from array import array
from dataclasses import dataclass
from pathlib import Path
import platform
from typing import Any, Literal, Protocol, cast
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from .qwenchat import QwenChatAPI, QwenChatConfig
from .chat_backends import load_backend
from .memory_monitor import get_memory_status, get_gpu_memory_status
from .performance_config import PERFORMANCE_CONFIG

_DEFAULT_PENDING_REQUESTS = int(cast(int, PERFORMANCE_CONFIG["MAX_CONCURRENT_REQUESTS"])) * 2 or 2

ENV = {
	"MODEL_PATH": os.getenv("MODEL_PATH", "./models/qwen/Qwen2.5-0.5B-Instruct"),
	"MODEL_QUANTIZATION": os.getenv("MODEL_QUANTIZATION", None),
	"TORCH_DTYPE": os.getenv("TORCH_DTYPE", "auto"),
	"COMPILE_MODEL": os.getenv("COMPILE_MODEL", "0") in {"1", "true", "TRUE", "True"},
	"LOW_CPU_MEM_USAGE": os.getenv("LOW_CPU_MEM_USAGE", "1") in {"1", "true", "TRUE", "True"},
	"MAX_INPUT_TOKENS": int(os.getenv("MAX_INPUT_TOKENS", "2048")),
	"MAX_INPUT_CHARS": int(os.getenv("MAX_INPUT_CHARS", "12000")),
	"MODEL_UNLOAD_ENABLED": os.getenv("MODEL_UNLOAD_ENABLED", "1") in {"1", "true", "TRUE", "True"},
	"MODEL_IDLE_TIMEOUT": int(os.getenv("MODEL_IDLE_TIMEOUT", "3600")),
	"MODEL_UNLOAD_CHECK_INTERVAL": int(os.getenv("MODEL_UNLOAD_CHECK_INTERVAL", "600")),
	"TORCH_NUM_THREADS": int(os.getenv("TORCH_NUM_THREADS", "0")),
	"MODEL_DEVICE_MAP": os.getenv("MODEL_DEVICE_MAP", "auto"),
	"ENABLE_TF32": os.getenv("ENABLE_TF32", "1") in {"1", "true", "TRUE", "True"},
	"EMBEDDING_DEVICE": os.getenv("EMBEDDING_DEVICE", "auto"),
	"AUTO_MEMORY_MANAGEMENT": PERFORMANCE_CONFIG["AUTO_MEMORY_MANAGEMENT"],
	"MAX_CONCURRENT_REQUESTS": PERFORMANCE_CONFIG["MAX_CONCURRENT_REQUESTS"],
	"GLOBAL_SEED": int(os.getenv("GLOBAL_SEED", "0")) if os.getenv("GLOBAL_SEED") else None,
	"DEFAULT_SYSTEM_PROMPT": os.getenv("DEFAULT_SYSTEM_PROMPT", None),
	"STRICT_MEMORY_CHECK": os.getenv("STRICT_MEMORY_CHECK", "0") in {"1", "true", "TRUE", "True"},
	"MEMORY_EXPANSION_FP": float(os.getenv("MEMORY_EXPANSION_FP", "2.2")),
	"MEMORY_EXPANSION_8BIT": float(os.getenv("MEMORY_EXPANSION_8BIT", "1.3")),
	"MEMORY_EXPANSION_4BIT": float(os.getenv("MEMORY_EXPANSION_4BIT", "1.15")),
	"MIN_REMAIN_AFTER_LOAD_GB": float(os.getenv("MIN_REMAIN_AFTER_LOAD_GB", "1.0")),
	"MAX_COMPLETION_TOKENS": int(os.getenv("MAX_COMPLETION_TOKENS", "1024")),
	"REQUEST_QUEUE_TIMEOUT": float(os.getenv("REQUEST_QUEUE_TIMEOUT", "15.0")),
	"MAX_PENDING_REQUESTS": int(os.getenv("MAX_PENDING_REQUESTS", str(_DEFAULT_PENDING_REQUESTS))),
	"MAX_CHAT_MESSAGES": int(os.getenv("MAX_CHAT_MESSAGES", "64")),
	"MAX_CHAT_TOTAL_CHARS": int(os.getenv("MAX_CHAT_TOTAL_CHARS", "20000")),
	"MAX_EMBEDDING_INPUTS": int(os.getenv("MAX_EMBEDDING_INPUTS", "32")),
	"MAX_EMBEDDING_TOTAL_CHARS": int(os.getenv("MAX_EMBEDDING_TOTAL_CHARS", "20000")),
	"MAX_EMBEDDING_ITEM_CHARS": int(os.getenv("MAX_EMBEDDING_ITEM_CHARS", "8000")),
	# Embedding pipeline tuning
	"EMBEDDING_BATCH_SIZE": int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
	# Adaptive resource controls
	"ADAPTIVE_LIMITS": os.getenv("ADAPTIVE_LIMITS", "1") in {"1", "true", "TRUE", "True"},
	"ADAPTIVE_INTERVAL": int(os.getenv("ADAPTIVE_INTERVAL", "30")),
	"ADAPTIVE_HIGH": float(os.getenv("ADAPTIVE_HIGH", "0.82")),
	"ADAPTIVE_CRITICAL": float(os.getenv("ADAPTIVE_CRITICAL", "0.90")),
	"DEGRADE_TO_GGUF": os.getenv("DEGRADE_TO_GGUF", "1") in {"1", "true", "TRUE", "True"},
	# Security and middleware (all optional; disabled by default)
	"REQUIRE_API_KEY": os.getenv("REQUIRE_API_KEY", "0") in {"1", "true", "TRUE", "True"},
	"API_KEY": os.getenv("API_KEY"),
	# Comma-separated origins, e.g. "https://a.com,https://b.com"; empty string disables CORS middleware
	"CORS_ORIGINS": os.getenv("CORS_ORIGINS", ""),
	# Hard cap for incoming request body (bytes). Uses Content-Length header; if absent, check is skipped.
	"MAX_REQUEST_BYTES": int(os.getenv("MAX_REQUEST_BYTES", "1048576")),  # 1 MiB default
}

ENV["MAX_CONCURRENT_REQUESTS"] = max(1, int(ENV["MAX_CONCURRENT_REQUESTS"]))
ENV["MAX_COMPLETION_TOKENS"] = max(1, int(ENV["MAX_COMPLETION_TOKENS"]))
ENV["MAX_PENDING_REQUESTS"] = max(0, int(ENV["MAX_PENDING_REQUESTS"]))
ENV["MAX_CHAT_MESSAGES"] = max(1, int(ENV["MAX_CHAT_MESSAGES"]))
ENV["MAX_CHAT_TOTAL_CHARS"] = max(512, int(ENV["MAX_CHAT_TOTAL_CHARS"]))
ENV["MAX_EMBEDDING_INPUTS"] = max(1, int(ENV["MAX_EMBEDDING_INPUTS"]))
ENV["MAX_EMBEDDING_TOTAL_CHARS"] = max(512, int(ENV["MAX_EMBEDDING_TOTAL_CHARS"]))
ENV["MAX_EMBEDDING_ITEM_CHARS"] = max(128, int(ENV["MAX_EMBEDDING_ITEM_CHARS"]))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').propagate = False

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
	level=getattr(logging, LOG_LEVEL, logging.INFO),
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
# 專案名稱已更名為 anita_slm_server，統一 logger 名稱前綴
app_logger = logging.getLogger("anita_slm_server")

try:
	from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
	SentenceTransformer = None  # type: ignore

try:
	import numpy as np  # type: ignore
except Exception:  # pragma: no cover
	np = None  # type: ignore

try:
	import torch  # type: ignore
	if ENV["TORCH_NUM_THREADS"] > 0:
		torch.set_num_threads(ENV["TORCH_NUM_THREADS"])
		app_logger.info(f"已設定 PyTorch 執行緒數量為 {ENV['TORCH_NUM_THREADS']}")
	if ENV["ENABLE_TF32"] and hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[attr-defined]
		with contextlib.suppress(Exception):
			torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
		with contextlib.suppress(Exception):
			torch.set_float32_matmul_precision("high")  # type: ignore[attr-defined]
except ImportError:
	torch = None  # type: ignore

app = FastAPI(title="Qwen OpenAI-Compatible API")

# Optional: GZip compression for larger responses (harmless for tests)
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Optional: CORS if explicitly configured
if _cors_origins_raw := (ENV.get("CORS_ORIGINS") or "").strip():
	if _origins := [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]:
		app_logger.info(f"啟用 CORS，允許來源: {_origins}")
		app.add_middleware(
			CORSMiddleware,
			allow_origins=_origins,
			allow_credentials=True,
			allow_methods=["*"],
			allow_headers=["*"]
		)

# Optional: lightweight request size limiter and API key guard
_OPEN_ENDPOINTS = {"/health", "/health/memory", "/health/detailed", "/metrics"}

@app.middleware("http")
async def _security_guard(request: Request, call_next):  # pragma: no cover - covered indirectly by tests
	# Content-Length based size check (skip if header missing)
	try:
		max_bytes = int(ENV.get("MAX_REQUEST_BYTES", 0) or 0)
	except Exception:
		max_bytes = 0
	if max_bytes > 0:
		cl = request.headers.get("content-length")
		if cl and cl.isdigit() and int(cl) > max_bytes:
			return JSONResponse({"detail": f"request body too large (>{max_bytes} bytes)"}, status_code=413)

	# API key guard (optional)
	if ENV.get("REQUIRE_API_KEY") and request.url.path not in _OPEN_ENDPOINTS:
		provided = request.headers.get("x-api-key") or request.headers.get("authorization")
		# Support simple "Bearer <key>" or raw key
		if provided and provided.lower().startswith("bearer "):
			provided = provided[7:].strip()
		if not provided or provided != ENV.get("API_KEY"):
			return JSONResponse({"detail": "unauthorized"}, status_code=401)

	return await call_next(request)

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "transformers").lower()
class _BackendProto(Protocol):
	def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
	capabilities: dict[str, bool] | None  # optional capability hints

class _MainModelState:
	def __init__(self):
		self.backend: _BackendProto | None = None
		self.loaded_time: float | None = None
		self.last_used_time: float | None = None

MODEL_STATE = _MainModelState()
qwen_init_lock = threading.Lock()


class _TransformersBackendWrapper:
	def __init__(self, inner: QwenChatAPI):
		self.inner = inner
		self.capabilities: dict[str, bool] | None = {
			"tools": True,
			"tool_choice": True,
			"max_input_tokens": True,
			"repetition_penalty": True,
			"stream": False,
		}

	def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
		max_new_tokens = kwargs.get("max_new_tokens", 256)
		temperature = kwargs.get("temperature", 0.7)
		top_p = kwargs.get("top_p", 0.9)
		repetition_penalty = kwargs.get("repetition_penalty", 1.1)
		tools = kwargs.get("tools")
		tool_choice = kwargs.get("tool_choice")
		max_input_tokens = kwargs.get("max_input_tokens")
		result = self.inner.chat_messages(
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
			"completion_tokens": result.prompt_tokens if getattr(result, "completion_tokens", None) is None else result.completion_tokens,
			"total_tokens": result.total_tokens,
		}


def _build_transformers_backend(env_cfg: dict[str, Any]) -> _TransformersBackendWrapper:
	quantization = env_cfg.get("MODEL_QUANTIZATION")
	if isinstance(quantization, str) and not quantization.strip():
		quantization = None
	global_seed = env_cfg.get("GLOBAL_SEED")
	if isinstance(global_seed, str):
		global_seed = int(global_seed) if global_seed.strip().isdigit() else None
	device_map = env_cfg.get("MODEL_DEVICE_MAP", "auto")
	if isinstance(device_map, str) and device_map.lower() in {"", "none"}:
		device_map = None
	dtype = env_cfg.get("TORCH_DTYPE", "auto")
	if isinstance(dtype, str) and not dtype:
		dtype = "auto"
	low_cpu_mem_usage = env_cfg.get("LOW_CPU_MEM_USAGE", True)
	if isinstance(low_cpu_mem_usage, str):
		low_cpu_mem_usage = low_cpu_mem_usage.lower() in {"1", "true", "yes"}
	config = QwenChatConfig(
		model_path=str(env_cfg.get("MODEL_PATH", ENV["MODEL_PATH"])),
		device_map=device_map,
		dtype=dtype,
		quantization=quantization,
		compile_model=bool(env_cfg.get("COMPILE_MODEL", False)),
		max_input_tokens=int(env_cfg.get("MAX_INPUT_TOKENS", ENV["MAX_INPUT_TOKENS"])),
		default_system_prompt=env_cfg.get("DEFAULT_SYSTEM_PROMPT"),
		global_seed=global_seed,
		low_cpu_mem_usage=bool(low_cpu_mem_usage),
	)
	backend_obj = QwenChatAPI(config)
	return _TransformersBackendWrapper(backend_obj)


def _is_memory_related_error(message: str) -> bool:
	lower = message.lower()
	return (
		"1455" in lower
		or "page file" in lower
		or "pagefile" in lower
		or "頁面文件" in lower
		or "页面文件" in message
		or ("memory" in lower and any(token in lower for token in ("fail", "error", "insufficient", "unable", "not enough", "allocate")))
	)


def _bitsandbytes_available() -> bool:
	with contextlib.suppress(Exception):
		return importlib.util.find_spec("bitsandbytes") is not None
	return False


def _locate_gguf_model() -> str | None:
	if env_override := os.getenv("GGUF_MODEL_PATH"):
		override_path = Path(env_override).expanduser()
		if override_path.is_file():
			return str(override_path)
	model_path = ENV.get("MODEL_PATH")
	if isinstance(model_path, str):
		current = Path(model_path)
		if current.is_file() and current.suffix.lower() == ".gguf":
			return str(current)
		candidates = [current.parent / "Qwen2.5-0.5B-Instruct-GGUF"]
	else:
		candidates = []
	candidates.extend(
		[
			Path("models/qwen/Qwen2.5-0.5B-Instruct-GGUF"),
			Path("models/Qwen2.5-0.5B-Instruct-GGUF"),
			Path("models/qwen"),
		]
	)
	for candidate in candidates:
		if candidate.is_file() and candidate.suffix.lower() == ".gguf":
			return str(candidate)
		if candidate.is_dir():
			for item in candidate.glob("*.gguf"):
				return str(item)
	return None


def _prepare_env_for_plan(overrides: dict[str, Any]) -> dict[str, Any]:
	env_cfg = dict(ENV)
	env_cfg.update(overrides)
	if "MODEL_PATH" in env_cfg and env_cfg["MODEL_PATH"] is not None:
		env_cfg["MODEL_PATH"] = str(env_cfg["MODEL_PATH"])
	else:
		env_cfg["MODEL_PATH"] = str(ENV["MODEL_PATH"])
	return env_cfg


def _load_backend_with_env(backend_name: str, env_cfg: dict[str, Any], load_kwargs: dict[str, Any] | None = None) -> _BackendProto:
	if load_kwargs is None:
		load_kwargs = {}
	if backend_name == "transformers":
		return cast(_BackendProto, _build_transformers_backend(env_cfg))
	return cast(
		_BackendProto,
		load_backend(
			model_path=str(env_cfg["MODEL_PATH"]),
			backend=backend_name,
			**load_kwargs,
		),
	)


def _record_successful_plan(env_cfg: dict[str, Any], backend_name: str) -> None:
	globals()["MODEL_BACKEND"] = backend_name
	ENV.update(env_cfg)
	os.environ["MODEL_BACKEND"] = backend_name
	for key in ("MODEL_PATH", "MODEL_QUANTIZATION", "COMPILE_MODEL", "MAX_INPUT_TOKENS", "TORCH_DTYPE", "MODEL_DEVICE_MAP", "LOW_CPU_MEM_USAGE"):
		if key not in env_cfg:
			continue
		value = env_cfg[key]
		if value is None:
			with contextlib.suppress(KeyError):
				del os.environ[key]
		elif isinstance(value, bool):
			os.environ[key] = "1" if value else "0"
		else:
			os.environ[key] = str(value)


def _build_memory_fallback_plans() -> list[dict[str, Any]]:
	plans: list[dict[str, Any]] = []
	if MODEL_BACKEND == "transformers":
		if ENV.get("COMPILE_MODEL"):
			plans.append(
				{
					"name": "disable torch.compile",
					"backend": "transformers",
					"overrides": {"COMPILE_MODEL": False},
				}
			)
		is_windows = platform.system().lower().startswith("win")
		quant_setting = ENV.get("MODEL_QUANTIZATION")
		if not is_windows and (not quant_setting or str(quant_setting).lower() == "none") and _bitsandbytes_available():
			plans.append(
				{
					"name": "enable 4bit quantization",
					"backend": "transformers",
					"overrides": {"MODEL_QUANTIZATION": "4bit", "COMPILE_MODEL": False},
				}
			)
	if gguf_path := _locate_gguf_model():
		max_tokens = int(ENV.get("MAX_INPUT_TOKENS", 2048) or 2048)
		max_tokens = max(256, min(max_tokens, 1024))
		reduced_ctx_tokens = 512
		batch_hint = max(32, min(128, max_tokens // 4))
		plans.append(
			{
				"name": "switch to llama.cpp GGUF Q4_K_M",
				"backend": "llama.cpp",
				"overrides": {
					"MODEL_PATH": gguf_path,
					"MODEL_QUANTIZATION": None,
					"COMPILE_MODEL": False,
					"MAX_INPUT_TOKENS": max_tokens,
				},
				"load_kwargs": {"n_ctx": max_tokens, "n_batch": batch_hint},
			}
		)
		if max_tokens > reduced_ctx_tokens:
			plans.append(
				{
					"name": "llama.cpp GGUF Q4_K_M (reduced context)",
					"backend": "llama.cpp",
					"overrides": {
						"MODEL_PATH": gguf_path,
						"MODEL_QUANTIZATION": None,
						"COMPILE_MODEL": False,
						"MAX_INPUT_TOKENS": reduced_ctx_tokens,
					},
					"load_kwargs": {"n_ctx": reduced_ctx_tokens, "n_batch": 64},
				}
			)
	return plans

embedding_model_cache: dict[str, Any] = {}
_embedding_last_used_times: dict[str, float] = {}
_EMBEDDING_CACHE_LOCK = threading.RLock()

_metrics_lock = threading.Lock()
_metrics: dict[str, float] = {
	"requests_total": 0,
	"chat_requests_total": 0,
	"embeddings_requests_total": 0,
	"errors_total": 0,
	"errors_memory": 0,
	"errors_timeout": 0,
	"errors_validation": 0,
	"chat_latency_sum_seconds": 0.0,
	"chat_latency_max_seconds": 0.0,
	"chat_latency_count": 0.0,
	"chat_tokens_total": 0.0,
	"chat_tokens_per_sec_sum": 0.0,
	"chat_tokens_per_sec_count": 0.0,
	"chat_tokens_per_sec_max": 0.0,
	"queue_rejections_total": 0.0,
	"queue_timeout_total": 0.0,
	"queue_wait_time_sum_seconds": 0.0,
	"queue_wait_time_count": 0.0,
	"embedding_fallbacks_total": 0.0,
}

def _metrics_inc(key: str, val: float = 1.0):
	with _metrics_lock:
		_metrics[key] = _metrics.get(key, 0) + val

def _metrics_observe_chat_latency(lat: float):
	with _metrics_lock:
		_metrics["chat_latency_sum_seconds"] += lat
		_metrics["chat_latency_max_seconds"] = max(_metrics["chat_latency_max_seconds"], lat)
		_metrics["chat_latency_count"] = _metrics.get("chat_latency_count", 0.0) + 1.0

def _metrics_observe_chat_tokens(total_tokens: float, latency: float):
	if total_tokens <= 0:
		return
	lat = max(latency, 1e-6)
	throughput = total_tokens / lat
	with _metrics_lock:
		_metrics["chat_tokens_total"] = _metrics.get("chat_tokens_total", 0.0) + total_tokens
		_metrics["chat_tokens_per_sec_sum"] = _metrics.get("chat_tokens_per_sec_sum", 0.0) + throughput
		_metrics["chat_tokens_per_sec_count"] = _metrics.get("chat_tokens_per_sec_count", 0.0) + 1.0
		_metrics["chat_tokens_per_sec_max"] = max(_metrics.get("chat_tokens_per_sec_max", 0.0), throughput)

def _metrics_observe_queue_wait(wait_seconds: float):
	if wait_seconds < 0:
		return
	with _metrics_lock:
		_metrics["queue_wait_time_sum_seconds"] = _metrics.get("queue_wait_time_sum_seconds", 0.0) + wait_seconds
		_metrics["queue_wait_time_count"] = _metrics.get("queue_wait_time_count", 0.0) + 1.0


class _QueueFullError(RuntimeError):
	pass


class _QueueTimeoutError(RuntimeError):
	pass


class _RequestLimiter:
	def __init__(self, max_concurrent: int, max_pending: int, acquire_timeout: float):
		self._max_concurrent = max(1, max_concurrent)
		self._max_pending = max(0, max_pending)
		self._timeout = max(0.1, acquire_timeout)
		self._semaphore = asyncio.Semaphore(self._max_concurrent)
		self._lock = asyncio.Lock()
		self._waiting = 0
		# dynamic ceilings default to static
		self._dyn_max_concurrent = self._max_concurrent
		self._dyn_max_pending = self._max_pending

	def _active(self) -> int:
		available = int(getattr(self._semaphore, "_value", 0) or 0)
		return max(0, self._max_concurrent - available)

	async def set_dynamic_limits(self, max_concurrent: int | None = None, max_pending: int | None = None) -> None:
		async with self._lock:
			if max_concurrent is not None:
				self._dyn_max_concurrent = max(1, min(self._max_concurrent, int(max_concurrent)))
			if max_pending is not None:
				self._dyn_max_pending = max(0, min(self._max_pending, int(max_pending)))

	async def _acquire(self) -> None:
		start = time.perf_counter()
		async with self._lock:
			available = getattr(self._semaphore, "_value", 0)
			# enforce dynamic pending cap
			if self._dyn_max_pending == 0 and available <= 0:
				raise _QueueFullError("request queue is full")
			if self._dyn_max_pending > 0 and self._waiting >= self._dyn_max_pending:
				raise _QueueFullError("request queue is full")
			self._waiting += 1
		try:
			deadline = start + self._timeout
			# pre-wait until under dynamic concurrent cap
			while True:
				active = self._active()
				if active < self._dyn_max_concurrent:
					break
				remaining = deadline - time.perf_counter()
				if remaining <= 0:
					raise _QueueTimeoutError("request queue wait timed out")
				await asyncio.sleep(min(0.05, remaining))
			await asyncio.wait_for(self._semaphore.acquire(), timeout=max(0.0, deadline - time.perf_counter()))
			wait_time = time.perf_counter() - start
			_metrics_observe_queue_wait(wait_time)
		except asyncio.TimeoutError as exc:
			raise _QueueTimeoutError("request queue wait timed out") from exc
		finally:
			async with self._lock:
				self._waiting = max(0, self._waiting - 1)

	def _release(self) -> None:
		self._semaphore.release()

	def __call__(self) -> _LimiterContext:
		return _LimiterContext(self)

	@property
	def waiting(self) -> int:
		return self._waiting

	@property
	def max_concurrent(self) -> int:
		return self._max_concurrent

	@property
	def max_pending(self) -> int:
		return self._max_pending

	@property
	def dyn_max_concurrent(self) -> int:
		return self._dyn_max_concurrent

	@property
	def dyn_max_pending(self) -> int:
		return self._dyn_max_pending

	@property
	def timeout(self) -> float:
		return self._timeout


class _LimiterContext:
	def __init__(self, limiter: _RequestLimiter):
		self._limiter = limiter
		self._acquired = False

	async def __aenter__(self):
		await self._limiter._acquire()
		self._acquired = True
		return self

	async def __aexit__(self, exc_type, exc, tb):
		if self._acquired:
			self._limiter._release()
			self._acquired = False
		return False


REQUEST_LIMITER = _RequestLimiter(
	max_concurrent=ENV["MAX_CONCURRENT_REQUESTS"],
	max_pending=ENV["MAX_PENDING_REQUESTS"],
	acquire_timeout=ENV["REQUEST_QUEUE_TIMEOUT"],
)

@app.get("/metrics")
def metrics():
	lines = []
	with _metrics_lock:
		metrics_snapshot = dict(_metrics)
	lines.extend(f"agent_slm_{k} {v}" for k, v in metrics_snapshot.items())
	if latency_count := metrics_snapshot.get("chat_latency_count", 0.0):
		avg_latency = metrics_snapshot.get("chat_latency_sum_seconds", 0.0) / latency_count
		lines.append(f"agent_slm_chat_latency_avg_seconds {avg_latency}")
	if throughput_count := metrics_snapshot.get("chat_tokens_per_sec_count", 0.0):
		avg_tps = metrics_snapshot.get("chat_tokens_per_sec_sum", 0.0) / throughput_count
		lines.append(f"agent_slm_chat_tokens_per_sec_avg {avg_tps}")
	if queue_wait_count := metrics_snapshot.get("queue_wait_time_count", 0.0):
		avg_wait = metrics_snapshot.get("queue_wait_time_sum_seconds", 0.0) / queue_wait_count
		lines.append(f"agent_slm_queue_wait_time_avg_seconds {avg_wait}")
	for gpu in get_gpu_memory_status():
		idx = gpu.get("index", 0)
		name = str(gpu.get("name", "cuda"))
		sanitized_name = name.replace('"', "'")
		labels = f'index="{idx}",name="{sanitized_name}"'
		if (total_bytes := gpu.get("total_bytes")) is not None:
			lines.append(f"agent_slm_gpu_memory_total_bytes{{{labels}}} {total_bytes}")
		if (free_bytes := gpu.get("free_bytes")) is not None:
			lines.append(f"agent_slm_gpu_memory_free_bytes{{{labels}}} {free_bytes}")
		if (allocated_bytes := gpu.get("allocated_bytes")) is not None:
			lines.append(f"agent_slm_gpu_memory_allocated_bytes{{{labels}}} {allocated_bytes}")
		if (reserved_bytes := gpu.get("reserved_bytes")) is not None:
			lines.append(f"agent_slm_gpu_memory_reserved_bytes{{{labels}}} {reserved_bytes}")
	if MODEL_STATE.loaded_time is not None:
		lines.append(f"agent_slm_model_loaded_timestamp {MODEL_STATE.loaded_time}")
	return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

def _discover_embedding_models() -> dict[str, str]:
	base_paths = [Path("models/embedding"), Path("models")]
	mapping: dict[str, str] = {}
	for base in base_paths:
		if not base.exists():
			continue
		for sub in base.iterdir():
			if not sub.is_dir():
				continue
			if (sub / "config_sentence_transformers.json").exists() or (sub / "tokenizer.json").exists():
				mapping[sub.name] = str(sub)
	return mapping

class _EmbeddingState:
	def __init__(self):
		self.map: dict[str, str] = _discover_embedding_models()
		self.last_refresh: float = time.time()

EMBEDDINGS = _EmbeddingState()
DEFAULT_EMBEDDING_MODEL_ID = "paraphrase-MiniLM-L6-v2"
_EMBEDDING_MAP_LOCK = threading.Lock()
_EMBEDDING_MAP_TTL_SECONDS = 300

def _resolve_embedding_device() -> str:
	device_cfg = ENV.get("EMBEDDING_DEVICE", "auto")
	if device_cfg and device_cfg.lower() != "auto":
		return device_cfg
	if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[attr-defined]
		return "cuda"
	return "cpu"

def get_embedding_model_map(force: bool = False) -> dict[str, str]:
	now = time.time()
	if force or (now - EMBEDDINGS.last_refresh > _EMBEDDING_MAP_TTL_SECONDS):
		with _EMBEDDING_MAP_LOCK:
			if force or (time.time() - EMBEDDINGS.last_refresh > _EMBEDDING_MAP_TTL_SECONDS):
				EMBEDDINGS.map = _discover_embedding_models()
				EMBEDDINGS.last_refresh = time.time()
	return EMBEDDINGS.map

def unload_models_if_idle():
	current_time = time.time()
	idle_timeout = ENV["MODEL_IDLE_TIMEOUT"]
	if not ENV["MODEL_UNLOAD_ENABLED"]:
		return
	if MODEL_STATE.backend is not None and MODEL_STATE.last_used_time is not None:
		idle_seconds = current_time - MODEL_STATE.last_used_time
		if idle_seconds > idle_timeout:
			with qwen_init_lock:
				if MODEL_STATE.backend is not None:
					_extracted_from_unload_models_if_idle_11(idle_seconds)
	models_to_unload: list[str] = []
	for model_id, last_used in list(_embedding_last_used_times.items()):
		idle_seconds = current_time - last_used
		if idle_seconds > idle_timeout and model_id in embedding_model_cache:
			models_to_unload.append(model_id)
	for model_id in models_to_unload:
		try:
			app_logger.info(f"嵌入模型 '{model_id}' 閒置，卸載中")
			with _EMBEDDING_CACHE_LOCK:
				embedding_model_cache.pop(model_id, None)
				_embedding_last_used_times.pop(model_id, None)
			gc.collect()
		except Exception as e:
			app_logger.error(f"卸載嵌入模型 '{model_id}' 時發生錯誤: {e}")


# TODO Rename this here and in `unload_models_if_idle`
def _extracted_from_unload_models_if_idle_11(idle_seconds):
	app_logger.info(f"主模型已閒置 {idle_seconds:.1f} 秒，正在卸載...")
	MODEL_STATE.backend = None
	MODEL_STATE.loaded_time = None
	MODEL_STATE.last_used_time = None
	gc.collect()
	with contextlib.suppress(Exception):
		if torch and torch.cuda.is_available():  # type: ignore[attr-defined]
			torch.cuda.empty_cache()
	app_logger.info("主模型已成功卸載")

@app.on_event("startup")
async def startup_tasks():  # noqa: PLR0915
	mapping = get_embedding_model_map(force=True)
	app_logger.info(f"可用 embedding 模型: {list(mapping.keys())}")
	app_logger.info(f"預設 embedding 模型: {DEFAULT_EMBEDDING_MODEL_ID}")
	app_logger.info(f"模型路徑: {ENV['MODEL_PATH']}")
	app_logger.info(f"量化模式: {ENV['MODEL_QUANTIZATION'] or '無'}")
	if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[attr-defined]
		for gpu in get_gpu_memory_status():
			app_logger.info(
				f"GPU[{gpu['index']}] {gpu['name']} total={gpu['total_gb']}GB free={gpu['free_gb']}GB allocated={gpu['allocated_gb']}GB"
			)
	else:
		app_logger.info("未檢測到可用 GPU，將使用 CPU 路徑")
	queue_desc = "不允許排隊" if REQUEST_LIMITER.max_pending == 0 else f"排隊上限 {REQUEST_LIMITER.max_pending}"
	app_logger.info(
		f"併發限制: 最大同時請求 {REQUEST_LIMITER.max_concurrent}，{queue_desc}，排隊逾時 {REQUEST_LIMITER.timeout:.1f}s"
	)
	if ENV["MODEL_UNLOAD_ENABLED"]:
		app_logger.info(f"模型卸載啟用，閒置超時: {ENV['MODEL_IDLE_TIMEOUT']} 秒")
		async def model_unload_monitor():
			try:
				while True:
					await asyncio.sleep(ENV["MODEL_UNLOAD_CHECK_INTERVAL"])
					unload_models_if_idle()
			except asyncio.CancelledError:
				return
			except Exception as e:  # pragma: no cover
				app_logger.error(f"模型卸載監控錯誤: {e}")
		asyncio.get_event_loop().create_task(model_unload_monitor())

	# Adaptive resource-based limiter and optional backend degrade
	if ENV.get("ADAPTIVE_LIMITS"):
		async def resource_manager():
			critical_hits = 0
			degraded_once = False
			interval = int(ENV.get("ADAPTIVE_INTERVAL", 30) or 30)
			while True:
				await asyncio.sleep(interval)
				status = get_memory_status()
				usage = 1.0
				with contextlib.suppress(Exception):
					usage = float(status.get("current_usage_percent", 1.0)) # type: ignore
				if usage >= float(ENV.get("ADAPTIVE_CRITICAL", 0.90)):
					target_conc = max(1, REQUEST_LIMITER.max_concurrent // 4)
					target_pending = 0
					critical_hits += 1
				elif usage >= float(ENV.get("ADAPTIVE_HIGH", 0.82)):
					target_conc = max(1, REQUEST_LIMITER.max_concurrent // 2)
					target_pending = max(0, REQUEST_LIMITER.max_pending // 2)
					critical_hits = 0
				else:
					target_conc = REQUEST_LIMITER.max_concurrent
					target_pending = REQUEST_LIMITER.max_pending
					critical_hits = 0
				await REQUEST_LIMITER.set_dynamic_limits(target_conc, target_pending)
				app_logger.debug(f"[adaptive] mem={usage:.1%} dyn_limits: conc={REQUEST_LIMITER.dyn_max_concurrent} pending={REQUEST_LIMITER.dyn_max_pending}")
				# optional degrade to GGUF when sustained critical
				if (not degraded_once) and ENV.get("DEGRADE_TO_GGUF") and critical_hits >= 3:
					gguf_path = _locate_gguf_model()
					if gguf_path and globals().get("MODEL_BACKEND") == "transformers":
						if REQUEST_LIMITER.waiting == 0 and REQUEST_LIMITER._active() == 0:
							app_logger.warning("[adaptive] Sustained critical memory. Switching backend to llama.cpp GGUF.")
							with qwen_init_lock:
								globals()["MODEL_BACKEND"] = "llama.cpp"
								os.environ["MODEL_BACKEND"] = "llama.cpp"
								ENV["MODEL_PATH"] = gguf_path
								os.environ["MODEL_PATH"] = gguf_path
								MODEL_STATE.backend = None
								MODEL_STATE.loaded_time = None
								MODEL_STATE.last_used_time = None
								gc.collect()
								with contextlib.suppress(Exception):
									if torch and torch.cuda.is_available():
										torch.cuda.empty_cache()
								degraded_once = True
		asyncio.get_event_loop().create_task(resource_manager())

class ReloadEmbeddingsResponse(BaseModel):
	refreshed: bool
	models: list[str]
	default_model: str
	ttl_seconds: int

@app.post("/v1/reload-embeddings", response_model=ReloadEmbeddingsResponse)
def reload_embeddings():
	mapping = get_embedding_model_map(force=True)
	return ReloadEmbeddingsResponse(
		refreshed=True,
		models=list(mapping.keys()),
		default_model=DEFAULT_EMBEDDING_MODEL_ID,
		ttl_seconds=_EMBEDDING_MAP_TTL_SECONDS,
	)

try:
	import tiktoken  # type: ignore
	_tiktoken_enc = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
	tiktoken = None  # type: ignore
	_tiktoken_enc = None

def _count_tokens(text: str) -> int:
	if _tiktoken_enc is not None:
		with contextlib.suppress(Exception):
			return len(_tiktoken_enc.encode(text))
	return max(1, math.ceil(len(text) / 4))

@lru_cache(maxsize=8192)
def _count_tokens_cached(text: str) -> int:
	return _count_tokens(text)

def _truncate_messages_by_tokens(messages: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
	if limit is None or limit <= 0:
		return messages[-1:] if messages else []
	total = 0
	kept: list[dict[str, Any]] = []
	for m in reversed(messages):
		content = m.get("content")
		if not isinstance(content, str):
			kept.append(m)
			continue
		length = _count_tokens(content)
		if total + length <= limit:
			kept.append(m)
			total += length
		else:
			take = max(0, limit - total)
			if take > 0:
				approx_ratio = take / length
				cut_chars = int(len(content) * approx_ratio)
				m2 = dict(m)
				m2["content"] = content[-cut_chars:]
				kept.append(m2)
				total += take
			break
	return list(reversed(kept))

def _truncate_messages_by_chars(messages: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
	"""Character-based truncation (approx) from the tail while preserving newest messages."""
	if limit is None or limit <= 0:
		return messages[-1:] if messages else []
	total = 0
	kept: list[dict[str, Any]] = []
	for m in reversed(messages):
		content = m.get("content")
		if not isinstance(content, str):
			kept.append(m)
			continue
		length = len(content)
		if total + length <= limit:
			kept.append(m)
			total += length
		else:
			take = max(0, limit - total)
			if take > 0:
				m2 = dict(m)
				m2["content"] = content[-take:]
				kept.append(m2)
				total += take
			break
	return list(reversed(kept))

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

def _extract_tool_calls(text: str) -> tuple[list[ToolCall], str]:
	"""Parse <tool_call>{ ... }</tool_call> blocks from model output.
	Returns (tool_calls, cleaned_content).
	"""
	tool_calls: list[ToolCall] = []
	idx = 0

	def _safe_json_loads(s: str):
		try:
			return json.loads(s)
		except Exception:  # noqa: BLE001
			return None

	for m in _TOOL_CALL_PATTERN.finditer(text):
		blob = m.group(1).strip()
		data = _safe_json_loads(blob) or {}
		name = data.get("name") or ""
		args = data.get("arguments", {})
		if isinstance(args, str):
			j = _safe_json_loads(args)
			if j is not None:
				args = j
		tool_calls.append(
			ToolCall(
				id=f"call_{uuid.uuid4().hex[:8]}_{idx}",
				function=ToolFunctionCall(
					name=name,
					arguments=args if isinstance(args, str) else json.dumps(args, ensure_ascii=False),
				),
			)
		)
		idx += 1

	content_clean = _TOOL_CALL_PATTERN.sub("", text).strip()
	return tool_calls, content_clean


def _validate_chat_request(req: ChatCompletionsRequest) -> int:
	if not req.messages:
		raise HTTPException(status_code=400, detail="messages 不可為空")
	if len(req.messages) > ENV["MAX_CHAT_MESSAGES"]:
		raise HTTPException(status_code=413, detail=f"messages 條目數超過上限 {ENV['MAX_CHAT_MESSAGES']}")
	total_chars = 0
	for message in req.messages:
		content = message.content
		if content is None:
			continue
		if not isinstance(content, str):
			raise HTTPException(status_code=400, detail="messages 內容必須為字串")
		total_chars += len(content)
	if total_chars > ENV["MAX_CHAT_TOTAL_CHARS"]:
		raise HTTPException(status_code=413, detail=f"messages 內容長度超過上限 {ENV['MAX_CHAT_TOTAL_CHARS']}")
	if req.max_tokens is not None and req.max_tokens <= 0:
		raise HTTPException(status_code=400, detail="max_tokens 必須大於 0")
	limit = ENV["MAX_COMPLETION_TOKENS"]
	if req.max_tokens is not None and req.max_tokens > limit:
		raise HTTPException(status_code=400, detail=f"max_tokens 超過上限 {limit}")
	requested = req.max_tokens if req.max_tokens is not None else min(limit, 512)
	return min(max(1, requested), limit)


def _validate_embedding_request(req: EmbeddingsRequest) -> list[str]:
	inputs = [req.input] if isinstance(req.input, str) else list(req.input)
	if not inputs:
		raise HTTPException(status_code=400, detail="input 不可為空")
	if len(inputs) > ENV["MAX_EMBEDDING_INPUTS"]:
		raise HTTPException(status_code=413, detail=f"input 條目數超過上限 {ENV['MAX_EMBEDDING_INPUTS']}")
	total_chars = 0
	for idx, item in enumerate(inputs):
		if not isinstance(item, str):
			raise HTTPException(status_code=400, detail=f"input[{idx}] 必須為字串")
		length = len(item)
		if length > ENV["MAX_EMBEDDING_ITEM_CHARS"]:
			raise HTTPException(status_code=413, detail=f"input[{idx}] 長度超過上限 {ENV['MAX_EMBEDDING_ITEM_CHARS']}")
		total_chars += length
	if total_chars > ENV["MAX_EMBEDDING_TOTAL_CHARS"]:
		raise HTTPException(status_code=413, detail=f"input 總長度超過上限 {ENV['MAX_EMBEDDING_TOTAL_CHARS']}")
	return inputs


def _is_cpu_oom_error(err: BaseException) -> bool:
	msg = str(err).lower()
	return ("not enough memory" in msg) or ("out of memory" in msg) or ("alloc_cpu" in msg)


@dataclass(frozen=True)
class _ChatRequestLimits:
	char_limit: int | None
	token_limit: int | None
	env_char_limit: int | None


def _apply_chat_truncation(
	messages: list[dict[str, Any]],
	req: ChatCompletionsRequest,
) -> tuple[list[dict[str, Any]], _ChatRequestLimits]:
	char_limit_env = ENV.get("MAX_INPUT_CHARS") or 0
	requested_char_limit = req.max_input_chars or char_limit_env
	if char_limit_env and requested_char_limit:
		requested_char_limit = min(requested_char_limit, char_limit_env)
	if requested_char_limit:
		messages = _truncate_messages_by_chars(messages, requested_char_limit)
	token_limit_env = ENV.get("MAX_INPUT_TOKENS") or 0
	req_token_cap = req.max_input_tokens or token_limit_env
	if token_limit_env and req_token_cap:
		req_token_cap = min(req_token_cap, token_limit_env)
	if req_token_cap:
		messages = _truncate_messages_by_tokens(messages, req_token_cap)
	limits = _ChatRequestLimits(
		char_limit=requested_char_limit or None,
		token_limit=req_token_cap or None,
		env_char_limit=char_limit_env or None,
	)
	return messages, limits


def _generate_with_fallback(
	messages: list[dict[str, Any]],
	req: ChatCompletionsRequest,
	effective_max_tokens: int,
	limits: _ChatRequestLimits,
) -> dict[str, Any]:
	backend_max_input_tokens = limits.token_limit
	try:
		return MODEL_STATE.backend.generate(  # type: ignore[union-attr]
			messages=messages,
			max_new_tokens=effective_max_tokens,
			temperature=req.temperature or 0.7,
			top_p=req.top_p or 0.9,
			repetition_penalty=req.repetition_penalty or 1.1,
			tools=[t.model_dump(exclude_none=True) for t in (req.tools or [])] if req.tools else None,
			tool_choice=req.tool_choice,
			max_input_tokens=backend_max_input_tokens,
		)
	except RuntimeError as err:
		if not _is_cpu_oom_error(err):
			raise
		fallback_chars_base = limits.char_limit or limits.env_char_limit or 12000
		fallback_chars = max(1000, fallback_chars_base // 2)
		messages_fb = _truncate_messages_by_chars(messages, fallback_chars)
		fallback_tokens = max(32, effective_max_tokens // 2)
		return MODEL_STATE.backend.generate(  # type: ignore[union-attr]
			messages=messages_fb,
			max_new_tokens=fallback_tokens,
			temperature=req.temperature or 0.7,
			top_p=req.top_p or 0.9,
			repetition_penalty=req.repetition_penalty or 1.1,
			tools=[t.model_dump(exclude_none=True) for t in (req.tools or [])] if req.tools else None,
			tool_choice=req.tool_choice,
			max_input_tokens=backend_max_input_tokens,
		)

@app.get("/health")
def health():
	return {"status": "ok"}

@app.get("/health/memory")
def health_memory():
	return get_memory_status()

@app.get("/health/detailed")
def health_detailed():
	memory_status = get_memory_status()
	model_status = {
		"main_model_loaded": MODEL_STATE.backend is not None,
		"embedding_models_count": len(embedding_model_cache),
		"embedding_models": list(embedding_model_cache.keys()),
	}
	MEMORY_SAFE_THRESHOLD = 0.9
	val = memory_status.get("current_usage_percent", 1.0)
	if isinstance(val, (int, float, str)):
		try:
			current_usage = float(val)
		except (ValueError, TypeError):
			current_usage = 1.0
	else:
		current_usage = 1.0
	memory_safe = current_usage < MEMORY_SAFE_THRESHOLD
	leak_detected = memory_status["leak_detected"]
	overall_status = "healthy" if memory_safe and not leak_detected else "warning"
	return {
		"status": overall_status,
		"memory": memory_status,
		"models": model_status,
		"warnings": {
			"high_memory_usage": not memory_safe,
			"memory_leak_detected": leak_detected,
		},
	}

class ToolFunctionCall(BaseModel):
	name: str
	arguments: Any

class ToolCall(BaseModel):
	id: str
	type: Literal["function"] = "function"
	function: ToolFunctionCall

class FunctionDef(BaseModel):
	name: str
	description: str | None = None
	parameters: dict[str, Any] | None = None

class Tool(BaseModel):
	type: Literal["function"]
	function: FunctionDef

class ChatMessage(BaseModel):
	role: Literal["system", "user", "assistant", "tool"]
	content: str | None = None
	name: str | None = None
	tool_call_id: str | None = None
	tool_calls: list[ToolCall] | None = None

class ChatCompletionsRequest(BaseModel):
	model: str = Field(default="qwen2.5-0.5b-instruct")
	messages: list[ChatMessage]
	temperature: float | None = 0.7
	top_p: float | None = 0.9
	max_tokens: int | None = 512
	repetition_penalty: float | None = 1.1
	stream: bool | None = False
	# OpenAI tool calling
	tools: list[Tool] | None = None
	tool_choice: Any | None = None  # "auto" | "none" | {"type":"function","function":{"name":...}}
	# input truncation options
	max_input_chars: int | None = ENV["MAX_INPUT_CHARS"]
	max_input_tokens: int | None = ENV["MAX_INPUT_TOKENS"]

class ChatChoice(BaseModel):
	index: int
	message: ChatMessage
	finish_reason: Literal["stop", "length", "tool_calls"] = "stop"

class ChatUsage(BaseModel):
	prompt_tokens: int = 0
	completion_tokens: int = 0
	total_tokens: int = 0

class ChatCompletionsResponse(BaseModel):
	id: str
	object: Literal["chat.completion"] = "chat.completion"
	created: int
	model: str
	choices: list[ChatChoice]
	usage: ChatUsage = ChatUsage()

class EmbeddingsRequest(BaseModel):
	model: str = Field(default=DEFAULT_EMBEDDING_MODEL_ID)
	input: str | list[str]
	encoding_format: Literal["float", "base64"] | None = Field(default="float")

class EmbeddingDataItem(BaseModel):
	object: Literal["embedding"] = "embedding"
	index: int
	embedding: list[float] | str

class EmbeddingsResponse(BaseModel):
	object: Literal["list"] = "list"
	data: list[EmbeddingDataItem]
	model: str
	usage: ChatUsage = ChatUsage()


def _process_chat_request(req: ChatCompletionsRequest, effective_max_tokens: int) -> ChatCompletionsResponse:
	start_t = time.time()
	_init_backend_if_needed()
	messages = [{"role": m.role, "content": m.content or ""} for m in req.messages]
	messages, limits = _apply_chat_truncation(messages, req)

	try:
		if MODEL_STATE.backend is None or not hasattr(MODEL_STATE.backend, "generate"):
			raise HTTPException(status_code=500, detail="backend 未初始化或不支援 generate")
		out = _generate_with_fallback(messages, req, effective_max_tokens, limits)
		text_out = out.get("text", "")
		tool_calls, content_clean = _extract_tool_calls(text_out)
		latency = time.time() - start_t
		_metrics_observe_chat_latency(latency)
		total_tokens = float(out.get("total_tokens", 0) or 0)
		_metrics_observe_chat_tokens(total_tokens, latency)
		now = int(time.time())
		if tool_calls:
			assistant_msg = ChatMessage(role="assistant", content=None, tool_calls=tool_calls)
			finish_reason: Literal["stop", "length", "tool_calls"] = "tool_calls"
		else:
			assistant_msg = ChatMessage(role="assistant", content=content_clean or text_out)
			finish_reason = "stop"
		return ChatCompletionsResponse(
			id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
			created=now,
			model=req.model,
			choices=[ChatChoice(index=0, message=assistant_msg, finish_reason=finish_reason)],
			usage=ChatUsage(
				prompt_tokens=out.get("prompt_tokens", 0),
				completion_tokens=out.get("completion_tokens", 0),
				total_tokens=out.get("total_tokens", 0),
			),
		)
	except HTTPException:
		raise
	except Exception as e:  # noqa: BLE001
		_metrics_inc("errors_total")
		app_logger.error(f"chat error: {e}")
		raise HTTPException(status_code=500, detail=f"chat failed: {e}") from e
	finally:
		MODEL_STATE.last_used_time = time.time()


def _process_embedding_request(req: EmbeddingsRequest, inputs: list[str]) -> EmbeddingsResponse:
	if SentenceTransformer is None:
		raise HTTPException(status_code=500, detail="sentence-transformers 未安裝")
	encoding_format = (req.encoding_format or "float").lower()
	if encoding_format not in {"float", "base64"}:
		raise HTTPException(status_code=400, detail="encoding_format 必須為 'float' 或 'base64'")
	mapping = get_embedding_model_map()
	requested_id = req.model
	model_key = requested_id if requested_id in mapping else DEFAULT_EMBEDDING_MODEL_ID
	if model_key not in mapping:
		raise HTTPException(status_code=404, detail=f"找不到嵌入模型: {model_key}")
	target_device = _resolve_embedding_device()
	model_obj = embedding_model_cache.get(model_key)
	if model_obj is None:
		with _EMBEDDING_CACHE_LOCK:
			model_obj = embedding_model_cache.get(model_key)
			if model_obj is None:
				try:
					app_logger.info(f"[embedding-load] {model_key} device={target_device}")
					model_obj = SentenceTransformer(mapping[model_key], device=target_device)  # type: ignore
					embedding_model_cache[model_key] = model_obj
				except Exception as e:  # noqa: BLE001
					_metrics_inc("errors_total")
					app_logger.exception(f"[embedding-load] 模型載入失敗 {model_key}: {e}")
					raise HTTPException(status_code=500, detail=f"嵌入模型載入失敗: {e}") from e
	current_device = getattr(model_obj, "device", None)
	if target_device and current_device != target_device and hasattr(model_obj, "to"):
		with contextlib.suppress(Exception):
			model_obj.to(target_device)
	_embedding_last_used_times[model_key] = time.time()

	def _serialize_vector(vec: Any) -> list[float] | str:
		if encoding_format == "base64":
			if np is not None:
				buffer = np.asarray(vec, dtype=np.float32).tobytes()
			else:
				values = vec.tolist() if hasattr(vec, "tolist") else list(vec)
				buffer = array("f", values).tobytes()
			return base64.b64encode(buffer).decode("ascii")
		values = vec.tolist() if hasattr(vec, "tolist") else list(vec)
		return [float(v) for v in values]

	def _encode_with_fallbacks(texts: list[str]) -> Any:
		"""Try encode with current device/batch, fallback on OOM: reduce batch, then CPU."""
		batch = int(ENV.get("EMBEDDING_BATCH_SIZE", 16) or 16)
		attempts = []
		# 1) current device, configured batch
		attempts.append((target_device, batch))
		# 2) current device, smaller batch
		attempts.append((target_device, max(1, batch // 2)))
		# 3) cpu device, small batch
		attempts.append(("cpu", min(8, max(1, batch // 2))))
		last_err: Exception | None = None
		for dev, bsz in attempts:
			try:
				if getattr(model_obj, "device", None) != dev and hasattr(model_obj, "to"):
					with contextlib.suppress(Exception):
						model_obj.to(dev)
				return model_obj.encode(texts, batch_size=bsz)
			except Exception as e:  # noqa: BLE001
				msg = str(e).lower()
				if "out of memory" in msg or "page file" in msg or "1455" in msg or "cannot allocate" in msg:
					_metrics_inc("embedding_fallbacks_total")
					last_err = e
					continue
				else:
					raise
		if last_err is not None:
			raise last_err
		return None

	try:
		vectors = _encode_with_fallbacks(inputs)
		data_items = [EmbeddingDataItem(index=i, embedding=_serialize_vector(vec)) for i, vec in enumerate(vectors)]
		# 改善 usage 統計：以 token 粗估輸入成本，避免使用 len(str(inputs)) 誤導。
		# completion_tokens 對 embeddings 無意義，保持為 0。
		prompt_token_estimate = 0
		for text in inputs:
			if isinstance(text, str):
				prompt_token_estimate += _count_tokens_cached(text)
		usage = ChatUsage(prompt_tokens=prompt_token_estimate, completion_tokens=0, total_tokens=prompt_token_estimate)
		return EmbeddingsResponse(data=data_items, model=model_key, usage=usage)
	except Exception as e:  # noqa: BLE001
		_metrics_inc("errors_total")
		app_logger.exception(f"[embeddings] 生成失敗 model={model_key}: {e}")
		raise HTTPException(status_code=500, detail=f"嵌入生成失敗: {e}") from e

@app.get("/v1/models")
def list_models():
	data = [
		{"id": "qwen2.5-0.5b-instruct", "object": "model", "owned_by": MODEL_BACKEND},
	]
	data.extend(
		{"id": mid, "object": "model", "owned_by": "local"}
		for mid in get_embedding_model_map().keys()
	)
	return {"object": "list", "data": data}

def _current_backend_capabilities() -> dict[str, bool]:
	# Default conservative capabilities
	caps: dict[str, bool] = {
		"tools": False,
		"tool_choice": False,
		"max_input_tokens": False,
		"repetition_penalty": False,
		"stream": False,
	}
	be = MODEL_STATE.backend
	if be is not None:
		c = getattr(be, "capabilities", None)
		if isinstance(c, dict):
			caps |= {k: bool(v) for k, v in c.items()}
	return caps

@app.get("/v1/capabilities")
def get_capabilities():
	main_model = {
		"id": "qwen2.5-0.5b-instruct",
		"backend": MODEL_BACKEND,
		"capabilities": _current_backend_capabilities(),
		"limits": {
			"max_input_tokens": ENV.get("MAX_INPUT_TOKENS"),
			"max_completion_tokens": ENV.get("MAX_COMPLETION_TOKENS"),
		},
		"loaded": MODEL_STATE.backend is not None,
	}
	embeddings = {
		"models": list(get_embedding_model_map().keys()),
		"capabilities": {
			"encoding_format_float": True,
			"encoding_format_base64": True,
		},
		"limits": {
			"max_inputs": ENV.get("MAX_EMBEDDING_INPUTS"),
			"max_total_chars": ENV.get("MAX_EMBEDDING_TOTAL_CHARS"),
			"max_item_chars": ENV.get("MAX_EMBEDDING_ITEM_CHARS"),
		},
	}
	return {"main_model": main_model, "embeddings": embeddings}

def _init_backend_if_needed():
	if MODEL_STATE.backend is not None:
		return
	with qwen_init_lock:
		if MODEL_STATE.backend is not None:
			return
		plans: list[dict[str, Any]] = [
			{
				"name": "default configuration",
				"backend": MODEL_BACKEND,
				"overrides": {},
			}
		]
		fallback_records: list[dict[str, str]] = []
		memory_failure = False
		idx = 0
		while idx < len(plans):
			plan = plans[idx]
			backend_name = plan.get("backend", MODEL_BACKEND)
			overrides = plan.get("overrides", {}) or {}
			load_kwargs = plan.get("load_kwargs", {}) or {}
			env_cfg = _prepare_env_for_plan(overrides)
			attempt_name = plan.get("name", f"plan-{idx}")
			app_logger.info(
				f"[backend-init] attempt '{attempt_name}': backend={backend_name} path={env_cfg['MODEL_PATH']}"
			)
			try:
				backend_instance = _load_backend_with_env(backend_name, env_cfg, load_kwargs)
				_record_successful_plan(env_cfg, backend_name)
				MODEL_STATE.backend = backend_instance  # type: ignore
				MODEL_STATE.loaded_time = time.time()
				app_logger.info(f"[backend-init] success via '{attempt_name}'")
				return
			except Exception as exc:  # noqa: BLE001
				msg = str(exc)
				fallback_records.append({"plan": attempt_name, "error": msg})
				if idx == 0:
					if _is_memory_related_error(msg):
						memory_failure = True
						app_logger.warning(
							f"[backend-init] default configuration failed due to memory constraints: {msg}"
						)
						fallback_plans = _build_memory_fallback_plans()
						if fallback_plans:
							plans.extend(fallback_plans)
						else:
							break
					else:
						app_logger.error(f"[backend-init] failed: {msg}")
						raise HTTPException(status_code=500, detail=f"後端載入失敗: {msg}") from exc
				else:
					app_logger.warning(f"[backend-init] fallback '{attempt_name}' failed: {msg}")
			idx += 1
		if memory_failure:
			suggestion = {
				"error": "模型/後端載入失敗 (內存或分頁檔不足)",
				"backend": MODEL_BACKEND,
				"original_error": fallback_records[0]["error"] if fallback_records else "",
				"auto_recovery_attempts": fallback_records[1:],
				"suggestions": [
					"增大 Pagefile 至 >=16GB",
					"嘗試 MODEL_BACKEND=llama.cpp + GGUF Q4_K_M",
					"設定 MODEL_QUANTIZATION=4bit 並安裝 bitsandbytes",
					"降低 MAX_INPUT_TOKENS 或關閉 COMPILE_MODEL",
				],
			}
			raise HTTPException(status_code=503, detail=suggestion) from None
		last_error = fallback_records[-1]["error"] if fallback_records else "未能載入模型"
		raise HTTPException(status_code=500, detail=f"後端載入失敗: {last_error}") from None

@app.post("/v1/chat/completions", response_model=ChatCompletionsResponse)
async def chat_completions(req: ChatCompletionsRequest):
	_metrics_inc("requests_total")
	_metrics_inc("chat_requests_total")
	try:
		effective_max_tokens = _validate_chat_request(req)
	except HTTPException:
		_metrics_inc("errors_validation")
		raise
	try:
		async with REQUEST_LIMITER():
			return await asyncio.to_thread(_process_chat_request, req, effective_max_tokens)
	except _QueueFullError:
		_metrics_inc("queue_rejections_total")
		_metrics_inc("errors_total")
		raise HTTPException(status_code=503, detail="server busy: request queue full") from None
	except _QueueTimeoutError:
		_metrics_inc("queue_timeout_total")
		_metrics_inc("errors_timeout")
		_metrics_inc("errors_total")
		raise HTTPException(status_code=503, detail="server busy: queue wait timeout") from None

@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
async def create_embeddings(req: EmbeddingsRequest):
	_metrics_inc("requests_total")
	_metrics_inc("embeddings_requests_total")
	try:
		inputs = _validate_embedding_request(req)
	except HTTPException:
		_metrics_inc("errors_validation")
		raise
	try:
		async with REQUEST_LIMITER():
			return await asyncio.to_thread(_process_embedding_request, req, inputs)
	except _QueueFullError:
		_metrics_inc("queue_rejections_total")
		_metrics_inc("errors_total")
		raise HTTPException(status_code=503, detail="server busy: request queue full") from None
	except _QueueTimeoutError:
		_metrics_inc("queue_timeout_total")
		_metrics_inc("errors_timeout")
		_metrics_inc("errors_total")
		raise HTTPException(status_code=503, detail="server busy: queue wait timeout") from None
