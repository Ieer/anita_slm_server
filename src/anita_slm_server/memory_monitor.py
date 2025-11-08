"""內存 / GPU 記憶體監控工具 (原 agent_slm_server.memory_monitor 已遷移).

提供:
 - 即時系統/程序記憶體快照
 - 簡易趨勢與疑似洩漏檢測
 - 垃圾回收與 CUDA 快取清理
 - GPU 顯存統計 (若可用)
"""

import contextlib
import gc
import logging
import time
from dataclasses import dataclass

import psutil

_MIN_SNAPSHOTS_FOR_TREND = 2
_MEMORY_DELTA_LOG_THRESHOLD_GB = 0.1

try:  # pragma: no cover - 動態可用性
	import torch  # type: ignore
except Exception:  # pragma: no cover
	torch = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
	timestamp: float
	total_memory_gb: float
	available_memory_gb: float
	used_memory_gb: float
	memory_percent: float
	swap_used_gb: float
	process_memory_gb: float


class MemoryMonitor:
	def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
		self.warning_threshold = warning_threshold
		self.critical_threshold = critical_threshold
		self.snapshots: list[MemorySnapshot] = []
		self.max_snapshots = 100

	def get_current_memory_info(self) -> MemorySnapshot:
		memory = psutil.virtual_memory()
		swap = psutil.swap_memory()
		process = psutil.Process()
		return MemorySnapshot(
			timestamp=time.time(),
			total_memory_gb=memory.total / (1024 ** 3),
			available_memory_gb=memory.available / (1024 ** 3),
			used_memory_gb=memory.used / (1024 ** 3),
			memory_percent=memory.percent / 100.0,
			swap_used_gb=swap.used / (1024 ** 3),
			process_memory_gb=process.memory_info().rss / (1024 ** 3),
		)

	def take_snapshot(self) -> MemorySnapshot:
		snapshot = self.get_current_memory_info()
		self.snapshots.append(snapshot)
		if len(self.snapshots) > self.max_snapshots:
			self.snapshots = self.snapshots[-self.max_snapshots :]
		self._check_memory_thresholds(snapshot)
		return snapshot

	def _check_memory_thresholds(self, snapshot: MemorySnapshot) -> None:
		if snapshot.memory_percent >= self.critical_threshold:
			logger.critical(
				f"內存使用率達到嚴重水平: {snapshot.memory_percent:.1%} (可用: {snapshot.available_memory_gb:.1f}GB)"
			)
		elif snapshot.memory_percent >= self.warning_threshold:
			logger.warning(
				f"內存使用率較高: {snapshot.memory_percent:.1%} (可用: {snapshot.available_memory_gb:.1f}GB)"
			)

	def get_memory_trend(self, minutes: int = 5) -> dict[str, float]:
		if len(self.snapshots) < _MIN_SNAPSHOTS_FOR_TREND:
			return {"trend": 0.0, "avg_usage": 0.0, "peak_usage": 0.0}
		cutoff_time = time.time() - (minutes * 60)
		recent = [s for s in self.snapshots if s.timestamp >= cutoff_time]
		if len(recent) < _MIN_SNAPSHOTS_FOR_TREND:
			recent = self.snapshots[-_MIN_SNAPSHOTS_FOR_TREND:]
		trend = recent[-1].memory_percent - recent[0].memory_percent
		usages = [s.memory_percent for s in recent]
		avg_usage = sum(usages) / len(usages)
		peak_usage = max(usages)
		return {
			"trend": trend,
			"avg_usage": avg_usage,
			"peak_usage": peak_usage,
			"sample_count": len(recent),
		}

	def detect_memory_leak(self, threshold_increase: float = 0.1, window_minutes: int = 10) -> bool:
		info = self.get_memory_trend(window_minutes)
		if info["trend"] > threshold_increase:
			logger.warning(
				f"檢測到可能的內存洩漏: {window_minutes} 分鐘內內存增長 {info['trend']:.1%}"
			)
			return True
		return False

	def force_garbage_collection(self) -> dict[str, float | int]:
		before = self.get_current_memory_info()
		collected = gc.collect()
		if torch is not None and torch.cuda.is_available():  # type: ignore[attr-defined]
			with contextlib.suppress(Exception):
				torch.cuda.empty_cache()  # type: ignore[attr-defined]
			with contextlib.suppress(Exception):
				torch.cuda.synchronize()  # type: ignore[attr-defined]
		after = self.get_current_memory_info()
		freed_gb = before.used_memory_gb - after.used_memory_gb
		logger.info(
			f"垃圾回收完成: 回收物件 {collected} 個, 釋放內存 {freed_gb:.2f}GB (使用率 {before.memory_percent:.1%} -> {after.memory_percent:.1%})"
		)
		return {
			"objects_collected": collected,
			"memory_freed_gb": freed_gb,
			"before_usage_percent": before.memory_percent,
			"after_usage_percent": after.memory_percent,
		}


_global_monitor = MemoryMonitor()


def get_memory_status() -> dict[str, object]:
	snapshot = _global_monitor.take_snapshot()
	trend = _global_monitor.get_memory_trend()
	status: dict[str, object] = {
		"current_usage_percent": snapshot.memory_percent,
		"available_gb": snapshot.available_memory_gb,
		"process_memory_gb": snapshot.process_memory_gb,
		"memory_trend": trend["trend"],
		"leak_detected": _global_monitor.detect_memory_leak(),
	}
	if gpu_info := _collect_gpu_memory_status():
		status["gpu_devices"] = gpu_info
	return status


def _collect_gpu_memory_status() -> list[dict[str, float | int | str]]:
	if torch is None or not torch.cuda.is_available():  # type: ignore[attr-defined]
		return []

	devices: list[dict[str, float | int | str]] = []
	for idx in range(torch.cuda.device_count()):  # type: ignore[attr-defined]
		with contextlib.suppress(Exception):
			torch.cuda.synchronize(idx)  # type: ignore[attr-defined]
		free_bytes = total_bytes = allocated_bytes = reserved_bytes = 0
		with contextlib.suppress(Exception):
			free_bytes, total_bytes = torch.cuda.mem_get_info(idx)  # type: ignore[attr-defined]
		with contextlib.suppress(Exception):
			allocated_bytes = torch.cuda.memory_allocated(idx)  # type: ignore[attr-defined]
		with contextlib.suppress(Exception):
			reserved_bytes = torch.cuda.memory_reserved(idx)  # type: ignore[attr-defined]
		name = f"cuda:{idx}"
		with contextlib.suppress(Exception):
			name = torch.cuda.get_device_name(idx)  # type: ignore[attr-defined]
		total_gb = total_bytes / (1024 ** 3)
		free_gb = free_bytes / (1024 ** 3)
		used_gb = total_gb - free_gb
		devices.append(
			{
				"index": idx,
				"name": name,
				"total_gb": round(total_gb, 3),
				"free_gb": round(free_gb, 3),
				"used_gb": round(used_gb, 3),
				"allocated_gb": round(allocated_bytes / (1024 ** 3), 3),
				"reserved_gb": round(reserved_bytes / (1024 ** 3), 3),
				"utilization": round((used_gb / total_gb) if total_gb else 0.0, 4),
				"total_bytes": int(total_bytes),
				"free_bytes": int(free_bytes),
				"allocated_bytes": int(allocated_bytes),
				"reserved_bytes": int(reserved_bytes),
			}
		)
	return devices


def get_gpu_memory_status() -> list[dict[str, float | int | str]]:
	"""公開 GPU 記憶體使用情況。"""
	return _collect_gpu_memory_status()


def memory_tracking(monitor: MemoryMonitor, operation_name: str = "operation"):
	"""簡易 context manager：追蹤區塊前後程序記憶體差異。"""
	before = monitor.take_snapshot()
	start = time.time()
	yield monitor
	after = monitor.take_snapshot()
	duration = time.time() - start
	delta = after.process_memory_gb - before.process_memory_gb
	if delta > _MEMORY_DELTA_LOG_THRESHOLD_GB:
		logger.info(
			f"{operation_name} 內存使用情況: 增長 {delta:.2f}GB, 耗時 {duration:.1f}s"
		)