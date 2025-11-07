import sys
import time
import traceback

sys.path.insert(0, r"e:\AI\anita_slm_server\src")

print("Importing QwenChatAPI...", flush=True)
from anita_slm_server.qwenchat import QwenChatAPI, QwenChatConfig  # noqa: E402
print("QwenChatAPI imported", flush=True)

cfg = QwenChatConfig()
print("Config prepared", cfg, flush=True)

try:
    print("Testing tokenizer load...", flush=True)
    tokenizer = None
    try:
        from transformers import AutoTokenizer as _AT  # noqa: E402
        tokenizer = _AT.from_pretrained(cfg.model_path, trust_remote_code=True)
        print("Tokenizer loaded", tokenizer.__class__.__name__, flush=True)
    except BaseException as exc:  # noqa: BLE001
        print("Tokenizer load failed", type(exc).__name__, exc)
        traceback.print_exc()

    print("Testing model load...", flush=True)
    try:
        from transformers import AutoModelForCausalLM as _AM  # noqa: E402
        start = time.time()
        model = _AM.from_pretrained(cfg.model_path, trust_remote_code=True, device_map=cfg.device_map, low_cpu_mem_usage=cfg.low_cpu_mem_usage)
        duration = time.time() - start
        print("Model loaded successfully in", round(duration, 2), "seconds", flush=True)
    except BaseException as exc:  # noqa: BLE001
        print("Model load failed", type(exc).__name__, exc, flush=True)
        traceback.print_exc()

    print("Creating QwenChatAPI instance...", flush=True)
    start = time.time()
    try:
        api = QwenChatAPI(cfg)
        print(f"OK: model loaded in {time.time() - start:.2f}s", flush=True)
    except BaseException as exc:  # noqa: BLE001
        print("ExceptionCaptured")
        print(type(exc).__name__, exc)
        traceback.print_exc()
except Exception as exc:
    print("ExceptionCaptured", flush=True)
    print(type(exc).__name__, exc, flush=True)
    traceback.print_exc()
