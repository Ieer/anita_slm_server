# Agent SLM Server

本地可運行、面向低資源機器的輕量化 OpenAI 相容 API 服務，預設整合 Qwen 0.5B 與多個 Embedding 模型。

## 🌟 特性

- Chat / Completion：`/v1/chat/completions`（亦可再包裝 `/v1/completions`）
- Embeddings：`/v1/embeddings` 自動掃描 `models/embedding/*`
- 模型列出與重新掃描：`/v1/models`，`POST /v1/reload-embeddings`
- 健康檢查與指標：`/health`, `/health/memory`, `/health/detailed`, `/metrics`
- 工具呼叫解析（`<tool_call>{...}</tool_call>` 標記 + OpenAI 風格輸出）
- 併發/排隊限制與逾時保護（內建 `_RequestLimiter`）
- 多後端：`transformers` / `llama.cpp` (GGUF) 可切換（`MODEL_BACKEND`）
- 自動釋放記憶體（主模型閒置卸載 + Embedding cache TTL）
- Token/字元雙層輸入截斷 + OOM 自動降級重試
- Embedding 支援 `float` 與 `base64` 兩種輸出格式

進階文檔：

| 主題 | 位置 |
|------|------|
| 使用 / 參數 / 範例 | `USAGE_GUIDE.md` |
| 記憶體優化 / 量化策略 | `docs/memory-optimization-guide.md` |
| 性能品質檢查摘要 | `docs/optimization_summary.md`, `docs/OPTIMIZATION_REPORT.md` |

---

## 🚀 快速開始

```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -e .[optimized]   # 或最小: pip install -e .
uvicorn anita_slm_server.slm_server:app --host 127.0.0.1 --port 8001
```

若需 GGUF / llama.cpp 後端：

```powershell
pip install -e .[llama]
$env:MODEL_BACKEND="llama.cpp"
$env:MODEL_PATH="models/qwen/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
uvicorn anita_slm_server.slm_server:app --host 127.0.0.1 --port 8001
```

快速腳本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_llama_cpp.ps1
```

或：

```bash
bash scripts/start_llama_cpp.sh
```

回退 transformers：移除 / 設置 `MODEL_BACKEND=transformers`。

---

## 📁 目錄概覽

```text
src/anita_slm_server/
  slm_server.py            # 主 FastAPI 服務（多端點 + 後端控制）
  slm_server_simple.py     # 精簡版
  qwenchat.py              # QwenChatAPI + QwenChatConfig 封裝
  chat_backends.py         # 後端載入抽象 (transformers / llama.cpp)
  memory_monitor.py        # 記憶體與 GPU 監控
  performance_config.py    # 性能/併發配置
scripts/
  start_server.py          # 啟動檢查 + 快速啟動
  start_optimized.(ps1|sh) # 低記憶體/量化自動化
  start_llama_cpp.(ps1|sh) # llama.cpp 後端啟動
models/
  qwen/Qwen2.5-0.5B-Instruct/
  qwen/Qwen2.5-0.5B-Instruct-GGUF/
  embedding/<多個 embedding 子模型>
tests/                     # pytest 測試
docs/                      # 深入文檔與報告
```

---

## 🔐 可選：啟用 API Key

```powershell
$env:REQUIRE_API_KEY="1"; $env:API_KEY="my-secret";
uvicorn anita_slm_server.slm_server:app --host 127.0.0.1 --port 8001
```

請求時加入：`X-API-Key: my-secret` 或 `Authorization: Bearer my-secret`。

---

## 🧪 Chat API 範例 (PowerShell)

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8001/v1/chat/completions -Method Post -Body (@{
  model = 'qwen2.5-0.5b-instruct'
  messages = @(@{role='user'; content='說一句早安'})
} | ConvertTo-Json -Depth 5) -ContentType 'application/json'
```

更多：工具呼叫 / embeddings / metrics 請見 `USAGE_GUIDE.md`。

---

## 📊 Metrics

Prometheus 抓取 `GET /metrics`，額外聚合：

| 指標 | 說明 |
|------|------|
| `agent_slm_requests_total` | 總請求數 |
| `agent_slm_chat_latency_avg_seconds` | 平均聊天延遲（計算後行追加）|
| `agent_slm_chat_tokens_per_sec_avg` | 平均 token/s |
| `agent_slm_queue_wait_time_avg_seconds` | 排隊平均等待 |
| `agent_slm_gpu_memory_*_bytes` | GPU 記憶體統計 |

健康端點：`/health`, `/health/memory`, `/health/detailed`。

---

## 🧬 核心環境變數（節選）

| 變數 | 預設 | 功能 |
|------|------|------|
| `MODEL_PATH` | models/qwen/Qwen2.5-0.5B-Instruct | 主模型路徑 (或 `.gguf` 檔) |
| `MODEL_BACKEND` | transformers | `transformers` / `llama.cpp` |
| `MODEL_QUANTIZATION` | (空) | 4bit / 8bit (需 bitsandbytes) |
| `MAX_INPUT_TOKENS` | 2048 | 輸入 token 截斷上限 |
| `MAX_COMPLETION_TOKENS` | 1024 | 生成 token 上限 |
| `MAX_CONCURRENT_REQUESTS` | performance_config | 併發限制 |
| `MAX_PENDING_REQUESTS` | 2*concurrent | 佇列上限（0=不排隊）|
| `MODEL_UNLOAD_ENABLED` | 1 | 閒置自動卸載主模型 |
| `MODEL_IDLE_TIMEOUT` | 3600 | 閒置秒數卸載 |
| `EMBEDDING_DEVICE` | auto | cuda / cpu / auto |
| `REQUIRE_API_KEY` | 0 | 啟用 API Key 驗證 |
| `MAX_REQUEST_BYTES` | 1048576 | 請求 Content-Length 硬限制 |

完整解釋：`USAGE_GUIDE.md`、`docs/memory-optimization-guide.md`。

---

## 🧩 QwenChatConfig 使用

```python
from anita_slm_server.qwenchat import QwenChatAPI, QwenChatConfig

cfg = QwenChatConfig(
    model_path="./models/qwen/Qwen2.5-0.5B-Instruct",
    dtype="bfloat16",
    quantization=None,
    max_input_tokens=2048,
    default_system_prompt="You are a helpful assistant.",
    low_cpu_mem_usage=True,
    global_seed=42,
)
api = QwenChatAPI(cfg)
print(api.chat("你好，介紹一下自己"))
```

舊樣式仍相容；建議統一 config 方便擴充（未來：分片 / KV cache 策略）。

---

## 🗂️ 模型準備策略

不追蹤大型權重檔；`models/` 只保留最小 metadata（config / tokenizer / vocab / merges / README 等）。

範例 HuggingFace 下載：

```python
from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="Qwen/Qwen2.5-0.5B-Instruct",
  local_dir="models/qwen/Qwen2.5-0.5B-Instruct",
  allow_patterns=["*.safetensors","config.json","tokenizer*","vocab*","merges.txt","special_tokens_map.json","generation_config.json"],
  ignore_patterns=["*.bin","*.onnx"]
)
```

若需測試 GGUF：

```bash
git lfs install
git lfs track "*.gguf"
git add .gitattributes
```

誤加大檔清理：

```bash
git rm --cached -r models/qwen/Qwen2.5-0.5B-Instruct
git commit -m "Remove large weights"
```

---

## 🛠️ 測試 / 開發

```powershell
pytest -q
python scripts/start_server.py
python qwenchat.py
```

---

## 🗺 Roadmap

- [ ] SSE 串流輸出（OpenAI 事件格式）
- [ ] 對話滑動視窗更精細 token 截斷
- [ ] 環境變數集中 config 化
- [ ] Embeddings：normalize / batch size 控制
- [ ] 工具呼叫串流模式支持
- [ ] Console entry point (`agent-slm-server`)
- [ ] MkDocs + 自動 API 文件

---

## 📝 授權

僅供本地研究與測試；請遵守各上游模型（Qwen / sentence-transformers 等）授權條款。

歡迎提交 Issue / PR 改進 🎉