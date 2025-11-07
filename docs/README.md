# 快速開始 (節選自根目錄 README)

> 此頁為根目錄 `README.md` 摘要，確保在 MkDocs 站點內可正常瀏覽。完整內容請查看原始檔案或專案首頁。

## 安裝

```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -e .
```

## 開發環境

```powershell
pip install -e .[dev]
```

## 啟動服務

```powershell
python -m uvicorn anita_slm_server.slm_server:app --host 127.0.0.1 --port 8001
```

## 啟用 API Key（建議於對外環境）

預設不啟用驗證。若需保護 API，請設定：

Windows PowerShell：

```powershell
$env:REQUIRE_API_KEY = "1"
$env:API_KEY = "my-secret-key"
uvicorn anita_slm_server.slm_server:app --host 127.0.0.1 --port 8001
```

Linux / macOS / WSL：

```bash
export REQUIRE_API_KEY=1
export API_KEY="my-secret-key"
uvicorn anita_slm_server.slm_server:app --host 127.0.0.1 --port 8001
```

呼叫時在 Header 加上金鑰（擇一）：`X-API-Key: my-secret-key` 或 `Authorization: Bearer my-secret-key`。

例如（PowerShell）：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8001/health -Headers @{ 'X-API-Key' = 'my-secret-key' }
```

## llama.cpp 後端

```powershell
pip install -e .[llama]
$env:MODEL_BACKEND="llama.cpp"
$env:MODEL_PATH="models/qwen/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
uvicorn anita_slm_server.slm_server:app --host 127.0.0.1 --port 8001
```

## 測試

```powershell
pytest -q
```

---

此頁僅作為文件站引用。
