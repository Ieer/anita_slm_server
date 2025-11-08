# 一鍵啟動主服務（使用 python -m uvicorn，避免 uvicorn.exe 問題）
param(
    [string]$BindHost = "127.0.0.1",
    [int]$PortNumber = 8000
)

Write-Host "Anita SLM Server - Main Starter" -ForegroundColor Blue

if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
    Write-Host "已啟用虛擬環境 .venv" -ForegroundColor Green
} else {
    Write-Host "未找到 .venv，將使用系統 Python" -ForegroundColor Yellow
}

Write-Host "啟動服務 http://$BindHost:$PortNumber" -ForegroundColor Yellow
python -m uvicorn src.anita_slm_server.slm_server:app --host $BindHost --port $PortNumber
