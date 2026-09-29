# Start the RFP Memory Assistant V0 server.
# Works from any folder:  powershell -ExecutionPolicy Bypass -File <path>\rfp-v0\run.ps1
# Optional port:          ... -File run.ps1 -Port 8002
param([int]$Port = 8001)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Creating the virtual environment and installing dependencies (first run only)..."
    python -m venv .venv
    & $python -m pip install -r requirements.txt
}

$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Warning "Port $Port is already in use (PID $($busy[0].OwningProcess)). Run again with -Port <another number>."
    exit 1
}

if (-not (Test-Path ".env")) {
    Write-Warning "No .env file. Copy .env.example to .env and set GEMINI_API_KEY or ANTHROPIC_API_KEY, or drafting will fail."
}

Write-Host "Open http://127.0.0.1:$Port   (Ctrl+C to stop)"
# No --reload: it can hang on Windows. The app re-reads .env on every request, so a key
# saved there works immediately; restart only after changing code.
& $python -m uvicorn backend.main:app --port $Port
