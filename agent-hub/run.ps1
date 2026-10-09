# Start the Agent Hub.
# Works from any folder:  powershell -ExecutionPolicy Bypass -File <path>\run.ps1
# Optional port:          ... -File run.ps1 -Port 8004
param([int]$Port = 8003)

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

Write-Host "Open http://127.0.0.1:$Port   (Ctrl+C to stop)"
& $python -m uvicorn agent_hub.main:app --app-dir src --host 127.0.0.1 --port $Port
