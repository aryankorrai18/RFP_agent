# One-click launcher for the agents and the hub.
#   .\run.ps1 start            start RFP assistant (8001), Deal Intelligence (8002) and the Hub (8003)
#   .\run.ps1 status           show which are running
#   .\run.ps1 stop             stop only the processes this script started
# Run with:  powershell -ExecutionPolicy Bypass -File .\run.ps1 start
param(
    [Parameter(Position = 0)][ValidateSet("start", "stop", "status")][string]$Action = "status"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$runDir = Join-Path $root ".run"
$pidFile = Join-Path $runDir "pids.json"

$apps = @(
    @{ Name = "rfp";   Folder = "rfp-v0";            Module = "rfp_assistant.main:app";   Port = 8001; Label = "RFP Memory Assistant" },
    @{ Name = "deals"; Folder = "deal-intelligence"; Module = "deal_intelligence.main:app"; Port = 8002; Label = "Deal Intelligence" },
    @{ Name = "hub";   Folder = "agent-hub";         Module = "agent_hub.main:app";        Port = 8003; Label = "Agent Hub" }
)

function Get-Listener([int]$Port) {
    $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($c) { return $c[0].OwningProcess }
    return $null
}

function Test-Health([int]$Port) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 2
        return $r.StatusCode -eq 200
    } catch { return $false }
}

function Read-Pids {
    if (Test-Path $pidFile) { return @(Get-Content $pidFile -Raw | ConvertFrom-Json) }
    return @()
}

function Show-Status {
    foreach ($app in $apps) {
        $up = Test-Health $app.Port
        $state = if ($up) { "running" } else { "stopped" }
        "{0,-22} http://127.0.0.1:{1}   {2}" -f $app.Label, $app.Port, $state
    }
}

function Start-Apps {
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $started = @()
    foreach ($app in $apps) {
        $dir = Join-Path $root $app.Folder
        $python = Join-Path $dir ".venv\Scripts\python.exe"
        if (-not (Test-Path $python)) { Write-Warning "$($app.Label): no virtual environment at $python (run its run.ps1 once first)."; continue }
        if (Get-Listener $app.Port) { Write-Host "$($app.Label): port $($app.Port) is already in use, leaving it alone."; continue }
        $uvArgs = @("-m", "uvicorn", $app.Module, "--app-dir", "src", "--host", "127.0.0.1", "--port", "$($app.Port)")
        $p = Start-Process -FilePath $python -ArgumentList $uvArgs -WorkingDirectory $dir -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $runDir "$($app.Name).out.log") -RedirectStandardError (Join-Path $runDir "$($app.Name).err.log")
        $started += @{ name = $app.Name; label = $app.Label; port = $app.Port; launcher = $p.Id; listener = $null }
        Write-Host "$($app.Label): starting..."
    }
    $deadline = (Get-Date).AddSeconds(90)
    foreach ($entry in $started) {
        while ((Get-Date) -lt $deadline -and -not (Test-Health $entry.port)) { Start-Sleep -Milliseconds 700 }
        $entry.listener = Get-Listener $entry.port
        $state = if (Test-Health $entry.port) { "ready" } else { "NOT READY (see .run\$($entry.name).err.log)" }
        Write-Host ("{0,-22} http://127.0.0.1:{1}   {2}" -f $entry.label, $entry.port, $state)
    }
    $existing = Read-Pids | Where-Object { $started.name -notcontains $_.name }
    ConvertTo-Json -InputObject @($existing + $started) | Set-Content -Path $pidFile
    Write-Host "Open the hub at http://127.0.0.1:8003"
}

function Stop-Apps {
    $entries = Read-Pids
    if (-not $entries) { Write-Host "Nothing recorded to stop."; return }
    foreach ($e in $entries) {
        $listener = Get-Listener $e.port
        if ($listener -and $e.listener -and $listener -eq $e.listener) {
            Stop-Process -Id $listener -Force
            Write-Host "$($e.label): stopped (PID $listener)."
        } elseif ($listener) {
            Write-Warning "$($e.label): port $($e.port) is now held by a different process (PID $listener); not touching it."
        } else {
            Write-Host "$($e.label): already stopped."
        }
        if ($e.launcher) {
            # One lookup: the launcher can exit between a check and a second call, and that must not abort the stop.
            $p = Get-Process -Id $e.launcher -ErrorAction SilentlyContinue
            if ($p -and $p.ProcessName -like "python*") { Stop-Process -Id $e.launcher -Force -ErrorAction SilentlyContinue }
        }
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}

switch ($Action) {
    "start"  { Start-Apps }
    "stop"   { Stop-Apps }
    "status" { Show-Status }
}
