# Back up (or restore) everything the three apps keep on disk.
#   .\backup.ps1 create              a dated copy under backups\, checked straight away
#   .\backup.ps1 list
#   .\backup.ps1 verify backups\<folder>
#   .\backup.ps1 restore backups\<folder>      stop the apps first:  .\run.ps1 stop
# .env files (your keys) are never included. See agent-hub\README.md ("Backups").
$hub = Join-Path $PSScriptRoot "agent-hub"
$env:PYTHONPATH = Join-Path $hub "src"
& (Join-Path $hub ".venv\Scripts\python.exe") -m agent_hub.backup @args
exit $LASTEXITCODE
