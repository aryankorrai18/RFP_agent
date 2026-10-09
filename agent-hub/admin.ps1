# Make companies and accounts for the hub. Run from this folder, for example:
#   .\admin.ps1 workspaces
#   .\admin.ps1 company add "Accenture" --deals accenture-deals --rfp accenture-rfp
#   .\admin.ps1 user add dana@accenture.com --company Accenture --name Dana
#   .\admin.ps1 user list
# See README.md ("Sign-in and companies").
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $here "src"
& (Join-Path $here ".venv\Scripts\python.exe") -m agent_hub.admin @args
exit $LASTEXITCODE
