# Run a command in the project folder on the Jetson Nano, e.g.
#   .\scripts\nano_run.ps1 "uv run memrag env check"
#   .\scripts\nano_run.ps1 "uv run pytest -q"
# `uv` is not on PATH for non-interactive SSH, so ~/.local/bin is added first.
# No password is stored here: ssh asks for it, or use an SSH key.
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Command,
    [string]$Target = "robo@172.28.0.147",
    [string]$RemoteDir = "50019/rag_test"
)
$ErrorActionPreference = "Stop"

& ssh $Target "export PATH=`$HOME/.local/bin:`$PATH; cd $RemoteDir && $Command"
exit $LASTEXITCODE
