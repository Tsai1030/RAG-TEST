# Copy run results from the Jetson Nano back to Windows (results/runs/<run_id>/).
#   .\scripts\nano_fetch.ps1 -RunId 20261107-210000_load_probe
# No password is stored here: scp asks for it, or use an SSH key.
param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [string]$Target = "robo@172.28.0.147",
    [string]$RemoteDir = "50019/rag_test"
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root "results\runs"
New-Item -ItemType Directory -Force $dest | Out-Null

& scp -q -r "${Target}:$RemoteDir/results/runs/$RunId" $dest
if ($LASTEXITCODE -ne 0) { throw "scp failed with exit code $LASTEXITCODE" }
Write-Host "Fetched $RunId to $dest"
