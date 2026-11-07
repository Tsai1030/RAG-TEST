# Copy everything in the project folder from Windows to the Jetson Nano.
# Only platform-specific and rebuildable folders are skipped. Files removed on
# Windows are not removed on the Nano, so results made on the Nano stay safe.
# No password is stored here: ssh and scp ask for it, or use an SSH key.
param(
    [string]$Target = "robo@172.28.0.147",
    [string]$RemoteDir = "50019/rag_test"
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$archive = Join-Path $env:TEMP "rag_test_sync.tar"
$remoteArchive = "50019/.rag_test_sync.tar"
$exclude = @(".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".import_linter_cache", ".hf_cache")

try {
    $tarArgs = @("-cf", $archive) + ($exclude | ForEach-Object { "--exclude=$_" }) + @("-C", $root, ".")
    & tar.exe @tarArgs
    if ($LASTEXITCODE -ne 0) { throw "tar failed with exit code $LASTEXITCODE" }

    & ssh $Target "mkdir -p $RemoteDir"
    if ($LASTEXITCODE -ne 0) { throw "ssh mkdir failed with exit code $LASTEXITCODE" }

    & scp -q $archive "${Target}:$remoteArchive"
    if ($LASTEXITCODE -ne 0) { throw "scp failed with exit code $LASTEXITCODE" }

    & ssh $Target "tar -xf $remoteArchive -C $RemoteDir --no-same-owner --warning=no-unknown-keyword; status=`$?; rm -f $remoteArchive; exit `$status"
    if ($LASTEXITCODE -ne 0) { throw "remote extract failed with exit code $LASTEXITCODE" }

    Write-Host "Synced $root to ${Target}:~/$RemoteDir"
}
finally {
    Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
}
