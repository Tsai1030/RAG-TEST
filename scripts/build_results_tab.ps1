# Embeds reports/overview.json and reports/records/*.json into the results tab of docs/project_plan.html.
# Temporary until `memrag report build` exists.
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$utf8 = New-Object System.Text.UTF8Encoding($false)
$page = Join-Path $root "docs\project_plan.html"

$overview = [IO.File]::ReadAllText((Join-Path $root "reports\overview.json"), $utf8)
$null = $overview | ConvertFrom-Json

$items = @()
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $root "reports\records") -Filter "*.json" | Sort-Object Name) {
    $text = [IO.File]::ReadAllText($file.FullName, $utf8)
    $null = $text | ConvertFrom-Json
    $items += '{"id":"' + $file.BaseName + '","record":' + $text + '}'
}
$json = ('{"overview":' + $overview + ',"records":[' + ($items -join ',') + ']}').Replace('</', '<\/')

$html = [IO.File]::ReadAllText($page, $utf8)
$open = '<script id="report-data" type="application/json">'
$start = $html.IndexOf($open)
if ($start -lt 0) { throw "report-data block not found in $page" }
$start += $open.Length
$end = $html.IndexOf('</script>', $start)
$html = $html.Substring(0, $start) + $json + $html.Substring($end)
[IO.File]::WriteAllText($page, $html, $utf8)

Write-Host "Embedded $($items.Count) records into $page"
