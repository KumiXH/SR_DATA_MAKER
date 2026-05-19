$repoRoot = "D:\repository\SR_DATA_MAKER"
$outputRoot = Join-Path $repoRoot "data\outputs\plant_mixed_teacher_x2"
$logDir = Join-Path $outputRoot "logs"

Write-Host "Logs:"
if (Test-Path $logDir) {
    Get-ChildItem $logDir | Select-Object Name, Length, LastWriteTime
} else {
    Write-Host "No logs yet."
}

Write-Host ""
Write-Host "Run summaries:"
$summaries = @(
    Join-Path $outputRoot "manifests\run_summary.json"
)
foreach ($summary in $summaries) {
    if (Test-Path $summary) {
        Write-Host $summary
        Get-Content $summary
    }
}
