$ErrorActionPreference = "Stop"

$repoRoot = "D:\repository\SR_DATA_MAKER"
$outputRoot = Join-Path $repoRoot "data\outputs\plant_mixed_teacher_x2"
$logDir = Join-Path $outputRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:PYTHONPATH = "src"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$jobs = @(
    @{
        Name = "realesrgan"
        Config = "configs/examples/plant_mixed_realesrgan_x2.yaml"
    },
    @{
        Name = "swinir"
        Config = "configs/examples/plant_mixed_swinir_x2.yaml"
    },
    @{
        Name = "hat"
        Config = "configs/examples/plant_mixed_hat_x2.yaml"
    }
)

foreach ($job in $jobs) {
    $logPath = Join-Path $logDir ($job.Name + ".log")
    "[$(Get-Date -Format s)] START $($job.Name)" | Tee-Object -FilePath $logPath -Append
    try {
        & python -m sr_data_maker.cli.main run --config $job.Config 2>&1 | Tee-Object -FilePath $logPath -Append
        if ($LASTEXITCODE -ne 0) {
            throw "Exit code $LASTEXITCODE"
        }
        "[$(Get-Date -Format s)] END $($job.Name) status=success" | Tee-Object -FilePath $logPath -Append
    } catch {
        "[$(Get-Date -Format s)] END $($job.Name) status=failed error=$($_.Exception.Message)" | Tee-Object -FilePath $logPath -Append
        throw
    }
}
