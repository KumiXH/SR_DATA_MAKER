param(
    [Parameter(Mandatory = $true)]
    [string]$ConfigPath,
    [Parameter(Mandatory = $true)]
    [string]$TaskName,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [int]$MaxRestarts = 20
)

$ErrorActionPreference = "Stop"

$repoRoot = "D:\repository\SR_DATA_MAKER"
$logDir = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir ($TaskName + ".log")

$env:PYTHONPATH = "src"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

function Get-SourceCount {
    param([string]$yamlPath)
    $script = @"
from sr_data_maker.config.loader import load_config
from sr_data_maker.sources.image_folder import ImageFolderSourceReader
config = load_config(r'''$yamlPath''')
source = dict(config["source"])
source.pop("type", None)
reader = ImageFolderSourceReader(**source, root=config["paths"]["input_root"])
print(sum(1 for _ in reader.iter_sources()))
"@
    $result = $script | python -
    return [int]$result.Trim()
}

function Get-CompletedCount {
    param([string]$statePath, [string]$taskName)
    if (-not (Test-Path $statePath)) {
        return 0
    }
    return (Get-Content $statePath | Where-Object { $_ -match ('"key": "' + [regex]::Escape($taskName) + '::') }).Count
}

$statePath = Join-Path $OutputRoot "manifests\state.jsonl"
$total = Get-SourceCount -yamlPath (Join-Path $repoRoot $ConfigPath)
"[$(Get-Date -Format s)] START task=$TaskName total=$total" | Tee-Object -FilePath $logPath -Append

for ($attempt = 1; $attempt -le $MaxRestarts; $attempt++) {
    $completedBefore = Get-CompletedCount -statePath $statePath -taskName $TaskName
    "[$(Get-Date -Format s)] ATTEMPT=$attempt completed_before=$completedBefore" | Tee-Object -FilePath $logPath -Append

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    cmd /c "python -m sr_data_maker.cli.main run --config ""$ConfigPath"" 2>&1" | Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference

    $completedAfter = Get-CompletedCount -statePath $statePath -taskName $TaskName
    "[$(Get-Date -Format s)] ATTEMPT=$attempt exit_code=$exitCode completed_after=$completedAfter" | Tee-Object -FilePath $logPath -Append

    if ($completedAfter -ge $total) {
        "[$(Get-Date -Format s)] END task=$TaskName status=success completed=$completedAfter total=$total" | Tee-Object -FilePath $logPath -Append
        exit 0
    }

    if ($completedAfter -le $completedBefore) {
        "[$(Get-Date -Format s)] END task=$TaskName status=stalled completed=$completedAfter total=$total" | Tee-Object -FilePath $logPath -Append
        throw "No progress detected for $TaskName. See $logPath"
    }

    Start-Sleep -Seconds 3
}

throw "Exceeded MaxRestarts=$MaxRestarts for $TaskName. See $logPath"
