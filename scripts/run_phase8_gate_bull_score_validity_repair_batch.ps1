param(
    [string]$LogPath = "",
    [string]$RunsRoot = "artifacts\runs"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $LogPath = Join-Path $RunsRoot "phase8_gate_bull_score_validity_repair_batch_$stamp.log"
}

$logDirectory = Split-Path -Parent $LogPath
if (-not [string]::IsNullOrWhiteSpace($logDirectory)) {
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
}

function Write-BatchLog {
    param([string]$Message)

    $line = "[{0}] {1}" -f (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"), $Message
    $line | Tee-Object -FilePath $LogPath -Append
}

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Command,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    Write-BatchLog "START $Description"
    Write-BatchLog ("COMMAND " + ($Command -join " "))
    $startedAt = Get-Date
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $Command[0] @($Command[1..($Command.Length - 1)]) 2>&1 |
            Tee-Object -FilePath $LogPath -Append
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    $duration = New-TimeSpan -Start $startedAt -End (Get-Date)
    Write-BatchLog ("FINISH {0} exit={1} duration={2}" -f $Description, $exitCode, $duration)
    if ($exitCode -ne 0) {
        throw "$Description failed with exit code $exitCode"
    }
    return @($output | ForEach-Object { $_.ToString() })
}

$configs = @(
    "configs\experiment.btc_phase8_bull_floor_gate_bull_prob100_score_validity.yaml",
    "configs\experiment.btc_phase8_bull_floor_gate_bull_prob100_score_validity_uncalibrated.yaml",
    "configs\experiment.btc_phase8_bull_floor_gate_bull_prob100_score_validity_low_turnover.yaml"
)

$completedRunDirs = @()

Write-BatchLog "Phase 8 BTC gate-bull score-validity repair batch started."
foreach ($config in $configs) {
    Write-BatchLog "CONFIG $config"
    $runOutput = Invoke-LoggedCommand `
        -Description "run-experiment $config" `
        -Command @("python", "scripts\run_marketlab.py", "run-experiment", "--config", $config)

    $runDir = @(
        $runOutput |
            Where-Object { $_ -match "^(?:[A-Za-z]:\\.*\\)?artifacts[\\/]+runs[\\/]" } |
            Select-Object -Last 1
    )
    if ($runDir.Count -ne 1) {
        throw "Could not determine run directory for $config"
    }
    $runDir = $runDir[0]
    $completedRunDirs += $runDir
    Write-BatchLog "RUN_DIR $config $runDir"

    Invoke-LoggedCommand `
        -Description "phase8-summary $runDir" `
        -Command @("python", "scripts\run_marketlab.py", "phase8-summary", "--run-dir", $runDir) | Out-Null
    Invoke-LoggedCommand `
        -Description "phase8-target-diagnostic $runDir" `
        -Command @("python", "scripts\run_marketlab.py", "phase8-target-diagnostic", "--run-dir", $runDir, "--config", $config) | Out-Null
    Invoke-LoggedCommand `
        -Description "phase8-bull-participation $runDir" `
        -Command @("python", "scripts\run_marketlab.py", "phase8-bull-participation", "--run-dir", $runDir, "--config", $config) | Out-Null
    Invoke-LoggedCommand `
        -Description "phase8-score-diagnostic $runDir" `
        -Command @("python", "scripts\run_marketlab.py", "phase8-score-diagnostic", "--run-dir", $runDir) | Out-Null
    Invoke-LoggedCommand `
        -Description "phase8-bull-counterfactual $runDir" `
        -Command @("python", "scripts\run_marketlab.py", "phase8-bull-counterfactual", "--run-dir", $runDir, "--config", $config) | Out-Null
    Invoke-LoggedCommand `
        -Description "phase8-regime-policy-sweep $runDir" `
        -Command @("python", "scripts\run_marketlab.py", "phase8-regime-policy-sweep", "--run-dir", $runDir, "--config", $config) | Out-Null
    Invoke-LoggedCommand `
        -Description "phase8-methodology-review $runDir" `
        -Command @("python", "scripts\run_marketlab.py", "phase8-methodology-review", "--run-dir", $runDir) | Out-Null
}

Invoke-LoggedCommand `
    -Description "phase8-grid-compare" `
    -Command @(
        "python",
        "scripts\run_marketlab.py",
        "phase8-grid-compare",
        "--runs-root",
        $RunsRoot,
        "--output",
        (Join-Path $RunsRoot "phase8_btc_grid_comparison.csv")
    ) | Out-Null

Write-BatchLog "Phase 8 BTC gate-bull score-validity repair batch completed."
foreach ($runDir in $completedRunDirs) {
    Write-BatchLog "COMPLETED_RUN $runDir"
}
