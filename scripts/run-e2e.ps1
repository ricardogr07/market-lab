[CmdletBinding()]
param(
    [string]$ConfigPath = "configs/experiment.weekly_rank.smoke.yaml",
    [switch]$SkipPytest
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath {
    param([string]$PathValue)

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return (Resolve-Path -LiteralPath $PathValue).Path
    }

    return (Resolve-Path -LiteralPath (Join-Path $script:RepoRoot $PathValue)).Path
}

function Assert-PathExists {
    param(
        [string]$PathValue,
        [string]$Description,
        [int]$ExitCode
    )

    if (-not (Test-Path -LiteralPath $PathValue)) {
        Write-Error "$Description not found: $PathValue"
        exit $ExitCode
    }
}

function Assert-OptionalPathExists {
    param(
        [string]$PathValue,
        [string]$Description
    )

    if (Test-Path -LiteralPath $PathValue) {
        Write-Host "$Description found: $PathValue"
    }
}

function Get-CommandOutputPath {
    param(
        [string[]]$OutputLines,
        [string]$Description,
        [int]$ExitCode
    )

    $resolvedLine = $OutputLines |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        Select-Object -Last 1

    if ([string]::IsNullOrWhiteSpace($resolvedLine)) {
        Write-Error "$Description did not return a path."
        exit $ExitCode
    }

    return $resolvedLine.Trim()
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ConfigFullPath = Resolve-RepoPath -PathValue $ConfigPath
$SourcePath = Join-Path $RepoRoot "src"
$LauncherPath = Join-Path $RepoRoot "scripts\\run_marketlab.py"
$env:MARKETLAB_CONFIG_PATH = $ConfigFullPath
$env:MARKETLAB_SOURCE_PATH = $SourcePath

try {
    $configInfo = & python -c "import os, sys; sys.path.insert(0, os.environ['MARKETLAB_SOURCE_PATH']); from marketlab.config import load_config; config = load_config(os.environ['MARKETLAB_CONFIG_PATH']); print(config.prepared_panel_path); print(config.output_dir / config.experiment_name)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to resolve config-derived paths for $ConfigFullPath."
        exit 12
    }

    $preparedPanelPath = Get-CommandOutputPath -OutputLines @($configInfo[0]) -Description "Config resolution" -ExitCode 13
    $runRoot = Get-CommandOutputPath -OutputLines @($configInfo[1]) -Description "Run root resolution" -ExitCode 14

    if (-not $SkipPytest) {
        Write-Host "Running fixture-backed pytest gate..."
        & python -m pytest -q --basetemp .pytest_tmp
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Pytest failed before the real-data E2E flow."
            exit 20
        }
    }

    Write-Host "Preparing real market data with $ConfigFullPath..."
    $prepareOutput = & python $LauncherPath prepare-data --config $ConfigFullPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "prepare-data failed."
        exit 30
    }

    $resolvedPanelPath = Get-CommandOutputPath -OutputLines @($prepareOutput) -Description "prepare-data" -ExitCode 31
    Assert-PathExists -PathValue $resolvedPanelPath -Description "Prepared panel" -ExitCode 32
    Assert-PathExists -PathValue $preparedPanelPath -Description "Configured prepared panel" -ExitCode 33

    Write-Host "Running baseline backtest..."
    $backtestOutput = & python $LauncherPath backtest --config $ConfigFullPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "backtest failed."
        exit 40
    }

    $backtestRunDir = Get-CommandOutputPath -OutputLines @($backtestOutput) -Description "backtest" -ExitCode 41
    Assert-PathExists -PathValue $backtestRunDir -Description "Backtest run directory" -ExitCode 42

    foreach ($artifact in @("metrics.csv", "performance.csv", "strategy_summary.csv", "monthly_returns.csv", "turnover_costs.csv", "report.md", "cumulative_returns.png", "drawdown.png", "turnover.png")) {
        Assert-PathExists -PathValue (Join-Path $backtestRunDir $artifact) -Description "Backtest artifact $artifact" -ExitCode 43
    }

    Start-Sleep -Seconds 1

    Write-Host "Running Phase 2 train-models flow..."
    $trainModelsOutput = & python $LauncherPath train-models --config $ConfigFullPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "train-models failed."
        exit 45
    }

    $trainModelsRunDir = Get-CommandOutputPath -OutputLines @($trainModelsOutput) -Description "train-models" -ExitCode 46
    Assert-PathExists -PathValue $trainModelsRunDir -Description "Train-models run directory" -ExitCode 47

    foreach ($artifact in @("folds.csv", "model_manifest.csv", "model_metrics.csv", "predictions.csv", "model_summary.csv", "fold_summary.csv")) {
        Assert-PathExists -PathValue (Join-Path $trainModelsRunDir $artifact) -Description "Train-models artifact $artifact" -ExitCode 48
    }
    Assert-PathExists -PathValue (Join-Path $trainModelsRunDir "models") -Description "Train-models models directory" -ExitCode 49

    Start-Sleep -Seconds 1

    Write-Host "Running full experiment flow..."
    $experimentOutput = & python $LauncherPath run-experiment --config $ConfigFullPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "run-experiment failed."
        exit 60
    }

    $experimentRunDir = Get-CommandOutputPath -OutputLines @($experimentOutput) -Description "run-experiment" -ExitCode 61
    Assert-PathExists -PathValue $experimentRunDir -Description "Run-experiment directory" -ExitCode 62

    foreach ($artifact in @("metrics.csv", "performance.csv", "strategy_summary.csv", "monthly_returns.csv", "turnover_costs.csv", "report.md", "cumulative_returns.png", "drawdown.png", "turnover.png", "model_summary.csv", "fold_summary.csv")) {
        Assert-PathExists -PathValue (Join-Path $experimentRunDir $artifact) -Description "Run-experiment artifact $artifact" -ExitCode 63
    }
    Assert-OptionalPathExists -PathValue (Join-Path $experimentRunDir "models") -Description "Run-experiment models directory"

    Assert-PathExists -PathValue $runRoot -Description "Configured run root" -ExitCode 64

    Write-Host "Real-data E2E completed successfully."
    Write-Host "Backtest run: $backtestRunDir"
    Write-Host "Train-models run: $trainModelsRunDir"
    Write-Host "Run-experiment run: $experimentRunDir"
    exit 0
}
finally {
    $env:MARKETLAB_CONFIG_PATH = $null
    $env:MARKETLAB_SOURCE_PATH = $null
}
