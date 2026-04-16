[CmdletBinding()]
param(
    [switch]$SkipPull,
    [string]$PytestBaseTemp = ".pytest_tmp_real_telegram_launch"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Get-GitPorcelainStatus {
    $status = & git status --porcelain=v1
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to inspect git worktree state."
    }

    return @($status | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

Push-Location $RepoRoot
try {
    if (-not $SkipPull) {
        $currentBranch = (& git branch --show-current).Trim()
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to resolve current git branch."
        }

        $dirtyEntries = Get-GitPorcelainStatus
        if ($dirtyEntries.Count -gt 0) {
            Write-Warning "Skipping git pull because the worktree has uncommitted changes. Re-run with a clean worktree or pass -SkipPull explicitly."
        }
        elseif ($currentBranch -ne "master") {
            Write-Warning "Skipping git pull because the current branch is '$currentBranch', not 'master'. Re-run from master or pass -SkipPull explicitly."
        }
        else {
            Write-Host "Pulling latest master..."
            git pull --ff-only origin master
        }
    }

    Write-Host "Running real Telegram smoke test..."
    $env:MARKETLAB_RUN_REAL_TELEGRAM = "1"
    .tox\py312\Scripts\python.exe -m pytest -q tests/integration/test_real_telegram_smoke.py --basetemp $PytestBaseTemp

    Write-Host "Restarting paper Docker services..."
    docker compose --env-file .env -f docker/compose.paper.yml up -d --build

    Write-Host "Paper stack status:"
    docker compose --env-file .env -f docker/compose.paper.yml ps

    $statusPath = Join-Path $RepoRoot "artifacts/paper/state/status.json"
    if (-not (Test-Path -LiteralPath $statusPath)) {
        throw "Paper status file not found: $statusPath"
    }

    Write-Host "Paper status summary:"
    Get-Content -LiteralPath $statusPath

    Write-Host "Scheduler log tail:"
    docker logs --tail 50 marketlab-paper-scheduler

    Write-Host "Agent log tail:"
    docker logs --tail 50 marketlab-paper-agent
}
finally {
    Remove-Item Env:MARKETLAB_RUN_REAL_TELEGRAM -ErrorAction SilentlyContinue
    Pop-Location
}
