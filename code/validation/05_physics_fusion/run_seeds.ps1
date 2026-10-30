param(
    [string]$Seeds = "42,43,44",
    [int]$Epochs = 50,
    [int]$MaxPerLabel = 0
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\shared\config.ps1")

$Manifest = Join-Path $Data26 "grouped_split.csv"
$Features = Join-Path $Data26 "physical_features.csv"
$OutputRoot = Join-Path $PSScriptRoot "runs"
$env:PYTHONPATH = Join-Path $Repo26 "src"

if (-not (Test-Path -LiteralPath $Features)) {
    & $Python (Join-Path $Repo26 "scripts\build_physical_features.py") --manifest $Manifest --output $Features
    if ($LASTEXITCODE -ne 0) {
        throw "Physical feature generation failed."
    }
}

$Rows = @()
foreach ($Seed in $Seeds.Split(",")) {
    $Seed = $Seed.Trim()
    if ([string]::IsNullOrWhiteSpace($Seed)) {
        continue
    }
    $RunDir = Join-Path $OutputRoot "seed_$Seed"
    $Arguments = @(
        "--manifest", $Manifest,
        "--physical-features", $Features,
        "--output-dir", $RunDir,
        "--epochs", "$Epochs",
        "--batch-size", "64",
        "--num-workers", "4",
        "--seed", $Seed
    )
    if ($MaxPerLabel -gt 0) {
        $Arguments += @("--max-per-label", "$MaxPerLabel")
    }
    & $Python (Join-Path $Repo26 "scripts\train_hybrid.py") @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Hybrid training failed for seed $Seed."
    }
    $Metrics = Get-Content -LiteralPath (Join-Path $RunDir "metrics.json") -Raw | ConvertFrom-Json
    $Rows += [PSCustomObject]@{
        seed = [int]$Seed
        best_val_accuracy = [double]$Metrics.best_val_accuracy
        test_accuracy = [double]$Metrics.test_accuracy
        test_macro_f1 = [double]$Metrics.test_macro_f1
        epochs_run = @($Metrics.history).Count
        parameters = [int]$Metrics.parameters
        output_dir = $RunDir
    }
}

$Summary = Join-Path $OutputRoot "hybrid_seed_summary.csv"
$Rows | Export-Csv -LiteralPath $Summary -NoTypeInformation -Encoding UTF8
Write-Host "Wrote $Summary"
