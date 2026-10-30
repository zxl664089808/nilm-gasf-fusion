param(
    [string]$Python = "D:\PLAID-projiect  2025.9.9\GLM-1\.venv\Scripts\python.exe",
    [string]$CyclesRoot = "D:\PLAID-projiect  2025.9.9\GLM-1\extracted_cycles_output",
    [string]$OutputRoot = "",
    [ValidateSet("gasf", "gadf")]
    [string]$Gramian = "gasf",
    [int]$MaxCyclesPerLabel = 5000,
    [int]$Workers = 8,
    [int]$Epochs = 50,
    [int]$BatchSize = 64,
    [int]$NumWorkers = 4,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python interpreter not found: $Python"
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $RepoRoot

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $WorkspaceRoot "clean_data\plaid_26state_gasf_64"
}

if ($Smoke) {
    $OutputRoot = Join-Path $WorkspaceRoot "clean_data\plaid_26state_smoke"
    $MaxCyclesPerLabel = 100
    $Workers = 2
    $Epochs = 2
    $NumWorkers = 0
}

$env:PYTHONPATH = Join-Path $RepoRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"

Write-Host "Repository: $RepoRoot"
Write-Host "Python: $Python"
Write-Host "Output: $OutputRoot"
Write-Host "Gramian: $Gramian"

& $Python (Join-Path $RepoRoot "scripts\generate_26state_dataset.py") `
    --cycles-root $CyclesRoot `
    --output-root $OutputRoot `
    --max-cycles-per-label $MaxCyclesPerLabel `
    --gramian $Gramian `
    --image-size 64 `
    --workers $Workers
if ($LASTEXITCODE -ne 0) {
    throw "Dataset generation failed with exit code $LASTEXITCODE"
}

$Manifest = Join-Path $OutputRoot "dataset_manifest.csv"
$Split = Join-Path $OutputRoot "grouped_split.csv"
$RunDir = Join-Path $OutputRoot "runs\conservative_cnn"

& $Python (Join-Path $RepoRoot "scripts\create_grouped_split.py") `
    --manifest $Manifest `
    --output $Split `
    --val-fraction 0.1 `
    --test-fraction 0.1 `
    --seed 42
if ($LASTEXITCODE -ne 0) {
    throw "Grouped split creation failed with exit code $LASTEXITCODE"
}

& $Python (Join-Path $RepoRoot "scripts\validate_dataset.py") `
    --manifest $Split `
    --sample-per-label 20
if ($LASTEXITCODE -ne 0) {
    throw "Dataset validation failed with exit code $LASTEXITCODE"
}

& $Python (Join-Path $RepoRoot "scripts\train_from_manifest.py") `
    --manifest $Split `
    --output-dir $RunDir `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --num-workers $NumWorkers `
    --seed 42 `
    --device auto
if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Pipeline completed."
Write-Host "Manifest: $Manifest"
Write-Host "Grouped split: $Split"
Write-Host "Metrics: $(Join-Path $RunDir 'metrics.json')"
