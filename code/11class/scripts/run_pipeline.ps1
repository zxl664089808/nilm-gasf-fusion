param(
    [string]$Python = "D:\PLAID-projiect  2025.9.9\GLM-1\.venv\Scripts\python.exe",
    [string]$SampleIndex = "D:\PLAID-projiect  2025.9.9\GLM-1\Save1_fixed\new_csv_fixed.pkl",
    [string]$CsvRoot = "D:\PLAID-projiect  2025.9.9\GLM-1\Save1_fixed\new_csv",
    [string]$OutputRoot = "",
    [int]$MaxCyclesPerClass = 5000,
    [int]$SamplesPerClass = 0,
    [int]$Workers = 4,
    [int]$Epochs = 50,
    [int]$BatchSize = 64,
    [int]$NumWorkers = 4,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $RepoRoot
$Gramian = "gasf"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python interpreter not found: $Python"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $WorkspaceRoot "clean_data\plaid_11class_gasf_64"
}
if ($Smoke) {
    $OutputRoot = Join-Path $WorkspaceRoot "clean_data\plaid_11class_smoke"
    $MaxCyclesPerClass = 100
    $SamplesPerClass = 5
    $Workers = 2
    $Epochs = 2
    $NumWorkers = 0
}

$env:PYTHONPATH = Join-Path $RepoRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"

Write-Host "Repository: $RepoRoot"
Write-Host "Python: $Python"
Write-Host "Output: $OutputRoot"
Write-Host "Classes: 11 appliance categories"
Write-Host "Gramian: GASF (fixed)"

& $Python (Join-Path $RepoRoot "scripts\generate_dataset.py") `
    --sample-index $SampleIndex `
    --csv-root $CsvRoot `
    --output-root $OutputRoot `
    --max-cycles-per-class $MaxCyclesPerClass `
    --samples-per-class $SamplesPerClass `
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

& $Python (Join-Path $RepoRoot "scripts\train_classifier.py") `
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

& $Python (Join-Path $RepoRoot "scripts\plot_training_history.py") `
    --metrics (Join-Path $RunDir "metrics.json") `
    --output-prefix (Join-Path $RunDir "training_history")
if ($LASTEXITCODE -ne 0) {
    throw "Training history plotting failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "11-class pipeline completed."
Write-Host "Manifest: $Manifest"
Write-Host "Grouped split: $Split"
Write-Host "Metrics: $(Join-Path $RunDir 'metrics.json')"
Write-Host "History plot: $(Join-Path $RunDir 'training_history.png')"
