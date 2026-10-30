param(
    [string]$SplitSeeds = "42",
    [string]$Channels = "i,vip",
    [int]$Epochs = 50,
    [int]$MaxPerLabel = 0,
    [switch]$SkipHybrid
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\shared\config.ps1")

$DatasetManifest = Join-Path $Data11 "dataset_manifest.csv"
$DeviceMap = Join-Path $Data11 "device_map.csv"
$Features = Join-Path $Data11 "physical_features.csv"
$env:PYTHONPATH = Join-Path $Repo11 "src"

if (-not (Test-Path -LiteralPath $DeviceMap)) {
    & $Python (Join-Path $Repo11 "scripts\build_device_map.py") `
        --metadata-docx "D:\PLAID-projiect  2025.9.9\sub_meta.docx" `
        --manifest $DatasetManifest `
        --output $DeviceMap
    if ($LASTEXITCODE -ne 0) { throw "Device-map creation failed." }
}

$FirstSplit = Join-Path $Data11 "device_split_seed42.csv"
if (-not (Test-Path -LiteralPath $Features)) {
    $FeatureManifest = if (Test-Path -LiteralPath $FirstSplit) {
        $FirstSplit
    } else {
        $DatasetManifest
    }
    & $Python (Join-Path $Repo11 "scripts\build_physical_features.py") `
        --manifest $FeatureManifest `
        --csv-root "D:\PLAID-projiect  2025.9.9\GLM-1\Save1_fixed\new_csv" `
        --output $Features
    if ($LASTEXITCODE -ne 0) { throw "Physical-feature creation failed." }
}

foreach ($SplitSeed in $SplitSeeds.Split(",")) {
    $SplitSeed = $SplitSeed.Trim()
    if ([string]::IsNullOrWhiteSpace($SplitSeed)) { continue }
    $SplitPath = Join-Path $Data11 "device_split_seed$SplitSeed.csv"
    if (-not (Test-Path -LiteralPath $SplitPath)) {
        & $Python (Join-Path $Repo11 "scripts\create_device_split.py") `
            --manifest $DatasetManifest `
            --device-map $DeviceMap `
            --output $SplitPath `
            --seed $SplitSeed
        if ($LASTEXITCODE -ne 0) { throw "Device split creation failed for seed $SplitSeed." }
    }
    $SeedRoot = Join-Path $PSScriptRoot "runs\11class\device_seed$SplitSeed"

    foreach ($Channel in $Channels.Split(",")) {
        $Channel = $Channel.Trim()
        if ([string]::IsNullOrWhiteSpace($Channel)) { continue }

        $ImageRun = Join-Path $SeedRoot "image_$Channel"
        $ImageArgs = @(
            "--manifest", $SplitPath,
            "--output-dir", $ImageRun,
            "--channels", $Channel,
            "--epochs", "$Epochs",
            "--batch-size", "64",
            "--num-workers", "4"
        )
        if ($MaxPerLabel -gt 0) {
            $ImageArgs += @("--max-per-label", "$MaxPerLabel")
        }
        & $Python (Join-Path $Repo11 "scripts\train_classifier.py") @ImageArgs
        if ($LASTEXITCODE -ne 0) { throw "Image model failed: seed=$SplitSeed channel=$Channel" }
        & $Python (Join-Path $Repo11 "scripts\evaluate_predictions.py") `
            --predictions (Join-Path $ImageRun "test_predictions.csv") `
            --output-dir $ImageRun `
            --group-column device_id
        if ($LASTEXITCODE -ne 0) { throw "Prediction evaluation failed for image_$Channel" }

        if (-not $SkipHybrid) {
            $HybridRun = Join-Path $SeedRoot "hybrid_$Channel"
            $HybridArgs = @(
                "--manifest", $SplitPath,
                "--physical-features", $Features,
                "--output-dir", $HybridRun,
                "--channels", $Channel,
                "--epochs", "$Epochs",
                "--batch-size", "64",
                "--num-workers", "4"
            )
            if ($MaxPerLabel -gt 0) {
                $HybridArgs += @("--max-per-label", "$MaxPerLabel")
            }
            & $Python (Join-Path $Repo11 "scripts\train_hybrid.py") @HybridArgs
            if ($LASTEXITCODE -ne 0) { throw "Hybrid model failed: seed=$SplitSeed channel=$Channel" }
            & $Python (Join-Path $Repo11 "scripts\evaluate_predictions.py") `
                --predictions (Join-Path $HybridRun "test_predictions.csv") `
                --output-dir $HybridRun `
                --group-column device_id
            if ($LASTEXITCODE -ne 0) { throw "Prediction evaluation failed for hybrid_$Channel" }
        }
    }
}

Write-Host "Device-level experiment matrix completed."
