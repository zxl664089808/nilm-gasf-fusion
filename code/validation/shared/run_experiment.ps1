param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "channel",
        "encoding",
        "resolution",
        "models",
        "physics",
        "multitask",
        "seeds",
        "device"
    )]
    [string]$Experiment,
    [ValidateSet("26state", "11class")]
    [string]$Dataset = "26state",
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [int]$Epochs = 50,
    [int]$MaxPerLabel = 0,
    [string]$SeedChannels = "",
    [string]$DeviceChannels = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "config.ps1")

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python interpreter not found: $Python"
}

$env:PYTHONPATH = if ($Dataset -eq "26state") {
    Join-Path $Repo26 "src"
} else {
    Join-Path $Repo11 "src"
}

$Data = if ($Dataset -eq "26state") { $Data26 } else { $Data11 }
$Repo = if ($Dataset -eq "26state") { $Repo26 } else { $Repo11 }
$Manifest = Join-Path $Data "grouped_split.csv"
$ImageCache = Join-Path $Data "images_64.npy"
$TrainScript = if ($Dataset -eq "26state") {
    Join-Path $Repo "scripts\train_from_manifest.py"
} else {
    Join-Path $Repo "scripts\train_classifier.py"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$CommonArgs = @(
    "--manifest", $Manifest,
    "--train-script", $TrainScript,
    "--output-root", $OutputRoot,
    "--epochs", "$Epochs",
    "--batch-size", "64",
    "--num-workers", "4"
)
if ($MaxPerLabel -gt 0) {
    $CommonArgs += @("--max-per-label", "$MaxPerLabel")
}
if (Test-Path -LiteralPath $ImageCache) {
    $CommonArgs += @("--image-cache", $ImageCache)
}

switch ($Experiment) {
    "channel" {
        & $Python (Join-Path $Repo "scripts\run_benchmark.py") @CommonArgs --only channels
    }
    "models" {
        & $Python (Join-Path $Repo "scripts\run_benchmark.py") @CommonArgs --only models
    }
    "seeds" {
        if ([string]::IsNullOrWhiteSpace($SeedChannels)) {
            $SeedChannels = if ($Dataset -eq "26state") { "i,p,vip" } else { "i,p,vi" }
        }
        & $Python (Join-Path $Repo "scripts\run_benchmark.py") @CommonArgs --only seeds --seeds 42,43,44 --seed-channels $SeedChannels
    }
    "resolution" {
        if ($Dataset -ne "26state") {
            throw "Resolution experiments are currently configured for 26-state data."
        }
        $Manifest96 = Join-Path $CleanData "plaid_26state_gasf_96\grouped_split.csv"
        $Manifest128 = Join-Path $CleanData "plaid_26state_gasf_128\grouped_split.csv"
        if (-not (Test-Path -LiteralPath $Manifest96) -or -not (Test-Path -LiteralPath $Manifest128)) {
            throw "Generate and split 96/128 datasets first. See 03_resolution_ablation/README.md."
        }
        & $Python (Join-Path $Repo "scripts\run_benchmark.py") @CommonArgs --only resolution --manifest-96 $Manifest96 --manifest-128 $Manifest128
    }
    "encoding" {
        if ($Dataset -ne "26state") {
            throw "Raw 1D and V-I encoding baselines are currently configured for 26-state data."
        }
        $RawCache = Join-Path $Data "raw_waveforms.npy"
        if (-not (Test-Path -LiteralPath $RawCache)) {
            & $Python (Join-Path $Repo "scripts\build_raw_cache.py") --manifest $Manifest --output $RawCache
        }
        $Raw1D = Join-Path $OutputRoot "raw1d"
        $VI = Join-Path $OutputRoot "vi_trajectory"
        & $Python (Join-Path $Repo "scripts\train_raw.py") --manifest $Manifest --cache $RawCache --output-dir $Raw1D --input-mode raw1d --epochs $Epochs --batch-size 64 --num-workers 4
        & $Python (Join-Path $Repo "scripts\train_raw.py") --manifest $Manifest --cache $RawCache --output-dir $VI --input-mode vi --epochs $Epochs --batch-size 64 --num-workers 4
    }
    "physics" {
        if ($Dataset -ne "26state") {
            throw "Physics fusion is currently configured for 26-state data."
        }
        $Features = Join-Path $Data "physical_features.csv"
        if (-not (Test-Path -LiteralPath $Features)) {
            & $Python (Join-Path $Repo "scripts\build_physical_features.py") --manifest $Manifest --output $Features
        }
        & $Python (Join-Path $Repo "scripts\train_hybrid.py") --manifest $Manifest --physical-features $Features --output-dir (Join-Path $OutputRoot "hybrid_cnn_physics") --epochs $Epochs --batch-size 64 --num-workers 4
    }
    "multitask" {
        if ($Dataset -ne "26state") {
            throw "The multitask type/state model requires 26-state labels."
        }
        & $Python (Join-Path $Repo "scripts\train_multitask.py") --manifest $Manifest --output-dir (Join-Path $OutputRoot "multitask_cnn") --epochs $Epochs --batch-size 64 --num-workers 4
    }
    "device" {
        if ([string]::IsNullOrWhiteSpace($DeviceChannels)) {
            $DeviceChannels = if ($Dataset -eq "26state") { "vip" } else { "i" }
        }
        $DeviceRunDir = Join-Path $OutputRoot "leave_device_out_$DeviceChannels"
        if ($Dataset -eq "11class") {
            $DeviceMap = Join-Path $Data "device_map.csv"
            $DeviceSplit = Join-Path $Data "device_split.csv"
            if (-not (Test-Path -LiteralPath $DeviceMap)) {
                & $Python (Join-Path $Repo "scripts\build_device_map.py") --metadata-docx "D:\PLAID-projiect  2025.9.9\sub_meta.docx" --manifest $Manifest --output $DeviceMap
            }
            if (-not (Test-Path -LiteralPath $DeviceSplit)) {
                & $Python (Join-Path $Repo "scripts\create_device_split.py") --manifest (Join-Path $Data "dataset_manifest.csv") --device-map $DeviceMap --output $DeviceSplit
            }
            $DeviceArgs = @(
                "--manifest", $DeviceSplit,
                "--output-dir", $DeviceRunDir,
                "--epochs", "$Epochs",
                "--batch-size", "64",
                "--num-workers", "4",
                "--channels", $DeviceChannels
            )
            if ($MaxPerLabel -gt 0) {
                $DeviceArgs += @("--max-per-label", "$MaxPerLabel")
            }
            & $Python (Join-Path $Repo "scripts\train_classifier.py") @DeviceArgs
        } else {
            $DeviceSplit = Join-Path $Data "device_split.csv"
            if (-not (Test-Path -LiteralPath $DeviceSplit)) {
                throw "Provide a source_id,device_id map and create device_split.csv first."
            }
            $DeviceArgs = @(
                "--manifest", $DeviceSplit,
                "--output-dir", $DeviceRunDir,
                "--epochs", "$Epochs",
                "--batch-size", "64",
                "--num-workers", "4",
                "--channels", $DeviceChannels
            )
            if ($MaxPerLabel -gt 0) {
                $DeviceArgs += @("--max-per-label", "$MaxPerLabel")
            }
            & $Python (Join-Path $Repo "scripts\train_from_manifest.py") @DeviceArgs
        }
    }
}

if ($LASTEXITCODE -ne 0) {
    throw "Experiment failed with exit code $LASTEXITCODE"
}
