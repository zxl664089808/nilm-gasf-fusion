param(
    [ValidateSet("26state", "11class")][string]$Dataset = "26state",
    [int]$Epochs = 50,
    [int]$MaxPerLabel = 0
)
& (Join-Path $PSScriptRoot "..\shared\run_experiment.ps1") -Experiment channel -Dataset $Dataset -OutputRoot (Join-Path $PSScriptRoot "runs\$Dataset") -Epochs $Epochs -MaxPerLabel $MaxPerLabel
