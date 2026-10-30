param(
    [ValidateSet("26state", "11class")][string]$Dataset = "26state",
    [int]$Epochs = 50,
    [int]$MaxPerLabel = 0,
    [string]$Channels = ""
)
& (Join-Path $PSScriptRoot "..\shared\run_experiment.ps1") -Experiment seeds -Dataset $Dataset -OutputRoot (Join-Path $PSScriptRoot "runs\$Dataset") -Epochs $Epochs -MaxPerLabel $MaxPerLabel -SeedChannels $Channels
