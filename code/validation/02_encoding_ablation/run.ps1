param([int]$Epochs = 50, [int]$MaxPerLabel = 0)
& (Join-Path $PSScriptRoot "..\shared\run_experiment.ps1") -Experiment encoding -Dataset 26state -OutputRoot (Join-Path $PSScriptRoot "runs") -Epochs $Epochs -MaxPerLabel $MaxPerLabel
