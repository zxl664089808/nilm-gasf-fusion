$SuiteRoot = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $SuiteRoot
$Python = "D:\PLAID-projiect  2025.9.9\GLM-1\.venv\Scripts\python.exe"
$Repo26 = Join-Path $WorkspaceRoot "nilm-clean-pipeline"
$Repo11 = Join-Path $WorkspaceRoot "nilm-11class-clean"
$CleanData = Join-Path $WorkspaceRoot "clean_data"
$Data26 = Join-Path $CleanData "plaid_26state_gasf_64"
$Data11 = Join-Path $CleanData "plaid_11class_gasf_64"
