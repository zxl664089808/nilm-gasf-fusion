# 01 Channel ablation

Compares V, I, P, V+I, and V+I+P.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Dataset 26state
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Dataset 11class
```

Results are written under `runs/<dataset>`.
