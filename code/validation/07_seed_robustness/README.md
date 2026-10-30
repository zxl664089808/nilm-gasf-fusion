# 07 Seed robustness

Runs seeds 42, 43, and 44 and writes a combined benchmark summary.

The candidate channel combinations can be selected explicitly:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Dataset 26state
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Dataset 11class
```

For example, the 26-state candidate matrix is:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run.ps1 `
  -Dataset 26state `
  -Channels "i,p,vip"
```
