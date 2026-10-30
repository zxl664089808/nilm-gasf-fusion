# 在 VS Code 中运行修正版程序

## 1. 打开正确的文件夹

在 VS Code 中选择：

```text
File > Open Folder
```

打开：

```text
D:\论文1\nilm-clean-pipeline
```

不要打开旧的 `D:\PLAID-projiect  2025.9.9\GLM-1`。

## 2. 选择 Python 解释器

按 `Ctrl+Shift+P`，选择：

```text
Python: Select Interpreter
```

输入并选择现有带 CUDA 的环境：

```text
D:\PLAID-projiect  2025.9.9\GLM-1\.venv\Scripts\python.exe
```

这个环境已经包含 PyTorch、NumPy、Pandas 和 Pillow。

## 3. 打开 VS Code 终端

使用：

```text
Terminal > New Terminal
```

终端应位于：

```text
D:\论文1\nilm-clean-pipeline
```

如果 PowerShell 不允许执行脚本，仅对当前终端放开：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## 4. 先跑小样本烟测

```powershell
.\scripts\run_pipeline.ps1 -Smoke
```

它会依次完成：

1. 每类生成 100 张纯图像；
2. 按原始录波文件分组；
3. 检查不同集合之间没有来源重叠；
4. 检查图像顶部没有标题文字；
5. 训练 2 轮并输出指标。

烟测输出目录：

```text
D:\论文1\clean_data\plaid_26state_smoke
```

确认烟测成功后，再跑全量。

## 5. 跑全量修正实验

```powershell
.\scripts\run_pipeline.ps1
```

默认配置：

- 26 个标签；
- 每类最多 5,000 张；
- 64×64 纯 RGB 图像；
- GASF；
- 8 个数据生成进程；
- 80:10:10 按录波文件分组；
- 50 个训练 epoch；
- 自动使用 CUDA（如果可用）。

全量输出目录：

```text
D:\论文1\clean_data\plaid_26state_gasf_64
```

主要文件：

```text
dataset_manifest.csv
grouped_split.csv
runs\conservative_cnn\metrics.json
runs\conservative_cnn\confusion_matrix.csv
runs\conservative_cnn\classification_report.txt
runs\conservative_cnn\best_model.pth
```

训练结束后生成训练历史图：

```powershell
python .\scripts\plot_training_history.py `
  --metrics "D:\论文1\clean_data\plaid_26state_gasf_64\runs\conservative_cnn\metrics.json" `
  --output-prefix "D:\论文1\clean_data\plaid_26state_gasf_64\runs\conservative_cnn\training_history"
```

会生成 `training_history.png`、`training_history.pdf` 和
`training_history.csv`。

## 6. 中断后继续

数据生成器会按类别保存 manifest。如果中途停止，重新执行同一条命令即可跳过已经完成的类别：

```powershell
.\scripts\run_pipeline.ps1
```

如果希望强制重新生成所有类别：

```powershell
.\scripts\run_pipeline.ps1 -OutputRoot "D:\论文1\clean_data\plaid_26state_gasf_64_new"
```

## 7. 使用 GADF

如果你最终决定使用差值形式 GADF：

```powershell
.\scripts\run_pipeline.ps1 `
  -Gramian gadf `
  -OutputRoot "D:\论文1\clean_data\plaid_26state_gadf_64"
```

旧程序实际计算的是：

```text
sin(phi_i + phi_j)
```

这是 GASF。论文中的名称必须与最终运行参数一致。

## 8. 手动运行各个步骤

如果不使用 PowerShell 脚本，也可以分开执行：

```powershell
$env:PYTHONPATH = "D:\论文1\nilm-clean-pipeline\src"

python .\scripts\generate_26state_dataset.py `
  --cycles-root "D:\PLAID-projiect  2025.9.9\GLM-1\extracted_cycles_output" `
  --output-root "D:\论文1\clean_data\plaid_26state_gasf_64" `
  --max-cycles-per-label 5000 `
  --gramian gasf `
  --image-size 64 `
  --workers 8

python .\scripts\create_grouped_split.py `
  --manifest "D:\论文1\clean_data\plaid_26state_gasf_64\dataset_manifest.csv" `
  --output "D:\论文1\clean_data\plaid_26state_gasf_64\grouped_split.csv"

python .\scripts\validate_dataset.py `
  --manifest "D:\论文1\clean_data\plaid_26state_gasf_64\grouped_split.csv"

python .\scripts\train_from_manifest.py `
  --manifest "D:\论文1\clean_data\plaid_26state_gasf_64\grouped_split.csv" `
  --output-dir "D:\论文1\clean_data\plaid_26state_gasf_64\runs\conservative_cnn"
```

## 9. 注意

- 不要使用 VS Code 右上角的“Run Python File”按钮运行这些脚本，因为生成和训练脚本需要命令行参数。
- 使用集成终端运行上面的命令。
- 旧项目的 99.67% 和 99.96% 不能与新结果直接比较。
