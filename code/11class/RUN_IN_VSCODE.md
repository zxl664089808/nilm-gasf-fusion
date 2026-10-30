# 在 VS Code 中运行 11 类项目

## 打开项目

```text
D:\论文1\nilm-11class-clean
```

按 `Ctrl+Shift+P`，选择 `Python: Select Interpreter`：

```text
D:\PLAID-projiect  2025.9.9\GLM-1\.venv\Scripts\python.exe
```

打开 `Terminal > New Terminal`。

## 烟测

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1 -Smoke
```

烟测行为：

- 每类读取前 5 个原始 CSV 样本；
- 每类最多保留 100 个真实周期；
- 生成无标题纯图像；
- 按原始 CSV 样本分组；
- 验证无来源重叠；
- 训练 2 轮并生成训练历史图。

输出：

```text
D:\论文1\clean_data\plaid_11class_smoke
```

## 全量实验

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1
```

默认行为：

- 11 类电器；
- 每类从全部原始 CSV 样本中抽取最多 5,000 个周期；
- 使用真实周期，不复制图片补足数量；
- 64×64 纯 RGB 图像；
- 固定使用 GASF（Gramian Angular Summation Field）；
- 80:10:10 按原始 CSV 样本分组；
- 50 个训练 epoch。

输出：

```text
D:\论文1\clean_data\plaid_11class_gasf_64
```

结果文件：

```text
runs\conservative_cnn\metrics.json
runs\conservative_cnn\confusion_matrix.csv
runs\conservative_cnn\classification_report.txt
runs\conservative_cnn\best_model.pth
runs\conservative_cnn\training_history.png
runs\conservative_cnn\training_history.pdf
runs\conservative_cnn\training_history.csv
```

## 中断恢复

每类完成后会保存独立 manifest。重新执行同一命令时，已完成的类别会跳过。

## 注意

这个项目只区分 11 种电器类别，不包含 ON/OFF 状态。第六步的 26 状态识别在另一个项目 `nilm-clean-pipeline` 中。
