# 图像生成程序标签泄漏审查报告

审查日期：2026-10-21  
审查范围：11 类 55,000 张实验、26 状态 130,000 张实验的图像生成、训练读取和数据划分  
审查结论：**确认存在明确的标签泄漏。现有 99% 级准确率不能作为波形分类的泛化性能。**

## 一、最终结论

生成脚本把电器类别和 ON/OFF 状态直接写进了训练 PNG 的标题区域。训练脚本读取整张 PNG，并把它缩放到 64×64 后送入 CNN。因此模型输入不仅包含电压、电流和功率的 Gramian 图形，还包含可直接区分 11 类或 26 类的文字。

这不是文件名中的标签被直接传给网络，而是**标签文字被绘制成了图像像素**。它比普通文件名泄漏更隐蔽，因为从目录结构看不出来，只有打开 PNG 才能发现。

## 二、26 状态实验的泄漏路径

### 2.1 生成函数

文件：

`D:\PLAID-projiect  2025.9.9\GLM-1\1.22all_appliances_chuangkou-data_amplitude-RGB -all.py`

关键代码：

```python
def save_rgb_image(rgb_image, sample_index, output_dir, appliance_name, status):
    image_filename = f"{safe_appliance_name}_{status}_gadf_rgb_{sample_index:04d}.png"
    image_path = os.path.join(output_dir, image_filename)

    plt.figure(figsize=(8, 6))
    plt.imshow(rgb_image)
    plt.title(f'{appliance_name} {status} - RGB GADF (Sample {sample_index})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(image_path, dpi=150, bbox_inches='tight')
```

对应当前文件大致行号：

- 第 58 行：`save_rgb_image`
- 第 64 行：生成包含类别和状态的图片文件名
- 第 67 行：创建 Matplotlib figure
- 第 68 行：绘制 RGB 矩阵
- 第 69 行：把 `appliance_name` 和 `status` 写为标题
- 第 72 行：保存整张 figure，而不是只保存 `rgb_image` 矩阵

标题示例：

`Air Conditioner ON - RGB GADF (Sample 1)`

### 2.2 训练读取

文件：

`D:\PLAID-projiect  2025.9.9\GLM-1\1.22all_appliances_chuangkou-data_amplitude-RGB-recognize -all.py`

关键代码：

```python
image = Image.open(self.image_paths[idx]).convert('RGB')
label = self.labels[idx]

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```

对应当前文件大致行号：

- 第 116 行：读取整张 PNG
- 第 128 行：直接缩放到 64×64
- 第 266–318 行：从 `rgb_gadf_images` 目录加载所有 PNG
- 第 342–362 行：按图像随机划分训练、验证和测试集

原始 PNG 尺寸约为 851×883。缩放到 64×64 后，顶部标题仍会进入模型输入。

### 2.3 实际图像证据

代表性文件：

`D:\PLAID-projiect  2025.9.9\GLM-1\1.22all_appliances_chuangkou-data_amplitude-RGB -all\comprehensive_plots\Air_Conditioner\ON\rgb_gadf_images\Air_Conditioner_ON_gadf_rgb_0001.png`

打开后顶部明确显示：

`Air Conditioner ON - RGB GADF (Sample 1)`

多类标题带拼图：

`00_project_admin/evidence/title_band_examples.png`

该图显示 Air Conditioner、Blender、Coffee maker 和 Fan 的 ON/OFF 标题均被烧录进 PNG 像素。

## 三、11 类实验同样存在泄漏

文件：

`D:\PLAID-projiect  2025.9.9\GLM-1\1.2.1jibo_5500gaf(max-min-xiangliang).py`

关键代码位于当前文件第 158–164 行附近：

```python
plt.figure(figsize=(8, 6))
plt.imshow(rgb_image)
plt.title(f'{appliance_type} - RGB GADF (Cycle {i+1})')
plt.axis('off')
plt.tight_layout()
plt.savefig(image_path, dpi=150, bbox_inches='tight')
plt.close()
```

训练脚本：

`D:\PLAID-projiect  2025.9.9\GLM-1\1.12CNN_55000_conservative(max-min-xiangliang).py`

训练同样读取整张 PNG，并缩放到 64×64：

```python
image = Image.open(self.image_paths[idx]).convert('RGB')
...
transforms.Resize((64, 64))
```

已核对 `Save1_1.2.1jibo_5500gaf(max-min-xiangliang)`：

- 11 个类别；
- 每类 5,000 张；
- 合计 55,000 张；
- 所有图像均由同一带标题的保存逻辑生成。

因此 55,000 张实验的 99.96% 结果也受到同样的标签泄漏影响。

## 四、文字标题是否足以完成分类

### 4.1 顶行分类实验

使用每类 200 张的独立诊断缓存，只取缩放后 64×64 图像的最顶部一行，将其展平后训练多分类逻辑回归，测试 26 个标签。

结果：

| 顶部区域 | 输入内容 | 测试准确率 | Macro-F1 |
|---|---|---:|---:|
| 第 1 行 | 带标题原图 | 99.83% | 99.83% |
| 第 1 行 | 去除标题后的图像 | 92.56% | 92.59% |
| 第 3 行 | 带标题原图 | 99.83% | 99.83% |
| 第 3 行 | 去除标题后的图像 | 95.46% | 95.46% |

只保留顶部一行时，26 类仍达到 99.83%，说明顶部区域包含几乎确定性的类别信息。视觉检查确认该区域是类别和状态标题。

### 4.2 顶部区域 CNN 消融

只保留 64×64 图像顶部 12 行，其余像素置零，使用相同轻量 CNN：

- 带标题：87.83%；
- 去除标题：76.42%。

顶部 12 行同时包含标题和少量波形内容，因此差值表示标题区域带来的额外可分性。仅一行输入的 99.83% 则是更强的直接证据。

### 4.3 复现命令

```powershell
python 03_experiments\analyze_title_band.py `
  --cache-dir D:\论文1\03_experiments\cache `
  --split-column split_image `
  --max-per-label 200 `
  --seed 42
```

## 五、第二个独立问题：同源周期跨集合

26 状态实验使用 `train_test_split` 直接按图像随机划分，没有按原始录波文件分组。每个原始录波平均贡献约 100–200 个相邻周期，这些周期高度相关。

对 130,000 张图像逐一恢复来源后的统计：

| 划分方式 | 测试图像所属录波文件同时出现在训练集中的比例 |
|---|---:|
| 图像级随机划分 | 99.98% |
| 按录波文件分组 | 0.00% |

因此，即使修复标题文字，历史图像级随机划分仍然不是一个独立测试。

## 六、对现有训练和识别结果的影响

### 6.1 可以确认的影响

- 99.67% 和 99.96% 不是纯波形分类性能；
- 不能作为论文中的泛化准确率；
- 不能据此声称模型已经学会了可靠的电压/电流特征；
- 在新采集的、不含标题的原始信号上，模型表现没有得到验证；
- 在真实 NILM 部署中不会有类别标题，因此当前模型输入的分布与部署输入不一致。

### 6.2 不能直接推断的部分

- 不能仅凭这个错误断言模型完全没有学到波形特征；
- 顶部条带消融的纯图像区域仍有一定分类能力，说明 GADF 图形本身可能包含有效信息；
- 但现有实验无法区分“模型依赖文字”和“模型依赖波形”的比例，必须重新训练后才能重新评估。

准确表述是：**已有训练过程和模型文件不必然完全无用，但当前评估结果无效，不能用于论文或真实识别性能声明。**

## 七、其他需要修正的代码问题

### 7.1 55,000 张脚本存在复制兜底逻辑

`1.12CNN_55000_conservative(max-min-xiangliang).py` 在图像不足 55,000 时循环复制已有图像，然后再随机划分。

当前实际检查为 11 × 5,000 = 55,000 张，因此这次运行没有触发复制。但该逻辑如果以后触发，会造成完全相同图像跨训练集和测试集。应删除，或改为仅在训练集内做数据增强。

### 7.2 11 类周期数据没有保存来源信息

`1.12jibo_waveforms.py` 保存周期时只写入：

- `voltage_cycles`
- `current_cycles`
- `device_type`
- `total_cycles`
- `extraction_time`

没有保存每个周期来自哪个 CSV、事件或录波文件。因此 11 类实验目前无法直接做按录波文件分组划分。需要重新提取并保留来源 ID。

### 7.3 GADF 实际为求和形式

两个生成脚本默认使用：

`sin(phi_i + phi_j)`

这是 Gramian Angular Summation Field (GASF)，不是差值形式 GADF：

`sin(phi_i - phi_j)`

这与标签泄漏独立，但论文中的方法名称必须与实现一致。

## 八、正确的图像保存方式

生成 RGB 矩阵后，不要创建 Matplotlib figure，直接保存数组：

```python
from PIL import Image
import numpy as np

rgb_uint8 = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
Image.fromarray(rgb_uint8, mode="RGB").save(image_path)
```

或者：

```python
plt.imsave(image_path, rgb_image, format="png")
```

正式训练图中禁止出现：

- `plt.figure()`
- `plt.title()`
- `plt.xlabel()` / `plt.ylabel()`
- `plt.colorbar()`
- `plt.savefig()`
- 文件名、类别名、日期、编号或水印

建议保存纯图像后再次读取，强制统一为固定尺寸，例如直接生成 64×64 或 128×128 的 RGB 数组。

## 九、重训前检查表

- [ ] 生成图不含标题、文字、边框和多余留白。
- [ ] 每类随机抽查至少 20 张，逐张打开确认。
- [ ] 用 OCR 或模板检查确认图像中无类别文字。
- [ ] 保存每个图像对应的原始录波文件 ID。
- [ ] 以原始录波文件为单位划分训练/验证/测试。
- [ ] 检查不同集合之间的 `group_id` 交集为 0。
- [ ] 删除 55,000 脚本中的图像复制兜底。
- [ ] 确认最终使用 GASF 还是 GADF。
- [ ] 重新训练 11 类和 26 状态两组实验。
- [ ] 用纯图像模型做一次真实原始波形推理测试。

## 十、最终判定

标签泄漏已经确认，来源是图像生成脚本中的 `plt.title(...)` 和整张 figure 保存。训练脚本读取整张 PNG，导致类别文字进入网络输入。

因此：

1. 所有基于带标题 PNG 的准确率都不能作为论文主结果；
2. 现有结果不能证明模型在真实无标题输入上的识别能力；
3. 必须重新生成纯图像并重新训练；
4. 同时需要修复图像级随机划分造成的同源周期泄漏；
5. 修复后的模型和结果才可以进入英文论文初稿的主结果部分。
