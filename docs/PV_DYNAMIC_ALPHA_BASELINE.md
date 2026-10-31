# 局部动态权重基线

模型流程图和逐模块输入/输出：[结构与张量接口](PV_DYNAMIC_ALPHA_ARCHITECTURE.md)。

分支：`codex/pv-dynamic-alpha-baseline`

基线实验：私有 Kaggle `zh0416/pv-brightness-dynamic-fusion`，Version 1

源码起点：GitHub 分支 `codex/sspnet-dinov3-pv`，提交 `aab8827`

## 当前最优候选

保留冻结的 DINOv3-SAT 空间响应、GeoRSCLIP 光伏语义响应和原亮度门控。轻量卷积网络根据两路响应、亮度门控及局部统计，为每个 56×56 位置预测语义融合权重 `alpha`：

```text
S_b = clip(S - P*g, -1, 1)
F = (1 - alpha) * D + alpha * S_b
```

`alpha` 由 8 通道的 3×3 卷积、GELU 和 1×1 卷积生成，经 sigmoid 限制在 `[0,1]`，初始化为 0.7。两路 CAM 和权重分别上采样到 896×896 后融合。只有权重网络可训练；训练目标是类别平衡 BCE、0.5 Dice、权重锚定和总变差正则。

## 已观测结果

以下来自同一批 24 张已多次查看的开发图；每种方案按相同 3 个 1-shot 支持选择评估，阈值由 12 张校准图确定。

| 方法 | 24 张 IoU | 大棚场景 13 张 IoU | 大棚场景 Recall |
|---|---:|---:|---:|
| 原亮度门控，固定 alpha=0.7 | 49.02% | 47.67% | 85.53% |
| 全局可学习 alpha | 49.41% | 47.94% | 85.26% |
| **局部动态 alpha** | **51.26%** | **49.80%** | **86.06%** |

局部动态方案的大棚场景背景 FP 均值为 726,852，相比固定权重的 789,872 减少约 7.98%。三个支持选择的大棚 IoU 均提高。17 张正图中有 14 张提高、3 张下降，所以该结果不代表逐图普遍改善。

这是小型融合先验实验，不是完整 SSPNet 的端到端分割成绩，也不是独立区域测试。学习权重的方案使用了 22 张训练掩膜；固定权重没有这项训练监督，整体差异同时包含额外监督收益。后续比较应保持数据角色和校准流程一致，并增加未查看过的空间区域。

## 训练选择与 checkpoint

训练配置固定为 seed 42、320 个 episode、AdamW、学习率 0.001；每 80 步按 12 张校准图选择 checkpoint。局部网络选中第 160 步，包含 593 个参数。锁定记录中的 checkpoint SHA256 为：

```text
local_selected.pt  91d355ca014a325dc566d2229539f29bd5e9d0e75ccfbf4c7422edd26f7bafc5
```

**当前 Git 工作区没有 `local_selected.pt` 权重文件。** Kaggle 输出下载中保留了指标、训练曲线和逐位置 alpha 图，但未包含 checkpoint；因此本分支保存了可复跑代码和评估证据，尚不能直接从选中权重继续训练。不要把 `weight_grids.npz` 当作模型权重，它只记录校准/查询图上的预测 alpha。需要继续微调时，应从上述 Kaggle 运行取回 `local_selected.pt`，并核对 SHA256；若无法取回，则按固定配置重新训练并重新选择模型。

## 文件与验证

- `pvseg/brightness_gate.py`、`pvseg/greenhouse_prior.py`：亮度与语义修正输入。
- `pvseg/dynamic_brightness_fusion.py`：局部 alpha 网络与凸组合。
- `pvseg/losses.py`：训练用类别平衡 BCE + Dice。
- `scripts/kaggle_dynamic_brightness.py`：训练、校准和查询评估。
- `scripts/verify_dynamic_brightness.py`：独立重建阈值与混淆计数。
- `scripts/diagnose_dynamic_brightness_errors.py`：锁定结果后的 FP/FN 能力诊断。
- `configs/dynamic_brightness_20261030.json`：固定训练参数和查询名单。
- `docs/PV_DYNAMIC_BRIGHTNESS_20261030.md`：完整实验表格和检查记录。

本地模块测试：

```powershell
python -m unittest discover -s tests -p test_dynamic_brightness_fusion.py -v
python -m unittest discover -s tests -p test_brightness_gate.py -v
python -m unittest discover -s tests -p test_greenhouse_prior.py -v
python -m unittest discover -s tests -p test_segmentation_loss.py -v
```

训练脚本要求 Kaggle GPU 环境、私有数据和先前缓存；权重和影像没有纳入 GitHub 仓库。
