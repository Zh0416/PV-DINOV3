# Cross-Matching PV-896 Kaggle 工程试跑与开发试验（2026-11-04）

本报告记录 `baseline` 提交 `4391ab963f1c518218720a372903f4d3b300d9a5` 的两次私有 Kaggle GPU 运行。工程试跑：[PV ResNet50 Cross Matching Smoke](https://www.kaggle.com/code/zh0416/pv-resnet50-cross-matching-smoke)；有界开发试验：[PV ResNet50 Cross Matching Pilot](https://www.kaggle.com/code/zh0416/pv-resnet50-cross-matching-pilot)。两次运行均为 `COMPLETE`，不代表精度复现成功。

## 固定输入与环境

- 数据集：`zh0416/pv-896`，实际路径 `/kaggle/input/datasets/zh0416/pv-896/PV-896`；`manifest.csv` SHA-256 为 `6ec028cd1f43255a61a969057799662270fc770498aab8b40fde8f7b6b338522`。
- ImageNet ResNet50 权重：Kaggle 会话内从 torchvision 官方 URL 下载 `resnet50-0676ba61.pth`；完整 SHA-256 为 `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`。没有下载 DINOv3-SAT、GeoRSCLIP 或 SCCNet 已训练匹配头。作者 README 声明其原先提供的 SCCNet 检查点已丢失；匹配头为全新随机初始化。
- 硬件：Tesla T4；PyTorch `2.11.0+cu130`，torchvision `0.26.0+cu130`。训练与验证从原始 896×896 影像裁取原分辨率 256×256 patch，验证用 16 块覆盖整图。
- `seed=42` 的空间分组将 `0825_Q_733feba5ac7648` 和 `0918_Q_4a6bb66935b24c` 留为开发验证，和此前项目开发数据存在重用；这不是独立测试集。没有从新区域测得泛化性能。

## 实测形状、显存与速度

每个前向的相关张量依次为 `[1,3,8,8,8,8]`、`[1,6,16,16,16,16]`、`[1,4,32,32,32,32]`。六个轴为 batch、同尺度特征层、Query 高/宽、Support 高/宽；没有在 896 原图尺度构造稠密逐像素矩阵。

工程试跑使用 3 个训练 Query、3 个整图开发 Query，1 epoch、3 次更新。完整的 ResNet50 官方权重下载、4 项合成测试、真实数据训练、整图验证和检查点生成均通过。预热后的两个训练 step 分别约 `0.078` 秒，峰值已分配 CUDA 显存约 `0.202 GiB`。工程试跑的前景 IoU 为 0，不作为性能结论。

有界开发试验使用 60 个训练 Query、12 个开发 Query，batch size 1、3 epoch、180 次更新。训练 step 预热后中位耗时 `0.0777` 秒，峰值已分配 CUDA 显存 `217183232` bytes（约 `0.202 GiB`）。这些是 PyTorch 的 **allocated** 峰值，不是 GPU 总占用或 reserved 显存；整轮实际耗时还包括影像读取与整图验证。

## 开发结果

| Epoch | Train loss | 1-shot 前景 IoU | TP | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4589 | 0 | 0 | 0 | 927120 |
| 2 | 0.4071 | 0 | 0 | 0 | 927120 |
| 3 | 0.3527 | 0 | 0 | 0 | 927120 |

按开发前景 IoU 选择的最佳权重来自 epoch 1。使用同一最佳权重和相同 12 张 Query 做 5-shot 整图验证，五张支持图来自五个不同空间组，结果仍为 TP=0、FP=0、FN=927120，前景 IoU=0。12 张 Query 中 8 张有光伏，全部未检出；4 张空掩膜 Query 无误检。训练损失下降只说明优化过程在运行，不能证明分割已学会前景。

当前结论是**工程路径通过，性能基线尚未建立**。下一步应先在训练侧记录前景预测分布、支持/查询 patch 的前景占比及训练 Query 的 TP/FP/FN，并做小批量正样本拟合检查；随后在独立对照下检验训练步数和类别不平衡的影响。任何加权损失或采样改动都需要单独标为光伏适配实验，保留当前开源结构与损失的结果作参照。不得把这两次开发运行的零 IoU 归因于某个单独模块，尚无足够诊断证据。

小型 JSON 报告已从 Kaggle 输出下载至本地 `G:\PV-GF\tmp\cross_matching_kaggle_smoke_20261104` 和 `G:\PV-GF\tmp\cross_matching_kaggle_pilot_20261104`。原始数据和训练检查点未提交到 GitHub。
