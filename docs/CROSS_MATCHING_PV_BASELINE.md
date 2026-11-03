# PV-896：ResNet50 + Cross-Matching 主基线

本基线对应 Wang 等人 SIGSPATIAL 2023 的 SCCNet **Cross-Matching 单分支**。方法依据为论文第 4.1 节及表 9 的第一行；开源结构参考 [linhanwang/SCCNet](https://github.com/linhanwang/SCCNet) 的提交 `741cbade237c803836c2b4192bfa91441964f461`。这里独立实现了所需结构并适配 PV-896，没有复制仓库文件。它不含 Self-Matching、谱分割、GeoRSCLIP 或 DINOv3-SAT，不能称为完整 SCCNet。

## 输入、输出与匹配张量

每次前向输入一张 Query RGB、一张 Support RGB 和对应 Support 二值掩膜，二者均为原始地面分辨率的 `256×256` patch。输出 `[B,2,256,256]` logits，通道 0 为背景，通道 1 为光伏。Query 真值仅用于训练损失或外部评估，绝不传给模型。

```text
Query RGB ──共享 ResNet50──┐
                         ├─ 逐层像素余弦相关 ─ 相关性金字塔 ─ 4D 空间压缩 ─ 2D 解码 ─ Query logits
Support RGB ─共享 ResNet50──┘       ▲
                                  Support 掩膜筛选特征
```

使用 ResNet50 的 `layer2`、`layer3`、`layer4` 各 bottleneck 最终 ReLU 前的 4、6、3 张特征图。模型按深到浅排序。以 `B=1`、`256×256` 输入为例，特征与相关张量实际形状为：

| 组 | Query/Support 单张特征 | 相关张量形状 |
| --- | --- | --- |
| `layer4` | `[1,2048,8,8]`，3 张 | `[1,3,8,8,8,8]` |
| `layer3` | `[1,1024,16,16]`，6 张 | `[1,6,16,16,16,16]` |
| `layer2` | `[1,512,32,32]`，4 张 | `[1,4,32,32,32,32]` |

相关张量的维度顺序是 **batch、同尺度特征层、Query 高、Query 宽、Support 高、Support 宽**。论文所称的“4D 相关性”指最后四个空间轴，并非一个固定为四维的输入格式。每个元素是一个 Query 位置与一个经过掩膜筛选的 Support 位置之间的非负余弦相似度。`CenterPivotConv4d` 分别沿 Query 和 Support 空间轴卷积，不把原始 `896×896` 像素直接展开成稠密 Query–Support 矩阵。实现没有用 `expand` 或 `repeat` 复制相关张量。

## 数据与实验协议

- **任务**：已知光伏类别的支持图条件分割。训练和验证都使用光伏标注；`1-shot/5-shot` 表示推理时每张 Query 使用的 Support 数量，不代表全程仅使用 1/5 张标注图。不能把结果称为论文的“基类到新类”迁移成绩。
- **空间隔离**：复用项目的 `scene_groups`，共享源影像名、影像哈希或相交地理范围的场景合并分组。先分训练/开发验证组，再选择 Query。训练 Query 与 Support 分属不同组；开发验证 Query 的 Support 均来自训练组；5-shot 的五张 Support 来自五个不同的训练组。
- **裁剪**：原始 `896×896` RGB/Mask 保持原地面分辨率。训练时裁取 `256×256` Query 和 Support patch；正样本 patch 含真实光伏像素，负样本 patch 随机裁取。Support 裁块若在 stride 8 的特征网格上失去全部光伏像素，会在同一空间组重新选取，并记录实际 Support ID。验证时覆盖整张 `896×896` Query，用 16 个可能重叠的 `256×256` patch 推理后在重叠处平均；不整图缩小，也不构造 896 像素级全量稠密相关矩阵。
- **训练**：ImageNet 预训练 ResNet50，冻结并保持 eval；仅训练相关性压缩和解码器。按 Query 标签类别逆频率采样训练 episode，训练每次用一张 Support。优化目标为两类交叉熵，SGD 基础学习率 `9e-4`、动量 `0.9`、权重衰减 `5e-4`，poly 学习率衰减指数 `0.9`。遵循开源优化器，匹配头权重实际学习率为基础值的 10 倍，偏置为 20 倍；冻结骨干无参数更新。训练图和掩膜同步旋转/翻转。5-shot 按开源代码的逐 Support 硬预测投票，使用峰值归一化和 `0.4` 阈值。
- **评估**：开发验证按整张 Query 累计 TP/FP/FN，主指标为光伏前景 IoU，同时记录 F1、Precision、Recall、负样本误检。空真值且空预测的前景 IoU 为未定义，不记作 1。输出按类别和逐 Query 保留支持 ID 与像素计数。

本仓库的 PV-896 场景已用于此前开发，因此自动留出的验证组是**开发集**，不是独立测试集。严格的新地域泛化结果须另行锁定独立数据。现有标注是否全为陆上电站、是否包含屋顶或边界疑难样本，还需依据人工标注规则核验。

论文第 5.3 节文字提到骨干以更低学习率一起训练，但所检查的开源 `model/sccnet.py` 在前向特征提取时使用 `torch.no_grad()`，其 README 训练示例也设置 `--freeze True`。本实现按开源前向路径冻结骨干，并明确记录该差异。开源 `common/my_optim.py` 对匹配头实际使用基础学习率的 10/20 倍，本实现也保留此分组。官方 README 已说明原先的预训练 SCCNet 检查点丢失，因此不能宣称加载了作者训练权重。ResNet50 的 ImageNet 权重与本基线训练所得权重分别记录。

## 运行

先确认实际 PV-896 挂载根目录，其下应有 `manifest.csv` 与三个类别的 `images/`、`labels/`。使用本地/挂载的 torchvision ResNet50 ImageNet 权重，或显式允许 torchvision 下载：

```bash
python scripts/train_cross_matching_pv.py \
  --data-root /kaggle/input/ACTUAL_RESOURCE/PV-896 \
  --output-dir /kaggle/working/outputs/cross_matching_pilot \
  --download-imagenet --epochs 1 --train-query-limit 60 \
  --validation-query-limit 12 --eval-shots 1
```

正式运行使用新输出目录，去掉两个 Query 数量限制；可加 `--eval-shots 5` 测试 5-shot。ResNet50 本地权重使用 `--imagenet-weights /实际路径/resnet50-0676ba61.pth`，与 `--download-imagenet` 二选一。首版训练入口拒绝非 256 patch，避免无意在原始 896 尺度构造巨大的相关张量。此仓库代码提交不包含数据、预训练权重或训练检查点。

`forward_profiles.jsonl` 对每个训练 step 记录 Query/Support backbone 各层形状、三组相关张量形状和元素数、forward 耗时、完整训练 step 耗时与 CUDA 峰值已分配显存；CPU 时显存字段为 `null`。`history.json` 还保存每张开发验证 Query 首个 tile 的前向形状、显存与耗时。`config.json`、`spatial_split.json`、`best_checkpoint.pt` 和 `completion.json` 记录复现所需配置、组划分与检查点。该结构为后续接入 DINOv3-SAT 留出 `encoder` 接口，但新编码器的通道、层数与 stride 必须显式适配，不能假定与 ResNet50 完全相同。

合成测试：

```bash
python -m unittest discover -s tests -p 'test_cross_matching.py' -v
```
