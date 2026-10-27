# SSPNet-DINOv3 光伏适配实现

本实现根据 Zhong 与 Zhao 的 SSPNet 论文（DOI: 10.1109/JSTARS.2026.3745987）重建方法，使用用户已保存在 Kaggle 的 DINOv3-SAT 替换 DINOv2。没有找到可核验的作者官方代码，因此这是**论文方法重建与单类别光伏适配**，不能称为官方逐行复现，也不能声称复现了论文的指标。

## 已实现的完整路径

1. ImageNet 初始化的 ResNet-50 / output stride 8 / PSPNet，在训练空间分组内进行二类有监督分割预训练。
2. 冻结 PSPNet 与 DINOv3，用训练图及其真实掩膜构造 1-shot 或 5-shot 支持集；查询图可包含无光伏的负样本。
3. DMSF：最后一个 DINO patch 特征 → 共享 Linear/LN/GELU 投影 → 支持前景原型产生空间 CAM；GeoRSCLIP 固定文本向量 → 三层跨模态映射 → 隐层投影产生语义 CAM；两个 CAM 分别进行有界残差修正；单个全局标量 sigmoid(beta) 融合，beta 初始为 0.7。
4. 在 CNN block4 与 PSP 分类器前的 512 通道特征之间，按论文 Eq. 10–12 选择前景与前景优势相关图。
5. 组装查询／支持特征，DCMA 进行 [1,1.1] 倍 CAM 调制、8 头有前景掩膜的 cross-attention、残差/LN/FFN；辅助头输出 [-5,5] 的伪掩膜 logits。
6. ASPP + 残差卷积 + meta 二类预测，经 Gram 支持权重及基学习器融合得到最终预测。
7. 损失为 `CE_final + Dice_meta + 0.2 BCE_pseudo + 0.04 Dice_spatial_CAM`。辅助伪掩膜只参与训练损失，不融合进最终预测。

推理接口严格为 `model(query_rgb, support_rgb, support_masks)`，不接收查询真值。评估标签只在模型外计算损失与指标。

## 与原论文的差异及原因

| 项目 | 当前实现／含义 |
| --- | --- |
| 任务 | 单个已知前景“光伏”与背景；文件夹的正／负／困难正样本不是三个语义类别 |
| 监督 | 两个训练阶段均使用真实像素掩膜；不是弱监督，不是仅总计 1/5 张标注 |
| 基类／新类 | 没有多类别 base→novel 不相交划分；PSPNet 已见过光伏标签 |
| 基学习器融合 | 按背景／光伏分别合并对应的 base/meta 响应；不能把同一个光伏基类当作“其他基类”来压制光伏前景 |
| DINO | ViT-L/16，最后一个 block 23，1024 维；原论文 ViT-B/14。SAT493M 是预训练标记，实际骨干约 303M 参数 |
| CNN stem | torchvision ResNet-50 的标准 7×7 stem；BAM 参考代码使用自定义深 stem。论文未给出足够实现细节，此处明确采用可维护的 torchvision 版本 |
| PSPNet | PPM bins 1/2/3/6，分类头 512 维；batch=1 时冻结 BN 运行统计，BN 仿射参数在第一阶段仍可学习 |
| 输入 | 原图严格校验 RGB BMP 896²、掩膜 0/255；首次实验 resize 到 512²，标签 nearest，不扩充或虚构小目标 |
| 归一化 | DINO SAT mean [.430,.411,.296] / std [.213,.156,.143]；CNN 使用 ImageNet 参数；二者均从同一份 [0,1] RGB 独立计算 |
| 文本 | 固定光伏遥感描述，见 `PV_PROMPT`；新增类别 prompt，不是论文原有类别文本 |
| 5-shot | DMSF 空间 CAM 等权平均，层次相关图按论文等权平均；CNN 原型用 Gram 权重。具体多 shot 细节无官方代码可核实 |
| Dice | meta 两类 Dice 等权平均，前景权重默认 1；不自动迁移 LoveDA 类别的 2 倍启发式权重 |
| 训练 | SGD .005 / .9 / 1e-4，poly .9，默认 25 epoch；batch=1 累积 4 次，不能宣称与论文 batch=4 完全等价 |
| 增强 | 同步翻转／90°旋转；尚未加入论文中的随机裁剪、颜色增强，避免初版裁剪消除微小支持前景 |
| attention | PyTorch SDPA，按查询 token 分块，使用同一 -1000 掩膜公式，不显式保存全尺寸 attention 矩阵 |

三层文本映射宽度、CAM 修正隐藏宽度（16）、FFN 4 倍扩展、dropout 等为透明的重建选择，不能假装是已核验的原作者设置。

## 层次选择的论文歧义

论文先对两层相关图分别 min-max，再比较各自最大值。非恒定图的最大值通常都接近 1，路由可能受 epsilon 和浮点差异影响。默认 `--routing literal` 保留该顺序，并记录 `near_tie`、`base_selected`。`--routing raw_max` 是显式实验选项，改为比较归一化前响应；不能将其指标归为严格原方法。

伪掩膜头最后一层零初始化，因此**第一次反向传播**的该项损失只能先更新末层，上游梯度在末层更新后才出现。单元测试检验了第二步的语义投影梯度；不把计算图可达性当成准确率提升的证明。

## 数据隔离

复用 `pvseg.spatial`：共享源图名、图像 hash 或空间相交的场景合并后，先分训练／验证组。`--pilot` 重用旧的最多 60 对开发集；默认使用同一空间分组下的全部样本。两阶段必须使用完全相同的 ID 划分 hash。

每个 episode 的支持图必须来自训练侧、与查询分属不同空间组、ID 与 hash 不同、足迹不相交；5 个支持之间也不能相交或重复。支持前景在模型输入与 DINO patch 网格消失时跳过该候选，若剩余不足则明确报错。负查询允许空掩膜，支持图不允许无前景。

验证支持选择由 seed 固定并写出每张查询的支持 ID；训练每 epoch 更新采样。现有 16 个来源场景已用于此前开发，**该验证集不是新的独立测试集**，正式研究需要另留未参与开发的区域／来源。

## 权重与运行位置

代码与配置保存在 GitHub。原始图像、预训练／训练权重、运行产物仅留 Kaggle；不下载到本地仓库。

- SAT：`/kaggle/input/notebook5f7aef8a4f/verified_sat/model.safetensors` 及相邻 `config.json`，挂载原 notebook v4。
- SAT SHA256：`8d18337f766102f951ac0ca543a22536f58caeb6b6c33cfef73326f866791286`。
- GeoRSCLIP：`Zilun/GeoRSCLIP` revision `4920188e6eba4e711ef9848cfd7cb77e874ee33f`，仅 `ckpt/RS5M_ViT-B-32.pt`。
- Geo SHA256：`129bafaa6a097b8be52e2babf27d24f0a934dae919201e538dc698611bd1ea01`。
- 只缓存冻结文本编码器的原始 512 维向量，学习投影每步重新计算。完整 CLIP 可在缓存后释放；缓存保存 prompt 和源权重 hash。
- GeoRSCLIP 使用其 OpenAI CLIP 初始化对应的 QuickGELU；只核对权重键名不足以发现 GELU／QuickGELU 激活不一致，因此显式指定并记录。
- PSPNet 第一阶段 checkpoint 包括全模型；第二阶段 checkpoint 包括冻结 CNN、原始文本向量、全部可训练模块、优化器、AMP scaler、随机状态与配置，省略已有 hash 校验的 SAT 权重。

## Kaggle 中的命令

先运行 `python scripts/kaggle_sspnet.py` 完成下载／缓存、真实权重工程检查。该入口只做第一、二阶段各 2 次更新和 5-shot 推理，不能作为准确率实验。

正式训练使用新输出目录，首次从冻结文本缓存与已挂载 SAT 权重开始：

```bash
python scripts/sspnet.py base --data-root /kaggle/input --output /kaggle/working/sspnet_base --epochs 25 --size 512 --batch-size 1 --accumulation 4

python scripts/sspnet.py episodic --data-root /kaggle/input --output /kaggle/working/sspnet_1shot --base-checkpoint /kaggle/working/sspnet_base/last.pt --weights /kaggle/input/notebook5f7aef8a4f/verified_sat/model.safetensors --text-cache /kaggle/working/sspnet_assets/georsclip_pv_text.pt --shots 1 --epochs 25
```

5-shot 使用 `--shots 5` 和独立输出目录重新训练其 reweight 模块。检查点推理评估使用相同结构、划分与依赖：

```bash
python scripts/sspnet.py evaluate --data-root /kaggle/input --output /kaggle/working/sspnet_eval --base-checkpoint /kaggle/working/sspnet_base/last.pt --weights /kaggle/input/notebook5f7aef8a4f/verified_sat/model.safetensors --text-cache /kaggle/working/sspnet_assets/georsclip_pv_text.pt --shots 1 --resume /kaggle/working/sspnet_1shot/last.pt
```

输出包含逐 epoch 四项训练损失、整体验证与分组 IoU/F1/误报/漏检指标、支持 ID、路由统计、依赖版本、划分与检查点。`--resume` 支持 epoch 边界恢复，正式训练要求保持 size／shots／划分／batch／累积／总 epoch 不变；不支持将半个 epoch 当成完整 epoch 恢复。首版保存 `last.pt`，不自动挑选最优模型或进行超参数搜索。

梯度范数截断为 10，也是首版工程设置。Kaggle API 可能把带 `/4` 的 notebook 输入解析为较新归档版本；若归档没有 SAT 文件，检查入口会只在 Kaggle 内从固定 HF revision 恢复相同 SHA256 的权重，并保存恢复记录与实际文件供后续挂载。

`--max-steps` 和 `--allow-untrained-base` 明确标记工程检查。仅运行两步的 PSPNet 会被标记 engineering_only，不能在未显式允许工程模式时冒充合格的第一阶段模型。

## 验证与引用

本地验证：`python -m unittest discover -s tests -v`，包含独立于预训练权重的数值、梯度、空掩膜、隔离、1/5-shot、checkpoint 测试。真实编码器 GPU 验证以 Kaggle 输出为准。

Gram discrepancy、排序后 k-shot reweight 和 learner merge 的参考来自 [chunbolang/BAM](https://github.com/chunbolang/BAM) 的 `model/BAM.py` 与 `model/PSPNet.py`，保留 MIT 许可于 `third_party/BAM-LICENSE`。未引用其结果作为本实现结果。GeoRSCLIP 来自 [RS5M](https://github.com/om-ai-lab/RS5M)；各上游权重许可仍适用。
