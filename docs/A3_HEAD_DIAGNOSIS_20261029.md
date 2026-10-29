# A3 下游全背景诊断

本轮只做训练侧定位，不读取校准或查询影像。复用 A3 learned v2 的基础网络，固定原训练划分的 4 张正图与首张负图，支持取该查询允许的首张非重叠正图，原生 896×896、无增强。

先检查旧 selected.pt 的 AMP/FP32 输出和各损失的梯度，再从相同随机初始化、相同 RNG、相同 episode 序列各训练 120 次。原结构、四项损失、AdamW lr=1e-4、weight decay=1e-4、梯度裁剪 10 均保持一致；只有混合精度与 FP32 不同。两组均启用相同 Dropout，评测时关闭。

记录中间特征尺度、base/meta/final 的逐图混淆像素、前景/背景概率、饱和比例、前景面积、logit 差，以及各模块梯度范数。独立记录 CE/meta Dice/pseudo BCE/spatial Dice 对预测头、解码器、注意力和 DMSF 的梯度。FP32 的小于半精度误差不等于解释收敛失败，须结合实际输出判断。

入口 `scripts/kaggle_a3_diagnosis.py`。诊断过拟合结果不是验证精度；若修改归一化、头初始化或损失，须作为适配方案单独记录并保留本轮原模型对照。门槛沿用工程检查：固定训练样本 final/meta 前景 micro-IoU 至少 0.60、每张正图有 TP、损失下降，同时报告负图误报。

## 原模型诊断结果

私有 [Kaggle diagnosis v1](https://www.kaggle.com/code/zh0416/pv-a3-head-diagnosis) 已完成，源码 `d399b38f64aa3aec98e1990ffb75ca6aceae5fa3`，Tesla T4 / Torch 2.11。两臂各完成 120 次更新，36 项检查通过；完整小报告见 [原模型证据](experiments/a3_head_diagnosis_20261029/original/summary.json)。

| 训练侧检查 | AMP | FP32 |
| --- | ---: | ---: |
| 冻结 base 前景 micro-IoU | 72.68% | 72.68% |
| 原初始化 meta/final IoU | 3.25% | 3.25% |
| 更新 1 次的 meta/final IoU | 0 | 0 |
| 更新 120 次的 meta/final IoU | 0 | 0 |
| 平均诊断损失（初始→120） | 1.52408→0.80204 | 1.52408→0.80251 |

以固定正图 `pv001179` 的 FP32 路径为例：初始 decoder RMS=0.165、meta logits RMS=0.0617；step20 分别增至 6.616、48.233，前景减背景 logit 范围为 [-104.77,-16.06]，所有像素前景概率低于 1e-6，预测头总梯度 RMS=1.26e-11。step120 meta logits RMS=160.03，margin 范围 [-351.53,-46.22]，预测头与 decoder 的梯度 RMS 均为 0。旧 selected.pt 同样饱和，改用 FP32 推理不能恢复。

这说明当前失效发生在 meta 分支、早于最终 base/meta 合并；不能归因于 AMP。meta softmax 饱和后，经过该 softmax 的 final CE 与 meta Dice 都无法提供足够纠错梯度。CNN 特征初始尺度有限，没有发现输入 NaN；ASPP/头的尺度在训练中快速增大。注意力 Q 的梯度也很小，仍是待研究项。以上定位的是观察到的失效机制，不证明唯一根因，更不能据此否定论文方法。

## 最小修正的预先指定方案

保留原模型、初始化、训练数据、episode、Dropout、优化器及四项原损失，只额外加入系数 1 的 **balanced raw-meta CE**：对每个 episode 分别计算前景/背景像素的平均交叉熵，再对存在的类别等权平均；纯负图只计算背景项。直接使用 meta 原始 logits，避开饱和 softmax 对该项梯度的衰减。它属于光伏适配改进，不能标成原文损失。

本地新增回归检查：极度错误饱和时仍有前景纠错梯度；复制背景像素不会稀释前景权重；纯负图结果有限且等于普通背景 CE。全部 39 项检查通过。原 `objective()` 默认权重为 0，默认行为保持四项损失。

下一轮只在相同 5 张训练图上运行 FP32，预先上限 240 次更新；第 120 步与原模型等更新数比较，第 240 步只作为延长诊断，不能把提升全部归因于损失。先核对 step0 与原 FP32 的逐图输出统计及 episode 序列。只使用训练侧指标判断工程门槛；不读取校准/查询图选方案。运行参数：`--meta-ce-weight 1 --steps 240 --output /kaggle/working/a3_head_repair`。

原始数据、冻结特征、权重和 checkpoint 留在 Kaggle。本地和 GitHub 只保存小型报告、源码及派生预览。
