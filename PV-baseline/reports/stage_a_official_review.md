# Stage A：PFENet 原论文与官方实现核验

核验来源：论文 *Prior Guided Feature Enrichment Network for Few-Shot Segmentation*（arXiv:2008.01449，TPAMI 2022）及官方仓库 `JIA-Lab-research/PFENet` 提交 `a5056754d232829644e84d36b1e1d30818bd4c4f`。源码暂放仓库外的本地审阅目录，尚未复制或改造模型代码。

## 原论文机制与官方代码

| 核查项 | 论文设计 | 官方实现与位置 |
|---|---|---|
| 任务 | PASCAL-5i / COCO 按类别分折，训练类与测试类不重合；1/5-shot episodic 分割 | `util/dataset.py:77-246` 每次按 query 类别选支持图；训练和验证用不同类别集合；只排除同一图像路径，不核查原图或地理空间 |
| Backbone | ImageNet 预训练 VGG-16、ResNet-50/101；冻结骨干 | `model/PFENet.py:65-93` 支持 VGG 或深 stem、膨胀卷积 ResNet；`train.py:103-112` 冻结 layer0–4，只优化后续投影、FEM、分类头 |
| 特征层 | ResNet conv3_x 与 conv4_x 的末层作为中层，conv5_x 末层作为高层 | `model/PFENet.py:184-218` 将 layer2 + layer3 拼接送 `down_query/down_supp`，layer4 负责 prior；支持特征经 Mask 加权 GAP |
| Prior Mask | 冻结高层特征的 query–support 稠密余弦相似度，query 每位置取支持位置最大相似度，再逐图 min-max 归一化 | `model/PFENet.py:201-248` 支持 Mask 先乘到 layer3 再送 layer4，随后在 layer4 输出再次乘 Mask；每 shot 计算最大余弦图，1/5-shot 多图平均；prior 在骨干 `no_grad` 内生成，无可训练 prior 头 |
| FEM | 多尺度上融合 query、中层支持原型、prior；细到粗的跨尺度传递和残差细化；各尺度监督 | `model/PFENet.py:119-175,256-286` 四个 pyramid bin `[60,30,15,8]`，`init_merge` 拼接 256+256+1，`alpha_conv` 与 `beta_conv` 跨尺度细化，`inner_cls` 监督，`res1/res2/cls` 合并输出 |
| 损失与输出 | 最终交叉熵加各尺度交叉熵平均，辅助系数 1.0 | `model/PFENet.py:289-304` 和 `train.py:264-272` 为二类 CE、四路辅助 CE 平均，输出原始 logits 在 eval；train 时返回 argmax、主损失和辅助损失 |
| 训练 | PASCAL-5i 用 SGD，lr 0.0025，batch 4，200 epochs，poly power 0.9，动量 0.9，weight decay 1e-4；473×473 随机裁剪 | `config/pascal/pascal_split0_resnet50.yaml` 符合上述设置，训练有缩放 0.9–1.1、旋转 ±10°、模糊、翻转，ImageNet mean/std，输入需满足 `(size-1)%8=0` |

## 论文与代码需要区分的细节

1. 论文说训练自由 prior 来自冻结高层特征；代码确实将骨干前向置于 `torch.no_grad()`，但支持高层特征是 **Mask 作用于 layer3 后再次经 layer4** 得到，再乘下采样 Mask，不是简单地对完整 layer4 特征做一次后遮罩。
2. 论文描述支持特征与 prior 的多尺度融合；代码 5-shot 对中层 Mask-GAP 原型和 prior 分别简单平均，并未实现五路注意力或显式背景原型。
3. 论文在验证时回到原图尺寸；官方 `test.py` 用 bilinear 上采样 logits 并计算二类 IoU。论文的 class mIoU 是多个测试类别平均；官方打印的 FB-IoU 是前景/背景两类平均。这两个定义在只有 PV 一个前景类的本任务中不能照搬为同一个数。
4. 官方代码使用 `forward` 参数默认值处的 `.cuda()`、旧式 PyTorch 1.4 API、`torch.load` 原始权重路径与 Pascal 类别采样。需要兼容性适配才能在当前 Kaggle 环境、PV 二值 Mask 与纯负 Query 上运行；不能删掉 prior/FEM/辅助头后仍称 PFENet。
5. 官方 ResNet50 初始化文件是 `./initmodel/resnet50_v2.pth`（`model/resnet.py:194-205`），不包含在仓库中；本地尚未发现该文件。没有核实权重与许可证前，不把随机初始化或 torchvision 普通 ResNet50 权重当作官方原始权重。

## PV-896 的必要适配方案（尚未实施）

- 固定目标类为 PV，Mask 的 255 必须先映射为前景 1；官方数据集将 255 视为 ignore，原样套用会把所有 PV 像素忽略。文件夹类别只用于采样和分组统计，实际目标从 Mask 决定。
- 保留官方 ResNet50 深 stem、膨胀率、layer2/3 中层特征、layer4 prior、四尺度 FEM、最终头和 CE＋四尺度辅助 CE。训练支撑集与查询集按原图/场景/空间组约束；纯负 Query 参与训练和评估，支持图必须含 PV，且 1/5-shot 的支持图彼此不能重叠。
- 首轮忠实适配使用 473×473 输入，RGB 双线性缩放、Mask 最近邻缩放，预测 logits 双线性回到 896×896 后计算指标和保存概率图。该缩放会损失细小 PV，作为明确的适配代价记录。以后是否改为 897×897 或分块，需要单独显存验证，不能与初版混成同一基线。
- 保留官方 SGD、poly、CE 权重作为初始设定；小规模 smoke 先检查 1-shot，再考虑 5-shot 的显存和训练预算。固定 PV 类训练与测试不具备原论文的“未见类别”设置，只能报告为 **PFENet 架构在 PV 同类跨场景任务上的适配基线**。

Stage A 未训练、未测试，也未获得 PFENet 可用 checkpoint。
