# PV-baseline：PFENet 陆上光伏基线

本目录当前完成 **Stage A 官方实现核验**与 **Stage B 数据审计**。模型迁移、Smoke Test、1/5-shot 训练与测试尚未开始；不得把现有其他模型的指标写作 PFENet 成绩。

- [官方 PFENet 与论文核验](reports/stage_a_official_review.md)
- [PV-896 数据审计与划分](reports/stage_b_data_audit.md)
- [机器可读数据审计](reports/data_audit.json)
- [固定空间划分和样本 ID](splits/pv896_seed321.json)
- [路径与划分配置](config.json)

重放数据审计：在仓库根目录运行 `python PV-baseline/audit_data.py`。脚本会实际读取全部 3,668 对 BMP 并计算哈希，约 11.8 GB 数据，运行时间取决于磁盘速度。当前没有模型 checkpoint、训练日志或可视化结果。

## 后续实施顺序

1. **Stage C：适配和 smoke。** 引入官方 ResNet50 PFENet，保持高层先验、四尺度 FEM、原预测头及主/辅助交叉熵。实现从固定划分生成可追溯的 1/5-shot episode；Mask 的 255 PV 映射为 1，全背景 Query 保留。先用 473×473 输入验证前向、损失、反向、优化器、checkpoint 恢复和 896×896 推理可视化。
2. **Stage D：小规模试验。** 固定少量训练 episode 和验证 episode，只运行 1-shot 少量 epoch，记录 loss、Mask 形态、显存和速度。验证集用于选择 checkpoint，测试集不参与模型或阈值选择。
3. **Stage E：正式实验。** 在同一空间划分上运行 1-shot、5-shot 和多种子实验，保存固定测试 episode、预测概率、逐图可视化、混淆计数与负样本误检统计；另找未接触的地理场景检验外推能力。

训练开始前还需取得并核验官方 `resnet50_v2.pth` 权重，或明确将替代权重作为不同初始化实验。473×473 缩放与正式测试区域是否保留现有开发划分，也是进入 Stage C 前应确认的方案选择。Stage C 成功后才能展开 Stage D/E。
