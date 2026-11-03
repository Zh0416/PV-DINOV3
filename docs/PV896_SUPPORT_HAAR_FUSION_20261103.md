# 一张支持图训练动态融合与 Haar 校正

前一轮 `PV896_SUPPORT_ONLY_20261103.md` 排除了动态融合和 Haar，只训练 DINOv3-SAT 上的线性头。那个实验不能用于判断这两个模块是否能从支持图训练。本轮恢复模块结构，从零初始化任务权重，在单张 `pv001183` 影像与 mask 上训练。

## 训练与推理

- 冻结通用预训练 DINOv3-SAT 和 GeoRSCLIP；不加载任何既有光伏任务 checkpoint、旧阈值、亮度参考或语义提示参数。
- 支持 mask 的光伏/背景 patch 分别生成 DINO 和 GeoRSCLIP 前景、背景原型，得到两路 CAM。RGB 在每个 16×16 patch 计算六路双尺度 Haar 方向能量。
- 动态融合模块从中性 `α=0.5` 开始，学习逐位置组合两路 CAM 和有界残差；Haar 校正模块用六路 Haar 特征学习额外分数修正。两组模块均从零初始化，使用支持 mask 的平衡 BCE＋Dice 做反向传播。
- 四组交错 patch 依次留出：每组模型只在其余三组的 mask 上训练，留出预测汇总后仅用支持 mask 选择阈值。每组模型 80 步，AdamW 学习率 0.002。查询图只做前向传播，四组模型预测平均；整图存在性由分割 mask 是否非空给出。
- 与前轮使用同一张支持图和同一批 300 张查询图。之前已查看过这 300 张的标签与结果，因此本轮成绩是**重复使用测试集的探索性比较**，不能作为全新独立泛化测试；本轮未用其标签修改训练参数或阈值。

源码：`pvseg/support_haar_fusion.py`；Kaggle 入口：`scripts/kaggle_pv896_support_haar_fusion.py`；封装入口：`scripts/prepare_pv896_support_haar_kernel.py`。私有运行：<https://www.kaggle.com/code/zh0416/pv-896-300-support-trained-haar-fusion/versions/1>。
