# SSPNet-DINOv3 光伏适配：真实权重运行记录

2026-10-28（北京时间），Kaggle 私有任务第 3 版运行完成，状态 **COMPLETE**，日志确认 `SSPNET REAL-WEIGHT SMOKE PASSED`。PSPNet 第一阶段、SSPNet 第二阶段各完成 **2 次优化器更新**，并完成真实 5-shot 推理。这是工程检查，尚未获得充分训练的分割模型。

## 可追溯版本

- 代码仓库：[Zh0416/PV-DINOV3](https://github.com/Zh0416/PV-DINOV3)。
- 分支：`codex/sspnet-dinov3-pv`。
- 实际运行源码：[`ddd7b88ee3b8d23443969935531a5cc81922ece1`](https://github.com/Zh0416/PV-DINOV3/commit/ddd7b88ee3b8d23443969935531a5cc81922ece1)。本报告随后的文档提交不改变该运行的源码版本。
- Kaggle：[SSPNet DINOv3 PV](https://www.kaggle.com/code/zh0416/sspnet-dinov3-pv)，在版本历史中选择第 3 版；CLI 标识为 `zh0416/sspnet-dinov3-pv/3`。
- 启动入口：`kaggle/sspnet_bootstrap.py` 固定完整 Git commit，执行 `scripts/kaggle_sspnet.py`。
- 方法重建、论文歧义与正式命令见 [实现说明](SSPNET_REPRODUCTION.md)。

## 实际检查范围

| 项目 | 实际执行与结果 |
| --- | --- |
| GPU | Kaggle 配置 Tesla T4 ×2，每张 15,636,037,632 bytes；程序使用 cuda:0，未做双卡并行 |
| 精度 | FP16 autocast + GradScaler；T4 不使用 BF16 |
| 输入 | 源 RGB / 掩膜 896×896；本轮缩放至 512×512，标签 nearest |
| 数据划分 | 固定 seed=42，空间分组先划分再抽样；48 训练对、12 验证对的开发集 |
| 行为测试 | Kaggle 中 21 项测试全部通过，耗时 5.909 秒 |
| Stage 1 | ImageNet ResNet-50 + PSPNet，2 个 batch、2 次更新；损失 0.569019 → 0.174447 |
| Stage 2 | 冻结第一阶段 CNN 与真实 SAT；1-shot、2 个 batch、2 次更新；总损失 1.505591 → 1.499538 |
| Checkpoint | 两阶段均保存并逐张量核对 checkpoint 内容；实际恢复能力另有行为测试覆盖 |
| 5-shot | 困难正样本、正样本、负样本各 1 个真实查询，每个有 5 张不同且不重叠的训练支持图；输出均为 `[1,2,512,512]`，损失有限 |
| 运行结束 | 总日志时间约 213 秒，包含依赖安装、权重准备、训练、推理与 Kaggle 导出 |

实际软件版本：Python 3.13.15、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、timm 1.0.29、open_clip_torch 3.2.0、NumPy 2.1.3、Pillow 13.0.0。两阶段划分 SHA256 均为 `ea8ccce20d3e0a2b136f9ae037e4f9958c0a1c271c0f7e1b24b76b3e9d02519b`。

| 过程 | 独立进程内 PyTorch 峰值 allocated | 峰值 reserved | 阶段耗时 |
| --- | --- | --- | --- |
| Stage 1 | 1,210,012,160 bytes（约 1.127 GiB） | 1,444,937,728 bytes | 17.81 秒 |
| Stage 2 | 1,891,646,464 bytes（约 1.762 GiB） | 2,006,974,464 bytes | 21.45 秒 |
| 5-shot 推理 | 1,985,012,736 bytes（约 1.849 GiB） | 2,090,860,544 bytes | 未单独计时 |

这些是本轮 batch=1、512² 的 allocator 统计，不能直接外推 896²、较大 batch 或长时间训练；也不等于整张 GPU 的总占用。

5-shot 使用两步 1-shot 检查点的共享参数，其独立 reweight MLP **没有训练**。这只检查 5-shot 数据与张量路径，正式 5-shot 实验必须单独训练。

单元测试覆盖掩膜 attention 的背景不变性、空前景回退、查询分块一致性、k-shot 置换等变、独立归一化、可学习投影梯度、第二步伪掩膜梯度、冻结编码器、空查询目标、层次路由近似平局、支持集空间隔离和检查点依赖验证。

## 预训练权重与修正

实际 SHA256 在 Kaggle 内计算并校验；数据、权重与模型输出留在 Kaggle，本地仅取回 JSON / TXT 报告和日志。

| 资产 | 固定来源 | SHA256 |
| --- | --- | --- |
| DINOv3-SAT | `timm/vit_large_patch16_dinov3.sat493m`，revision `ce148bd5ed3de1eb75ebdccafcb2c280bb3e8f7c` | `8d18337f766102f951ac0ca543a22536f58caeb6b6c33cfef73326f866791286` |
| GeoRSCLIP | `Zilun/GeoRSCLIP`，revision `4920188e6eba4e711ef9848cfd7cb77e874ee33f`，`ckpt/RS5M_ViT-B-32.pt` | `129bafaa6a097b8be52e2babf27d24f0a934dae919201e538dc698611bd1ea01` |

Kaggle API 将原 notebook 输入解析到较新的归档输出，归档中没有 SAT 文件。入口已增加固定 revision 的恢复路径，仅在 Kaggle 内下载相同字节并校验上述 hash，保存实际 `model.safetensors` 与相邻 `config.json`，供后续挂载。

GeoRSCLIP 的 OpenAI CLIP 初始化需要 **QuickGELU**。第 3 版显式指定该激活函数，并在原始文本缓存中记录；加载器拒绝未注明 QuickGELU 的旧缓存。第 2 版虽跑通流程，但其文本缓存不作为最终验证资产使用。

## 结果的含义与限制

两步训练后的短验证仅评估前 2 张查询，前景 IoU 为 **0**；不能称为有效精度，更不能与此前 SegDINO 的充分训练轮次或论文结果作性能比较。损失有限且能够更新参数只说明工程路径成立。

本任务仍是**使用真实像素掩膜的已知类别光伏分割适配**。PSPNet 第一阶段见过光伏标签，没有原论文的多类别 base→novel 不相交划分，1-shot / 5-shot 指每个 episode 的支持数量。三类文件夹是采样类别，并非三个语义类别，也不是弱监督标签。

论文层次选择公式先分别归一化再比较最大值，实测记录了近似平局。默认保持该顺序；`raw_max` 只作为另一个明确标记的消融配置。

本次没有运行全数据集 25 epoch 训练，也没有测量 896×896 原尺寸。当前 16 个来源场景曾参与开发，验证集不是独立测试集。正式比较应固定标注预算、划分、输入尺寸、训练轮次及支持采样，并补充未参与开发的区域。

## 保存位置与下一阶段

Kaggle 第 3 版输出：

```text
/kaggle/working/sspnet_smoke/
  completion.json, environment.json, asset_manifest.json, tests.txt
  sat_recovery.json, five_shot.json
  base/{last.pt,completion.json,config.json,history.json,split.json}
  episodic/{last.pt,completion.json,config.json,history.json,split.json}
/kaggle/working/sspnet_assets/
  verified_sat/{model.safetensors,config.json}
  RS5M_ViT-B-32.pt, georsclip_pv_text.pt
```

阶段检查点都带有 `engineering_only` 标记。正式训练需要新建输出目录，先充分训练 PSPNet，再冻结它训练 episodic 模块；不能直接把两步 PSPNet 当作合格的第一阶段预训练模型。正式 1-shot 与 5-shot 使用各自训练的检查点。

首次充分训练前建议先完成小规模收敛实验，确认前景能够被学习，再扩展训练规模，随后开展空间先验、语义先验、DCMA 和融合策略的消融。具体命令已写在实现说明中，本次任务到工程验证通过为止。

本地取回的第 3 版报告在 `E:/PVdataset/.research/sspnet-20261027/v3-reports/sspnet_smoke/`。其中 `completion.json` 的 SHA256 为 `68c31b102109e32b2f667250007ee63a672f098aa5731653db6dd82699469a77`，`asset_manifest.json` 为 `7ea85185729ab74bb3725f44ee3f22a7b68dffc2d84730e2b033d3a62a72f680`。二者分别确认任务通过、权重来源与 QuickGELU 缓存配置。
