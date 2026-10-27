# PV-896 错误复核、模型归档与独立评估准备（2026-10-27）

## 1. 完成状态与执行边界

**已验证：本轮已完成并停止，训练步数为 0。** 只执行产物读取、checkpoint 检查、固定模型重新推理、错误可视化、归档和空间元数据复核。没有修改原始 Mask、调整 0.5 阈值、更换 backbone/decoder、下载大型权重或扩大训练。

| 项目 | 状态 | 实际结果 |
| --- | --- | --- |
| 上一轮小规模试验 | 已验证 | Notebook v4 完成 48 训练 + 12 验证、3 epoch；本轮读取其保存输出 |
| 原 live working 目录 | 已验证 | 新会话开始时 `/kaggle/working/PV-DINOV3/outputs/pilot/` 不存在，不能称原会话文件系统仍在 |
| 从保存输出恢复 | 已验证 | 挂载明确的 v4 输出；未重新训练 |
| checkpoint 可读及归档 | 已验证 | 原文件 `torch.load(weights_only=True)` 成功；复制 SHA256 一致 |
| 固定模型复推理 | 已验证 | 12 张验证图预测与原始保存结果逐像素一致，逐图 TP/FP/FN 一致，参数未变化 |
| Kaggle 保存版本 | 已验证 | v6 状态 COMPLETE，归档 162 个文件列入保存输出；两个 checkpoint 的保存对象均 HTTP 200 且长度匹配 |
| 新会话重挂载 v6 并重载归档 | 未验证 | 本轮在 v6 会话内重读导出权重；未再启动一个会话重挂载 v6 |
| 永久 Kaggle Dataset/Model | 未验证 | 本轮未创建；当前依赖明确版本的 Notebook 保存输出 |
| 错误真实原因、源标注验收标准 | 待人工确认 | 图像和计数支持诊断方向，不能据此确认漏标或重标 |
| 可靠跨地域泛化精度 | 未验证 | 当前验证只有两个场景且参与过模型选择与复核 |

Notebook：[zh0416/notebook5f7aef8a4f](https://www.kaggle.com/code/zh0416/notebook5f7aef8a4f)。v5 是第一次只推理复核，因 JSON/torch 序列类型比较失败；v6 修复后完成。没有再次训练。

## 2. 实际输入、文件存在性与保存证据

**已验证**，先列挂载目录后确认：

```text
数据根目录：
/kaggle/input/datasets/zh0416/pv-896/PV-896/
  正样本/{images,labels}/
  负样本/{images,labels}/
  困难正样本/{images,labels}/

v4 保存的 pilot 产物，挂载为只读输入：
/kaggle/input/notebook5f7aef8a4f/PV-DINOV3/outputs/pilot/

v4 保存的固定 SAT 权重及相邻 config.json：
/kaggle/input/notebook5f7aef8a4f/verified_sat/model.safetensors

v6 创建并保存的归档：
/kaggle/working/PV-DINOV3/outputs/archive/pilot_baseline/
```

下表路径均相对于上述 **v4 pilot 输入根目录**。时间为实际挂载文件 mtime（UTC，2026-10-27），可能反映平台导出/挂载时间，不冒充原训练写入时间。JSON、CSV、PNG 均实际解析/打开；checkpoint 另经 torch 实际加载。

| 文件 | 字节数 | mtime UTC | 可读状态 |
| --- | ---: | --- | --- |
| best_decoder.pt | 40,900,283 | 00:22:21.648 | 已验证 |
| config.json | 131,840 | 00:22:21.310 | 已验证 |
| best_checkpoint_metadata.json | 16,839 | 00:22:21.085 | 已验证 |
| history.json | 173,262 | 00:22:21.714 | 已验证 |
| epoch_metrics.csv | 477 | 00:22:21.714 | 已验证 |
| train_manifest.csv | 52,886 | 00:22:21.731 | 已验证 |
| validation_manifest.csv | 13,475 | 00:22:21.725 | 已验证 |
| spatial_split.json | 2,723 | 00:22:21.737 | 已验证 |
| selected_pixel_audit.json | 20,325 | 00:22:21.714 | 已验证 |
| label_source_report.json | 1,335 | 00:22:21.714 | 已验证 |
| best_predictions/pv003283_prediction.png | 1,894 | 00:22:21.191 | 已验证 |
| best_predictions/pv003283_overlay.png | 121,250 | 00:22:21.197 | 已验证 |

114 个原产物的完整路径、字节数、mtime、SHA256 和读取状态保存于归档 `source_file_inventory.json`，包括逐 epoch 和最佳预测。它记录历史 60 对的 `selected_pixel_audit.json`；此前首次 12 对的独立 `sample_audit.csv` 不在该 pilot 目录中，本轮未声称重新取得该 CSV。全量 3668 对只复查清单/空间元数据，本轮实际像素推理限于原验证的 12 对。

**已验证：v6 完成 Save Version 保存输出。** CLI 保存文件清单含 162 个归档文件；另对两个 checkpoint 保存对象仅请求响应头、不下载正文：

| 对象 | HTTP | Content-Length | 保存对象 Last-Modified UTC |
| --- | ---: | ---: | --- |
| best_decoder.pt | 200 | 40,900,283 | 2026-10-27 02:31:11 |
| inference_decoder.pt | 200 | 13,600,904 | 2026-10-27 02:31:12 |

这些是平台保存对象时间，与挂载文件 mtime 含义不同。完整本地证据为 `E:\PVdataset\.research\pv896-review-20261027\saved_output_verification.json`，未记录签名下载 URL、令牌或权重正文。后续使用 Kaggle Version history 选择 **v6**，检查 Output，再把这个明确历史版本作为输入挂载；不得假定新会话 working 目录自动保留。当前没有另建永久 Dataset/Model。

## 3. 固定代码、模型及依赖

以下固定项均为**已验证**：

- 训练代码提交：`4cf2aef6fe490a04be36feca0c84efff8594d06f`。
- 本轮 v6 实际执行提交：`92b2d5e14a36d0e57716b4af9e3f8cbf0bacf886`。本报告为其后补充的文档；不能将文档提交冒充实验执行提交。
- 推理前确认核心 model/data/metrics/runtime 文件与训练提交字节一致。训练模型结构及预处理未变化。
- Backbone：timm `vit_large_patch16_dinov3.sat493m`，实际类 `timm.models.eva.Eva`。
- 权重来源：`timm/vit_large_patch16_dinov3.sat493m`，HF revision `ce148bd5ed3de1eb75ebdccafcb2c280bb3e8f7c`。
- SAT SHA256：`8d18337f766102f951ac0ca543a22536f58caeb6b6c33cfef73326f866791286`；实际挂载文件 1,212,347,640 bytes，mtime UTC `2026-10-27T00:22:39.212000+00:00`。使用 v4 保存文件，无下载，无复制 SAT 到复核归档或本地/GitHub。
- 输入原始 896×896 RGB；mean `[0.430,0.411,0.296]`，std `[0.213,0.156,0.143]`；固定 `sigmoid(logits)>=0.5`，无测试增强和阈值搜索。
- 中间层索引 `[4,11,17,23]`、`norm=True`；四层均 `[1,1024,56,56]`，输出 `[1,1,896,896]`。
- 保留上游根目录 decoder：1×1 投影 96/192/384/768、scratch 128、首层 4 倍转置卷积、双线性对齐/拼接/1×1 预测，最终插值 `align_corners=True`。非论文完整原版复现。
- seed=42；所有参数冻结、eval、`torch.inference_mode()`；复核不创建 optimizer、不反向传播。
- Python 3.13.15，torch 2.11.0+cu128，torchvision 0.26.0+cu128，CUDA 12.8，timm 1.0.29，numpy 2.1.3，Pillow 13.0.0，safetensors 0.8.0，huggingface-hub 1.29.0。
- T4×2 可见，每卡 15,636,037,632 bytes；实际推理用 cuda:0、FP16 AMP。复核程序耗时 108.453 秒，不含完整排队/导出时间。原训练显存峰值见 [pilot 报告](PILOT_20261027.md)；本轮未另记录独立推理峰值，不套用训练显存值。

**已验证的修复记录**：v5 中 torch 元数据的 tuple 与 JSON 数组 list 直接比较导致断言失败。v6 仅将 JSON 可表达的序列归一化后再比较；`metadata_comparison.json` 确认值相等，未改模型、权重或归一化。v6 无失败项。

## 4. checkpoint 实际内容与推理导出

**已验证**，原 `best_decoder.pt` 是第 2 epoch 的混合训练 checkpoint，不能称“仅 decoder 权重”。实际键：

```text
best_validation_iou, config, decoder_state, epoch, format, model,
optimizer_state, rng_cpu, scaler_state, software, spatial_split, validation
```

格式 `pvseg_decoder_v1`；optimizer 有 15 个状态条目，含 scaler 与 CPU RNG；没有 backbone 张量。归档原文件与来源文件完全一致：

```text
best_decoder.pt
bytes: 40900283
SHA256: f8db54b5a4ed970e1a4a8e7a51c6e22f0a2f0613b5f35ce7ebddfe8f175227b0
mtime UTC: 2026-10-27T00:22:21.648000+00:00
```

**已验证**，额外导出 `inference_decoder.pt`，格式 `pvseg_inference_decoder_v1`。包含 `decoder_state, epoch, format, model, software, source_checkpoint_sha256, training_git_commit`；无 optimizer/scaler/RNG/backbone。重新读取后逐张量与原 decoder 精确相等。

```text
inference_decoder.pt
bytes: 13600904
SHA256: 216386bfecf739c994a47cac17c5a5ccf57816dfe40829ca4b4211e4e3d19dbb
mtime UTC: 2026-10-27T02:29:19.411795+00:00
requires SAT SHA256: 8d18337f766102f951ac0ca543a22536f58caeb6b6c33cfef73326f866791286
```

两者都需要单独挂载上述精确 SAT 权重及相邻配置；不能作为包含 backbone 的完整模型独立使用。`checkpoint_metadata.json`、`inference_export.json` 保存实际张量形状、来源和校验结果。

**已验证的单张推理入口**：`scripts/infer_pilot.py` 已实际加载推理导出，在 pv003283 上生成与复核完全一致的结果，见 `single_image_test.json`。本轮实际路径示例（只是复现说明，当前不再次运行）：

```bash
python scripts/infer_pilot.py \
  --image /kaggle/input/datasets/zh0416/pv-896/PV-896/负样本/images/pv003283.bmp \
  --checkpoint /kaggle/working/PV-DINOV3/outputs/archive/pilot_baseline/inference_decoder.pt \
  --weights /kaggle/input/notebook5f7aef8a4f/verified_sat/model.safetensors \
  --output /kaggle/working/single_prediction.png
```

新会话须先检查实际输入挂载路径，再替换参数；脚本无自动下载、缩放或训练。

## 5. 逐样本错误复核

**已验证**：ID、类别、source_scene、路径均来自实际 manifest；12 对原图/Mask 实际哈希均与清单一致。以下是固定最佳 checkpoint 的逐图结果，不能与总体微平均混称。

场景简称：S1=`0825_Q_733feba5ac7648`；S2=`0918_Q_4a6bb66935b24c`。flag 为原始 `source_label_uncertain_filename`。

| ID | 类别 | 场景 | flag | TP | FP | FN | IoU | Precision | Recall | 状态 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pv003283 | 负样本 | S1 | True | 0 | 24463 | 0 | 0% | 0% | 无定义 | 已验证 |
| pv003289 | 困难正样本 | S1 | True | 610 | 0 | 5342 | 10.25% | 100% | 10.25% | 已验证 |
| pv003520 | 困难正样本 | S2 | False | 965 | 21 | 866 | 52.11% | 97.87% | 52.70% | 已验证 |
| pv003519 | 困难正样本 | S2 | False | 1051 | 118 | 920 | 50.31% | 89.91% | 53.32% | 已验证 |
| pv003313 | 困难正样本 | S1 | True | 7518 | 884 | 284 | 86.55% | 89.48% | 96.36% | 已验证 |

| ID | 已验证：图像/Mask/预测中直接可见的证据 | 待人工确认：可能原因及所需证据 |
| --- | --- | --- |
| pv003283 | 当前真值全背景；预测集中在亮色条纹矩形结构，按现有 Mask 为误检 | 背景混淆或漏标均有可能；需原分辨率影像、源矢量和标注规则确认结构性质，不能仅凭预测认定为真实光伏 |
| pv003289 | 真值前景 5952 像素，仅占整图约 0.74%，位于右侧边缘；检出 610 像素 | 小可见前景、切片边缘截断可能影响检出；需相邻窗口/完整原场景核验，不证明因果 |
| pv003520 | 真值前景 1831 像素，约 0.23%，位于底边；漏检 866 像素 | 小前景及边缘上下文不足的假设；需完整对象与原场景核验 |
| pv003519 | 真值前景 1971 像素，约 0.25%，位于底边；漏检 920 像素 | 同上；不能据两张图断言 decoder 结构缺陷 |
| pv003313 | 左边缘前景大部分检出，仍有 884 FP、284 FN；作为同类对照 | 轮廓外扩/边界标注差异需核对源矢量；边缘目标并非一律检出失败 |

困难类别仅占本验证集真前景 17556/653745≈2.69%，所以总体像素微平均受普通正样本影响较大，这是计数事实，不是已确认的训练失败原因。本轮未根据这些图自动改标签、删除样本或重训。

**已验证的输出**：归档 `errors/<ID>/` 对上述 5 个 ID 各保存 `rgb.png`、`ground_truth.png`、`prediction.png`、`prediction_overlay.png`、`error_overlay.png`、`review_panel.png`、`sample.json`。面板从左到右为 RGB、真值、预测、预测叠加、错误叠加；错误颜色绿=TP、红=FP、蓝=FN。原始像素尺寸图留在 Kaggle；本地只取回派生面板和报告，没有下载原始数据集或权重。

实际数据相对路径为 `<类别>/images/<ID>.bmp` 与 `<类别>/labels/<ID>.bmp`，类别见表；绝对路径、原始元数据、来源窗口、影像/Mask 哈希记录在各 `sample.json`。12 张完整逐图指标见 `sample_metrics.csv/json`。

## 6. 历史总体结果复现及空 Mask 规则

以下均为**已验证的历史验证结果复现**，不是新的独立评估：

| 范围 | TP | FP | FN | IoU | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全部 12 张 | 576346 | 69091 | 77399 | 79.7340% | 88.7244% | 89.2955% | 88.1607% |
| 普通正样本 4 张 | 566202 | 43605 | 69987 | 83.2902% | 90.8834% | 92.8494% | 88.9990% |
| 困难正样本 4 张 | 10144 | 1023 | 7412 | 54.5993% | 70.6333% | 90.8391% | 57.7808% |
| 负样本 4 张 | 0 | 24463 | 0 | 0% | 0% | 0% | 无定义 |

按累计 TP/FP/FN 计算前景 IoU、F1、Precision、Recall，不平均单图比值。分母为零记录 null。真值与预测均空的三张负样本不人为记 IoU/F1=1；pv003283 的 FP 进入整体。负样本图像误检为 1/4，像素误检率 `24463/(4×896×896)=0.7618%`；pv003283 单图误检像素率约 3.0471%。全部真前景漏检率 11.8393%，困难样本为 42.2192%；非空验证图没有整图完全零 TP，但存在严重局部漏检。

## 7. 空间划分与来源标记复核

**已验证**：读取 `spatial_split.json`、两份真实 manifest 及全量来源元数据，按提供的 EPSG:4326、仿射变换、源影像、row/col offsets 和窗口宽高重新检查；没有仅用 source_scene 不同作为结论。

| 集合/类别 | 样本数 | 场景数 | flag True | flag False | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| 训练全部 | 48 | 14 | 44 | 4 | 已验证 |
| 训练正样本 | 16 | 9 | 14 | 2 | 已验证 |
| 训练负样本 | 16 | 10 | 15 | 1 | 已验证 |
| 训练困难正样本 | 16 | 10 | 15 | 1 | 已验证 |
| 验证全部 | 12 | 2 | 8 | 4 | 已验证 |
| 验证正样本 | 4 | 2 | 3 | 1 | 已验证 |
| 验证负样本 | 4 | 2 | 3 | 1 | 已验证 |
| 验证困难正样本 | 4 | 2 | 2 | 2 | 已验证 |

**已验证的跨训练/验证检查**：

- 同 source_scene、同源影像 basename、同 manifest 影像哈希交集均为 0。
- 有一个相同 Mask 哈希，来源是全背景 Mask；这不能单独作为影像重复泄漏证据。
- 已选 48×12 窗口：地理边界框相交、接触/角点邻接、一个名义像素距离内的跨侧窗口均为 0。
- 已选样本最近边界框间隙约 25,090.159 米（训练 pv003425、验证 pv003309）。
- 扩展到原先分配给两侧的所有候选窗口：同样无相交、接触或一个像素内邻近；最近间隙约 1,521.905 米（pv001812、pv001774）。这项只读取元数据，未扩大像素处理。
- 重算分组和重叠边匹配原记录：16 场景、10 空间组、7 条跨场景重叠关系。详见 `spatial_review.json`。

**未验证/待人工确认的范围**：距离为基于纬度近似换算的边界框间隙，非精确测地多边形距离；没有独立核验原影像定位精度，也没有额外地理缓冲或空间相关性检验。当前不缺基本窗口坐标字段，但还需可靠定位/真实 footprint、跨期同地影像的来源关系、预先确定的缓冲规则及新地域来源，才能支持更强的独立性结论。元数据未见重叠不等于已证明统计独立。

**已验证的 flag 来源**：处理代码 `recrop_dataset_896.py:202`、`recrop_positive_896.py:237` 使用 `"不确定" in shp.name`。全量 3324/3668 带标记；它是源 SHP 文件命名元数据，不是忽略像素或已确认错标。**待人工确认**：原始命名原因和标注验收标准。原有字段均保留，不改 Mask。

当前验证场景仅 S1/S2，参与过选 epoch 和本轮错误分析，不能再称未触碰测试集。16 个场景都已有训练或验证使用记录，不能将剩余邻近切片自动视为新独立测试。优先寻找未参与开发的独立原始场景/地域。

## 8. 归档清单、代码变更与后续计划

**已验证**，归档保留原始 checkpoint、config、逐 epoch history/CSV、训练/验证清单、空间划分、审计报告、原逐 epoch/最佳预测；新增 checkpoint 实际内容报告、推理导出、完整原文件校验清单、12 图复推指标、5 图错误面板、空间复核、README 和单张推理验证。没有在归档中重复保存 SAT 权重。

本轮代码新增 `scripts/review_pilot.py`（只推理和归档）、`scripts/infer_pilot.py`（单图推理）；文档新增本报告、[独立评估计划](INDEPENDENT_EVALUATION_PLAN.md)，并更新 README。代码已提交至现有 `codex/segdino-sat-smoke` 分支，不覆盖项目或 Git 历史。生成物、完整来源清单、原始数据和权重不进入 GitHub。

本地报告/派生面板位于：

```text
E:\PVdataset\.research\pv896-review-20261027\output\PV-DINOV3\outputs\archive\pilot_baseline\
E:\PVdataset\.research\pv896-review-20261027\错误复核图集.md
```

本地独立核对通过：12 图计数求和等于总体、原预测像素差全为 0、训练步数 0、162 个保存文件、checkpoint HTTP 长度与会话内元数据一致；下载目录无 `.pt/.pth/.safetensors/.bin` 权重。

**待人工确认的下一轮计划，尚未执行**：先按原始场景/地域和预定缓冲规则锁定训练/开发验证/独立测试，再比较 A 当前冻结 SAT+原 decoder、B 冻结 SAT+Adapter+同 decoder；C InternVL 仅在明确必要性后研究。固定数据、输入、标签版本、指标、预算与模型选择规则，报告逐场景、困难样本、负样本误检及漏检；不预设 B/C 更好。若更换训练划分，应另行授权同条件重训 A，不能把额外数据和训练预算的效应归因于 Adapter。

本轮已到停止点，不开始任何新训练或独立测试执行。以下人工复核补充后，仍需确认疑似漏标、uncertainty 命名背景、独立地域及缓冲规则、永久资源保存方式。

## 9. 用户人工复核补充（2026-10-27）

本节记录用户查看图像后直接反馈的结论。第 5 节保留自动复核时的证据及当时待确认状态；本节补充最新人工判断，不覆盖历史记录，不修改数据、checkpoint 或历史指标。此补充尚未写入已保存的 Kaggle v6 归档，也没有为文档补充启动新 Notebook。

| ID | 用户反馈 | 当前判断及状态 |
| --- | --- | --- |
| pv003283 | 红色区域“可能是屋顶光伏” | **待人工确认**：存在疑似漏标，尚不能确定为屋顶光伏，也不能将 24463 个历史 FP 直接改算为 TP。保留原负样本类别和空 Mask，需原始大图/更清晰影像及源标注核验 |
| pv003289 | Mask 覆盖了光伏区域 | **已验证（用户人工核查）**：光伏区域已被标注，模型存在明显漏检（Recall 10.25%）。人工反馈支持当前目标标注，不证明每个边界像素已独立审定；漏检机理仍未验证 |
| pv003313 | 模型向背景扩张 | **已验证（用户人工核查）**：外扩区域属于模型对背景的误分，支持边界假阳性判断；现有 884 FP、284 FN 和原 Mask 均保留 |
| pv003519、pv003520 | Mask 准确，切片只切出了完整光伏区域的一个角 | **已验证（用户人工核查）**：这是正确标注的、被切片截断的局部光伏目标，存在模型部分漏检。裁切事实已获用户确认；是否因边缘上下文、小目标或其他模型因素导致漏检仍未验证 |

由此可区分两个后续问题：一是 pv003283 的疑似漏标需要继续核验；二是已标注光伏的漏检及向背景外扩，属于当前模型需要评估的错误。切片截断不自动构成错误样本，不据此删除 pv003519/pv003520，也不自动将其设为忽略像素。

后续独立评估计划增加按距切片边界、目标可见面积和是否截断进行诊断分层；类别组织、固定 0.5 阈值、原始标签及所有历史分数保持可追溯。当前不据此选择新模型或启动训练。
