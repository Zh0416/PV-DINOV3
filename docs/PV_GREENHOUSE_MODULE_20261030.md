# 光伏与大棚语义对比先验：第一版设计与实现

状态：已在 24 张真实开发影像、三个独立 1-shot 支持选择上完成冻结推理；保留原 A3 对照，未替换主训练流程。后续亮度门控小实验见 `PV_BRIGHTNESS_GATE_20261030.md`。

## 已完成的第一轮真实结果

沿用原 A3 阈值时，13 张大棚干扰场景的光伏 micro IoU 从 42.13% 提升至 45.64%，该场景全部非光伏背景 FP 减少 31.56%，但光伏召回从 85.41% 降至 77.73%。11 张大棚正图的逐图等权 IoU 从 41.60% 变为 41.56%，说明不能仅依据 micro IoU 宣称普遍改善。原 A3 的 24 × 3 组混淆计数和原校准阈值均与旧结果一致。

HTML：`outputs/greenhouse_pilot_20261030/index.html`。真实推理来自私有 Kaggle kernel `zh0416/pv-greenhouse-contrast-pilot` version 1；本地独立数值复核通过。大棚仍只有整图场景标签，不能将全部背景 FP 变化归因于大棚。

## 不需要新增大棚标签的前提

额外采集困难样本不是启用这个模块的前提。第一版利用 GeoRSCLIP 预训练中已有的光伏、大棚和其他地物概念，在其原有图文空间中进行竞争，并保留 DINOv3-SAT 的支持匹配。新增的是文本描述，不是大棚像素标注。

这不保证零样本就能分清全部大棚。若预训练编码器本身没有区分当地大棚与光伏，或者两者在影像分辨率下已经难以辨认，模块没有新的判别证据，调整融合公式无法消除这个限制。

若下一阶段训练，已有二值光伏掩膜已经把非光伏大棚标为背景，无需另画大棚类别。但只有现有图像实际含有大棚时，二值监督才包含这种背景的视觉证据。自动挖掘背景误报属于困难负例挖掘，应与“完全不使用困难负例”区分。

## 接口与公式

实现：`pvseg/greenhouse_prior.py` 的 `GreenhouseContrastPrior`。

- `geo`：与 A3 相同的 `[B,512,56,56]`。先将每个 GeoRSCLIP 图像窗口的最终投影向量 L2 归一化，再按坐标重叠平均；平均之后不重新归一化。不能传入未投影的 CLIP patch token 或 DINO 的 1024 维特征。
- `spatial`：A2 的 `[B,1,56,56]`，即支持前景／背景余弦差除以二。
- 文本由同一套 RS5M ViT-B/32 QuickGELU 编码器得到；模板内先逐条归一化，再平均。模板均值不再归一化，以保持平均余弦。
- 其他背景包含屋顶、水面、裸地、植被。这些是竞争概念，没有为图像虚构对应的类别标注。尤其“屋顶”与屋顶光伏可以共存，不能做硬排除。

记光伏、大棚、最强其他背景的文本相似度分别为 `s_pv, s_gh, s_other`：

```text
e = relu(min(s_gh - s_pv - margin_pv,
             s_gh - s_other - margin_other))
penalty = min(strength * e, max_penalty)
semantic_corrected = clamp(s_pv - penalty, -1, 1)
score = (1-alpha) * A2 + alpha * semantic_corrected
```

只有大棚同时胜过光伏和其他背景时，才触发有界语义修正。模糊区域、两类同分区域不会强制二选一。DINO 空间项保持原权重；这并不保证光伏召回不下降，因为文本也可能误判。修正仍须通过验证约束。

输出含原始两类响应、其他背景响应、语义差值、大棚证据、抑制幅度、修正前后融合分数和 `fused_cam`。`fused_score` 延续 A3 的 `[-1,1]` 分数接口，可以沿用分数直方图校准；`fused_cam = sigmoid(cam_scale * score + cam_bias)` 用于需要 `[0,1]` 的下游接口，默认参数未经校准，不应解释成概率。输出 896² 时先插值分数，再计算 sigmoid。

`alpha=.7, strength=1, max_penalty=.2` 只是起始设置，不是已证明有效的最优参数。尤其 `.7` 来自旧 A3 的开发校准，不代表新模块的最优权重。模块只有固定参数和文本缓冲区，没有可训练参数，也没有改变原 DMSF 或 A3 运行脚本。

## 复用 A3 编码器的最小接入示例

```python
from pvseg.greenhouse_prior import (
    DEFAULT_CONCEPT_PROMPTS, GreenhouseContrastPrior, encode_concepts,
)
from pvseg.frozen_baselines import (
    aligned_crop_features, foreground_background_response,
)

# clip 是 A3 已验证、冻结且 eval() 的 GeoRSCLIP；tokenizer 与之配套。
banks = encode_concepts(clip, tokenizer, device='cuda')
prior = GreenhouseContrastPrior(banks).cuda().eval()

# crop_embeddings: [49,512]，每个窗口已归一化；query_dino: [1,1024,56,56]
# fg, bg 来自唯一允许的支持掩膜。查询掩膜不参与以下计算。
geo = aligned_crop_features(crop_embeddings, boxes, (896, 896), (56, 56))
spatial = foreground_background_response(query_dino, fg, bg, (56, 56))
maps = prior(geo, spatial, output_size=(896, 896))
score = maps['fused_score']  # 校准阈值后才生成二值掩膜
cam = maps['fused_cam']     # 提供给后续先验接口，不是概率承诺
```

上述代码是接入示例，不是新实验的完成记录。需用实际 checkpoint 得到新增文本向量；原来的单个光伏文本缓存不足以构造此模块。49 个原图像窗口向量如已缓存，可以复用，无需为每条文本重跑图像编码器。

## 实验顺序

1. 复算原 A3。光伏文本保持旧版本；先固定旧 alpha、阈值，令 `strength=0`，结果应逐像素一致。
2. 仅开启大棚竞争修正；固定 alpha，比较 `strength=0` 与 `strength=1`，初版不增加多种光伏 prompt、额外投影层或动态门控。
3. 对第二组独立报告固定旧阈值与统一规则重新校准阈值两种结果。训练／校准集确定强度和上限，查询集不参与选择。新增尝试数与校准标签使用量需记录。
4. 正图报告光伏 Precision/Recall/AP；纯负图报告误报率；有光伏且伴随大棚的图像必须独立检查，避免把“整幅图压低”误当作判别能力提高。在相同光伏召回下比较背景误报，并展示两类分数与抑制图。
5. 现有二值掩膜只能证明非光伏误报变化；没有独立大棚区域标签时，不能声称测得“大棚像素分类准确率”或精确的大棚区域误报率。影像场景分组可作诊断，但其中背景还包括其他地物。

当前开发图已反复查看，最终泛化结论应来自未参与设计和校准的新空间区域。三个支持选择仍是三个独立 1-shot，不是三个训练种子。

## 若冻结模块仍难以分开

下一步可训练小型 DINO→Geo 特征适配器，利用已有二值掩膜直接监督融合 CAM。可以在训练图中的已标背景里自动选择高光伏响应区域施加额外背景损失；无需人工额外挑图，但应如实称为自动困难负例挖掘。背景监督只要求“不是光伏”，不要求把所有背景都学成“大棚”。

局部密集 Geo 特征、DINO 关系引导的聚合应单独测试。当前 224² 窗口包含混合地物时，窗口语义会扩散到实际光伏上；有界修正只减轻这一风险，不能恢复窗口内不存在的精细语义定位。小于 patch 尺度的线条和边界也不能通过简单上采样恢复。

该模块借鉴语义竞争和残差修正，尚不构成已验证的新论文方法。它刻意先回答一个可检验的问题：保留图文对齐与支持匹配、不新增大棚标签，显式比较大棚语义是否有助于降低当前误报？

## 本地验证

运行 `python -m unittest discover -s tests -p test_greenhouse_prior.py -v`。合成测试检查关闭时回退 A3、只在特定语义证据下抑制、混合区域不强制二选一、窗口／模板平均一致性、896² 输出、状态保存及错误特征空间拒绝。它们验证数值和接口，不验证真实光伏／大棚识别效果。
