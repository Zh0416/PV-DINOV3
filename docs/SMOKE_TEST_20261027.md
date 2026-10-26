# PV-896 SegDINO SAT smoke result: PASS

This is an engineering check of a fully supervised baseline, not an accuracy
evaluation. Stop point reached: no 60-pair pilot or full-dataset training ran.

## 1. Repository and actual Kaggle environment

- Original main: `c18f89e0f736808ec694c1c1df1567ce5e96beb4`; history/remote preserved.
- Implementation branch: `codex/segdino-sat-smoke`.
- Tested commit: `ef50d3ca012ab89e5e983509fe3fba89b61d0b73`.
- Notebook: https://www.kaggle.com/code/zh0416/notebook5f7aef8a4f
  (select version 2 in the notebook history; both versions 1 and 2 completed).
- Version 1: metadata/GPU/synthetic-test preflight only. Version 2: 12-pair smoke.
- Tesla T4 x2, 15,636,037,632 bytes each; smoke used only cuda:0. RAM 31.35 GiB.
- Python 3.13.15; torch 2.11.0+cu128; torchvision 0.26.0+cu128; CUDA 12.8;
  timm 1.0.29; numpy 2.1.3; Pillow 13.0.0; safetensors 0.8.0;
  huggingface-hub 1.29.0.
- AMP enabled: float16 and GradScaler. Actual returned feature/logit tensors
  were float32; enabling autocast does not require every operation to return fp16.

## 2. Mounted data and audit scope

Observed root: `/kaggle/input/datasets/zh0416/pv-896/PV-896`. The launcher first
inventories `/kaggle/input`, then discovers a unique root using manifest and
category directory structure. No raw dataset download or upload was needed.

| Folder category | Mounted pairs | Smoke pairs |
| --- | ---: | ---: |
| positive | 1300 | 4 |
| negative | 1500 | 4 |
| hard positive | 868 | 4 |

All 3668 manifest paths were checked by ID/file metadata. Pixel reads were
limited to the same 12 audited IDs in `configs/smoke_sample_ids.json`, seed 42.
All 12 loader checks passed: 896x896 RGB uint8 image, L uint8 mask with only
0/255, emitted float32 image `[1,3,896,896]` and mask `[1,1,896,896]`.
The four negatives were empty; positives and hard positives had foreground.
There are two segmentation classes, not three folder-derived classes.

The prior audit saved `sample_audit.csv`, `audit_report.json`, overlays and
structure reports in the separate Kaggle notebook `zh0416/pv896-data-audit-20261026`
version 2. Full pixel/hash verification of all 3668 pairs has not been performed.

## 3. Upstream and adaptation

Paper: https://arxiv.org/abs/2509.00833 (v1). Upstream root decoder inspected at
`script-Yang/SegDINO` commit `c078ca0e2c7578304e80eaa4e9d3f7c6dd0f3c41`.
Preserved 1x1 projections to 96/192/384/768, 3x3 scratch to 128, first-map 4x
transposed convolution, interpolation/concatenation, 1x1 binary prediction and
final input-size interpolation (`align_corners=True`). MIT notice retained.

Adapted torch.hub to timm SAT-L, normalized intermediate maps, strict binary
loading, verified satellite normalization, always-eval frozen backbone,
decoder-only AdamW, 0.5 BCE + 0.5 Dice, AMP, checkpoint and stage guards. This
is not the upstream fine-grained decoder or a claim of exact paper reproduction.

## 4. Actual model and pretrained weights

- Model: `vit_large_patch16_dinov3.sat493m`; actual class `timm.models.eva.Eva`.
- Constructor used `pretrained=True`, local pretrained file override,
  `num_classes=0`, `img_size=896`; no assumed `features_only` wrapper.
- Actual API `forward_intermediates`: indices `[4,11,17,23]` (zero-based),
  `norm=True`, `output_fmt='NCHW'`, `return_prefix_tokens=False`,
  `intermediates_only=True`.
- Actual four maps: each `[1,1024,56,56]`, float32, detached.
- Actual logits `[1,1,896,896]`, float32. Backbone 303,079,424 parameters;
  decoder 3,397,537 parameters. SAT-493M is the weight tag, not this parameter count.
- User-authorized official weight fetch occurred only inside Kaggle:
  `timm/vit_large_patch16_dinov3.sat493m`, revision
  `ce148bd5ed3de1eb75ebdccafcb2c280bb3e8f7c`, `model.safetensors`.
- Weight SHA-256: `8d18337f766102f951ac0ca543a22536f58caeb6b6c33cfef73326f866791286`.
- Verified mean `(0.430,0.411,0.296)`, std `(0.213,0.156,0.143)`.
- Encoder weights/cache stayed under `/kaggle/working/model_cache`, never
  downloaded locally or committed. The run outputs include the hashed cache
  blobs; snapshot symlinks may not survive notebook-output export. For future
  reuse, reconstruct real `model.safetensors` + `config.json` from those blobs
  inside Kaggle or fetch the same pinned revision. A permanent Kaggle
  Dataset/Model resource has not yet been created.

## 5. Smoke checks and measurements

| Check | Result |
| --- | --- |
| Six synthetic alignment/label/metric/split/loss/checkpoint tests | PASS |
| All 12 loader shapes, dtypes and strict binary masks | PASS |
| Real pretrained encoder API and four finite detached maps | PASS |
| Decoder shape equals original mask shape | PASS |
| Finite loss | 0.8086189628 |
| Exactly one backward + AdamW step | PASS; decoder grad norm 2.01711974 |
| Decoder parameter update | 15 tensors changed |
| Frozen backbone | eval; no gradients; parameter versions unchanged |
| Optimizer parameter ownership | decoder only |
| Save/reload decoder + optimizer | PASS; same verified frozen encoder reused |
| Reload prediction maximum absolute difference | 0.0 |
| Predictions/loss finite on all 12 IDs | PASS; masks and overlays saved |

| Measurement (batch size 1, 896x896) | Seconds |
| --- | ---: |
| Weight acquisition + hash | 15.650 |
| Model construction/loading + GPU transfer | 4.889 |
| First backbone forward | 2.340 |
| First decoder forward | 0.411 |
| First batch forward | 2.752 |
| Forward + loss + backward/checks/optimizer step | 3.718 |
| Entire smoke script | 74.466 |

Training peak PyTorch allocated memory: 1,630,515,712 bytes (1.519 GiB).
Training peak PyTorch reserved memory: 1,858,076,672 bytes (1.730 GiB).
These are one instrumented run, including cold kernels; not steady throughput
or a measurement of every CUDA/driver allocation. No larger batch was tested.

No smoke execution error required repair. A native-BF16 hardware guard was
added before version 2. Notebook-export tooling emitted nonfatal SyntaxWarnings;
the actual notebook status and structured smoke report were COMPLETE/PASS.
Three returned category overlays were visually inspected. Predictions were
mostly background with some false positives on negatives, as expected from an
untrained decoder after one update; they do not demonstrate segmentation ability.

## 6. Pilot status

Not run. Await user confirmation before up to 60 total pairs, at most 20 per
category and 3 epochs. Split by source scene before sampling; keep backbone
frozen. Report pooled and category TP/FP/FN metrics, with empty denominators
undefined and negative false positives separately reported. No full training.

## 7. Files, commands and saved artifacts

Added `pvseg/{data,model,losses,metrics,runtime}.py`, `smoke_test_pv896.py`,
`train_pv896.py`, `configs/smoke_sample_ids.json`, `requirements.txt`,
`tests/test_core.py`, `scripts/kaggle_smoke.py`, `kaggle/smoke_bootstrap.py`,
`third_party/SegDINO-LICENSE`; expanded README and this run record.

Tested command inside the pinned Kaggle checkout:

```bash
python scripts/kaggle_smoke.py --download_weights
```

Kaggle output root `/kaggle/working/PV-DINOV3/outputs/smoke/`:
`smoke_report.json`, `preflight_report.json`, `core_tests.txt`,
`file_inventory.csv`, `selected_samples.json`, `smoke_decoder.pt`,
`predictions/*_prediction.png`, `predictions/*_overlay.png`.
Checkpoint/encoder weights remain Kaggle-only. Only reports/logs and three
derived overlay PNGs were retrieved locally. The original notebook source and
both submission snapshots were backed up outside the Git checkout.

## 8. Limitations and next decision

3324/3668 manifest rows have an uncertain-source-filename flag, including all
12 selected rows. This is provenance, not an identified ignore-pixel encoding
or proof of label correctness. User elected to retain the labels and flag.
The metadata audit found 16 source scenes and 8108 overlapping same-source
window pairs. Scene-group splits are required; overlap between different
source scenes is not excluded. Mask semantic accuracy still needs domain review.
Full pixel audit, pilot metrics, generalization and weak-supervision design
remain untested. The next decision is approval of the bounded pilot only.
