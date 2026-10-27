# PV-DINOV3

Photovoltaic research workspace. GitHub stores source code, configuration,
notebooks, and documentation. Kaggle stores datasets, model weights, and run
outputs, and provides the compute environment.

## Connected services

- GitHub: https://github.com/Zh0416/PV-DINOV3
- Local checkout: `E:\PVdataset\PV-DINOV3`
- Kaggle CLI environment: `E:\PVdataset\.venv-kaggle`
- Authentication: existing GitHub CLI login and the `KAGGLE_API_TOKEN`
  environment variable. Never put tokens in Git or notebook source.

## Workflow

1. Edit source and configuration in this repository, then commit and push to
   GitHub.
2. Configure Kaggle input mounts in `kaggle/kernel-metadata.json` using
   `dataset_sources` and `model_sources`. Data and pretrained weights are read
   from `/kaggle/input`; generated weights and reports go to
   `/kaggle/working/outputs`.
3. Set `ENTRYPOINT` in `kaggle/run.py` to the desired repository script. The
   initial entrypoint checks connectivity and mounted paths without training.
4. Submit and inspect the private Kaggle run:

   ```powershell
   .\tools\kaggle.ps1 kernels push -p .\kaggle
   .\tools\kaggle.ps1 kernels status zh0416/pv-dinov3
   ```

`kaggle/run.py` clones the configured GitHub ref on each fresh run, prints the
resolved commit, and executes the repository entrypoint. Internet access is
enabled. The initial kernel uses CPU; enable GPU in its metadata for training.
It intentionally starts without data or model mounts. Add the required Kaggle
resources before training.

The GitHub repository is public, so Kaggle can clone it without a GitHub token.
Keep the Kaggle run private and keep secrets out of source files and logs.
Changing GitHub visibility to private requires updating Kaggle authentication.

## Local CLI

Run from this checkout:

```powershell
.\tools\kaggle.ps1 --version
.\tools\kaggle.ps1 datasets list --mine
.\tools\kaggle.ps1 kernels list --mine
```

The wrapper uses the isolated CLI environment and inherits the environment
token. It does not save credentials. To recreate that environment:

```powershell
python -m venv E:\PVdataset\.venv-kaggle
E:\PVdataset\.venv-kaggle\Scripts\python.exe -m pip install -r requirements-cli.txt
```

Kaggle notebook outputs remain attached to their run. Preserve checkpoints
needed by later runs as a Kaggle Dataset or Model version and mount that
resource in the next run. Do not download weights into this repository or
commit them to GitHub. `.gitignore` excludes common secrets, datasets, weights,
and generated artifacts.

## SegDINO SAT supervised baseline

The 2026-10-27 Kaggle GPU smoke test **passed** on the audited 12 pairs, with
one optimizer step only. See [the verified run record](docs/SMOKE_TEST_20261027.md)
for the tested commit, hardware, real tensor shapes, timings and limitations.
The subsequently authorized 60-pair pilot also **completed**: 48 training pairs,
12 validation pairs, batch size 1, seed 42, exactly 3 epochs. Best validation
IoU was 0.7973 at epoch 2. See [the pilot report](docs/PILOT_20261027.md)
for spatial grouping, category metrics, saved outputs and limitations.
The run stopped after the pilot; full-dataset training has not run.

The subsequent **inference-only review and archive completed** in Notebook v6:
all 12 validation predictions reproduced exactly, with zero training steps.
The original mixed training checkpoint and a separate inference decoder export
are verified in the saved v6 outputs under
`PV-DINOV3/outputs/archive/pilot_baseline/`. See the
[review/archive report](docs/REVIEW_ARCHIVE_20261027.md) for actual file hashes,
error cases, saved-output checks and spatial limitations, and the
[independent evaluation plan](docs/INDEPENDENT_EVALUATION_PLAN.md) for proposed
next steps. The two existing validation scenes are development data, not an
untouched test set. No new training or independent evaluation has been started.

This is a fully supervised **engineering baseline**, not an exact reproduction
of the paper or a weakly supervised method. Real pixel masks supervise training.
The upstream source is `script-Yang/SegDINO` commit
`c078ca0e2c7578304e80eaa4e9d3f7c6dd0f3c41`; adapted decoder code retains its MIT
notice in `third_party/SegDINO-LICENSE`.

Preserved root-decoder topology: four 1x1 projections to 96/192/384/768 channels,
four 3x3 scratch convolutions to 128 channels, 4x transposed convolution on the
first map, bilinear alignment of the other maps, concatenation and a binary 1x1
prediction layer. Final logits are resized to the input dimensions with
`align_corners=True`. This does not use the upstream fine-grained decoder.

Changed: timm SAT-L encoder instead of the upstream torch.hub encoder; strict
binary data handling; verified input normalization; permanently frozen/eval
encoder; decoder-only AdamW; 0.5 BCEWithLogits + 0.5 soft Dice; AMP and guarded
smoke/pilot entrypoints. The inspected upstream training entrypoints did not
call their `lock_backbone` method, and their satellite loss was BCE only.

### Environment and weights

Use Kaggle's CUDA PyTorch and torchvision; do not replace them with CPU wheels.
`requirements.txt` documents dependencies. The launcher pins only timm to
1.0.29 without reinstalling PyTorch. Missing/incompatible dependencies should
be fixed explicitly. CUDA hardware, package versions, memory and precision are
recorded for each actual run; prior CPU audit versions are not GPU guarantees.

The target `vit_large_patch16_dinov3.sat493m` is a timm **Eva** model. Read API:
`forward_intermediates(indices=[4,11,17,23], norm=True, output_fmt='NCHW',
return_prefix_tokens=False, intermediates_only=True)`. Indices are zero-based.
There is no assumed `features_only` wrapper. For 896 RGB inputs, four
`[B,1024,56,56]` maps are expected and must pass actual forward assertions.

Only timm-converted safetensors with adjacent `config.json` are supported.
Mount a Kaggle Dataset/Model containing these files and pass `--weights`.
Alternatively, after user confirmation, `--download_weights` fetches only
inside Kaggle from `timm/vit_large_patch16_dinov3.sat493m`, pinned revision
`ce148bd5ed3de1eb75ebdccafcb2c280bb3e8f7c`, into
`/kaggle/working/model_cache`. Weight license is the upstream DINOv3 license;
weights are never copied into GitHub. Both SHA-256 and source are recorded.
Configuration mean `(0.430,0.411,0.296)` and std `(0.213,0.156,0.143)` are
checked against the constructed model before normalization is used.

### Data contract and audit

Always inventory `/kaggle/input` first. The audit observed root
`/kaggle/input/datasets/zh0416/pv-896/PV-896`, but code discovers the mounted
root from `manifest.csv` and the three category/image/label directory pairs.
It rejects ambiguous roots rather than choosing an arbitrary directory.
The observed dataset has 1300 positive, 1500 negative and 868 hard-positive
pairs. Categories organize samples; the only segmentation classes are
background=0 and PV=1. The documented raw BMP label encoding is 0/255; unknown
values and non-RGB/non-L modes fail instead of silently merging labels.

Manifest IDs must match image and label basenames, and paths must stay within
the dataset root. Optional `--image_dir`/`--mask_dir` overrides must both be
provided and are also matched by ID, never by sorting. Train-only synchronized
horizontal/vertical flips and quarter rotations preserve mask alignment.
Validation has no random augmentation and neither path resizes raw masks.

First-round seed=42 selection is fixed in `configs/smoke_sample_ids.json`:
four pairs per category, exactly the already audited 12 unique pairs. Pixel
reads remain limited to those IDs during smoke testing. Full filename and
manifest scans are metadata only. Existing flags indicating uncertain source
label filenames are retained as provenance; they are not ignore labels.
3324/3668 source records carry this flag. Their label correctness has not
been independently established by the filename audit.

### Staged execution

`kaggle/smoke_bootstrap.py` is a Notebook cell template. Before submission,
pin `GIT_REF` to the reviewed Git commit. Its safe default runs only package,
CUDA, file-inventory and synthetic unit checks; it does not download weights.
After the mount/download choice is confirmed, use either
`RUN_ARGUMENTS=['--weights','/kaggle/input/<actual-resource>/model.safetensors']`
or `RUN_ARGUMENTS=['--download_weights']`. Keep `zh0416/pv-896` attached to
the existing private `zh0416/notebook5f7aef8a4f` and enable a GPU.

Equivalent commands after pulling the reviewed commit inside Kaggle:

```bash
python scripts/kaggle_smoke.py --preflight_only
python scripts/kaggle_smoke.py --weights /kaggle/input/ACTUAL_RESOURCE/model.safetensors
# Alternative, only after confirmation: python scripts/kaggle_smoke.py --download_weights
```

Smoke uses batch size 1, one optimizer step and only the audited 12 pairs. It
checks loader shapes/dtypes, actual four encoder maps, output `[1,1,896,896]`,
finite loss/gradients, frozen encoder state, decoder updates and checkpoint
prediction consistency. Decoder/optimizer reload reuses the same verified
frozen encoder. It records timing, hardware and peak GPU memory and saves 12
prediction masks/overlays. A random decoder after one step has no research
performance meaning. The script stops; it never invokes pilot training.

Outputs: `/kaggle/working/PV-DINOV3/outputs/smoke/` contains
`preflight_report.json`, `file_inventory.csv`, `core_tests.txt`,
`smoke_report.json`, `selected_samples.json`, `smoke_decoder.pt`, and
`predictions/`. Failures include stage, traceback and memory in the smoke
report. Decoder-only checkpoints include model/weight/normalization metadata,
software versions, configuration, CPU RNG and optimizer state; recovering the
model requires the separately mounted/pinned pretrained encoder weights.
These files are not a promise of exact training resumption: AMP scaler and
all generator/augmentation states are not restored by the current loader.

Only after every smoke check passes **and the user approves** may a pilot run:

```bash
python train_pv896.py --allow_pilot --input_root /kaggle/input \
  --weights /kaggle/input/ACTUAL_RESOURCE/model.safetensors \
  --img_size 896 --batch_size 1 --epochs 3 --seed 42 \
  --output_dir /kaggle/working/PV-DINOV3/outputs/pilot
```

The guarded pilot caps the combined train/validation set at 60 pairs, at most
20/category, and 3 epochs. Entire scenes are joined when their tile geographic
bounds overlap, their source-image basenames match, or their manifest image
hashes match. Groups are separated before sampling. This uses supplied CRS
and affine metadata, with no additional geographic buffer; it is not an
independent verification of geolocation. Missing spatial provenance stops
the pilot; there is no silent random fallback.
Logs include split IDs, category coverage, epoch losses and overall/category
metrics. Threshold is sigmoid>=0.5; foreground IoU/F1/precision/recall come
from cumulative TP/FP/FN, not an average of per-image scores. Undefined zero
denominators are null. Correct empty negatives get no artificial perfect
foreground score; false positives contribute to the pooled FP count, with
empty-target false-positive images and pixel ratios reported separately.
`best_decoder.pt` is selected by pooled validation foreground IoU. No command
in this implementation launches full-dataset training.

For the specifically approved pilot, `python scripts/kaggle_pilot.py` first
inventories actual mounts/cache, runs synthetic checks, and records the
source-filename uncertainty flag evidence. It restores matching cached SAT
bytes or fetches the same pinned official revision inside Kaggle, verifies
the Smoke Test SHA256, then runs exactly 3 epochs at batch size 1. It refuses
to retrain in an output directory that already contains epoch history.
Train/validation CSV manifests and configuration are saved before training.
Every epoch records per-sample/category TP/FP/FN and losses, timing, memory,
negative FP rates and positive images with no true-positive pixels. It saves
all validation masks/overlays, reloads the best checkpoint and verifies its
metrics. `completion.json` confirms the bounded stop. Weights remain in
Kaggle; `verified_sat/` uses real files so output export need not preserve
Hugging Face cache symlinks. Download only selected reports/visualizations
locally, never an unfiltered Notebook output bundle containing weights.

Synthetic checks (no real samples or weights):

```bash
python -m unittest discover -s tests -v
```
