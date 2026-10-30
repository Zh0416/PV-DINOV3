# A3 repaired full-development evaluation, prespecified before GPU execution

This is a supervised known-PV adaptation of reconstructed SSPNet, not the paper's novel-class benchmark. No query results will select the method, update count, checkpoint, or thresholds.

- Reuse exactly the PSPNet base checkpoint at step 160 from `zh0416/pv-a3-learned-prior/2`, source `a6733794eb2fc2cf43c762ecf62952c5b1e4b15c`. Verify its allocation, step, split hash and record checkpoint SHA256. No stage-one updates in this run.
- Initialize stage two afresh with seed 42; do not resume the five-image diagnostic checkpoint. Verify all 320 query/support pairs against the old learned-A3 schedule.
- Retain native 896 pixels, original 22 training masks (4 positive, 18 negative), AdamW 1e-4, weight decay 1e-4, gradient clip 10, no augmentation, frozen encoders and the original architecture.
- Use FP32 for training and inference. Add balanced raw-meta CE with weight 1 to the existing four losses. Present-class pixel CE means receive equal weights within each episode; empty masks supervise background only. This term is an explicit adaptation, not an original-paper loss.
- At updates 80/160/240/320, compute pooled foreground micro-IoU on the 12 calibration images separately for three reserved support choices. Select the largest mean **complete final-output IoU at probability >= 0.5**, ties at the earlier update. Probability equality predicts foreground, consistently with all previous diagnostic thresholding.
- At that selected checkpoint, calibrate additional final/meta/CAM thresholds using calibration histograms (4096 bins). These are separate reported outputs. A2 thresholds remain unchanged.
- Open query image and mask pixels only after selecting and restoring the checkpoint. Evaluate the original 58 statistical queries plus two explicitly excluded previews with the three original one-shot supports. Require exact per-image/per-support A2 confusion-count reproduction.
- Report full fixed/calibrated outputs, meta, frozen base, A2 and jointly learned priors. Priors are diagnostics, not full-network predictions or independently trained ablations. Pool confusion counts before computing each support's metrics, then average the three support results; do not ensemble predictions.
- Report all difficulty groups, negative-image FPR, complete update/gradient audits and all 60 derived preview panels. Keep original data, foundation weights, feature caches and learned checkpoints on Kaggle.

Relative to old learned A3, this run also changes AMP to FP32 and checkpoint selection from fused-prior to full-output performance. Therefore the full-run comparison is not a single-factor loss ablation. The earlier paired diagnostic provides the evidence specifically about saturation and the added loss.

The query set has already been inspected repeatedly. All results are development evidence; only one training spatial group is available. Three supports are three selections for one training run, not three training seeds. A separate spatial test set remains necessary for a generalization claim.
