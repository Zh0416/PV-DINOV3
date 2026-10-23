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
