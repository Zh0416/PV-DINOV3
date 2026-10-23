"""Check the execution environment without reading data or loading models."""

from pathlib import Path
import json
import os
import platform
import subprocess


def main():
    input_root = Path(os.environ.get("PV_DATA_ROOT", "/kaggle/input"))
    output_root = Path(os.environ.get("PV_OUTPUT_ROOT", "/kaggle/working/outputs"))
    in_kaggle = Path("/kaggle/working").is_dir()
    report = {
        "python": platform.python_version(),
        "github_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip(),
        "running_on_kaggle": in_kaggle,
        "data_root": str(input_root),
        "mounted_inputs": sorted(p.name for p in input_root.iterdir())
        if input_root.is_dir() else [],
        "output_root": str(output_root),
    }
    if in_kaggle:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "connection-check.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
