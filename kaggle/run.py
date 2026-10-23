"""Fetch GitHub source and run it using Kaggle-mounted data and weights."""

from pathlib import Path
import os
import subprocess
import sys


REPOSITORY = "https://github.com/Zh0416/PV-DINOV3.git"
GIT_REF = "main"
ENTRYPOINT = "scripts/check_runtime.py"


def main():
    working = Path("/kaggle/working")
    if not working.is_dir():
        raise RuntimeError("This launcher must run inside Kaggle.")

    repo = working / "PV-DINOV3"
    if repo.exists():
        origin = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=repo, text=True
        ).strip()
        if origin != REPOSITORY:
            raise RuntimeError("Existing checkout has an unexpected Git remote.")
    else:
        subprocess.run(["git", "clone", "--no-checkout", REPOSITORY, str(repo)], check=True)

    subprocess.run(["git", "fetch", "origin", GIT_REF], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo, check=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    print(f"GitHub commit: {commit}", flush=True)

    output = working / "outputs"
    output.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment["PV_DATA_ROOT"] = "/kaggle/input"
    environment["PV_OUTPUT_ROOT"] = str(output)
    subprocess.run(
        [sys.executable, str(repo / ENTRYPOINT)],
        cwd=repo,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
