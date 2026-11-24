"""Build a self-contained Kaggle GPU notebook for the corrected PINNS map convention."""

import argparse
import base64
import gzip
import io
import json
from pathlib import Path
import tarfile


def build_notebook(project_root: Path, out_ipynb: Path, *, suite: str = "full") -> None:
    if suite not in ("paired", "full"):
        raise ValueError("Expected a paired or full GPU suite")
    project_root, out_ipynb = Path(project_root), Path(out_ipynb)
    original = project_root / "kaggle_jobs" / "unstructured_shortpose_exp_v1.ipynb"
    if out_ipynb.resolve() == original.resolve() or (
        out_ipynb.exists() and original.exists() and out_ipynb.samefile(original)
    ):
        raise ValueError(
            "The archived V1 notebook must retain its original bytes. Select another output path."
        )

    members = [
        *sorted((project_root / "src").rglob("*.py")),
        project_root / "tests/test_invariants.py",
        project_root / "scripts/cloud_experiment_runner.py",
        project_root / "scripts/run_heldout_cloud.py",
        project_root / "scripts/kaggle_causal_entry.py",
    ]
    if suite == "paired":
        study = project_root / "reviews/pinns_map_axis_comparison/neural"
        members.extend((study / "run_neural.py", study / "PROTOCOL.md"))
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for path in members:
                raw = path.read_bytes()
                info = tarfile.TarInfo(path.relative_to(project_root).as_posix())
                info.size = len(raw)
                tar.addfile(info, io.BytesIO(raw))
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")

    setup = f"""# Prepare an isolated project directory for this exact payload.
import base64, hashlib, io, os, subprocess, sys, tarfile
from pathlib import Path

OUT_ROOT = Path("/kaggle/working")
if not OUT_ROOT.is_dir():
    raise RuntimeError("Use a private Kaggle GPU notebook with Internet enabled.")
PAYLOAD_B64 = "{payload}"
project_bytes = base64.b64decode(PAYLOAD_B64, validate=True)
PROJ_DIR = Path("/tmp") / ("shortpose_{suite}_" + hashlib.sha256(project_bytes).hexdigest())
PROJ_DIR.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(project_bytes), mode="r:gz") as tar:
    tar.extractall(PROJ_DIR)
RUN_ENV = dict(os.environ, PYTHONPATH=str(PROJ_DIR), CUBLAS_WORKSPACE_CONFIG=":4096:8")
"""
    prepare = """# Verify the private attachment and run the packaged invariant tests in a fresh interpreter.
matches = sorted(Path("/kaggle/input").rglob("pinns_source_label_causal_maps.bin"))
if not matches:
    raise FileNotFoundError("Attach the private dataset containing pinns_source_label_causal_maps.bin")
MAP_PATH = matches[0]
check = ("import hashlib,sys,torch; from pathlib import Path; "
         "from src.data.pinns_causal_input import MAP_SHA256; "
         "assert torch.cuda.is_available(), 'Kaggle GPU required'; "
         "assert hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()==MAP_SHA256, 'Incorrect map input'")
subprocess.run([sys.executable, "-c", check, str(MAP_PATH)], cwd=PROJ_DIR, env=RUN_ENV, check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime-gpu", "gdown"], check=True)
subprocess.run([sys.executable, str(PROJ_DIR / "tests/test_invariants.py")],
               cwd=PROJ_DIR, env=RUN_ENV, check=True)
"""
    execute = f"""# Recover the real observations, join the causal map and run the {suite} experiment.
subprocess.run([sys.executable, "-u", "-m", "scripts.kaggle_causal_entry",
                "--suite", "{suite}", "--input-map", str(MAP_PATH)],
               cwd=PROJ_DIR, env=RUN_ENV, check=True)
"""
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "id": name,
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            }
            for name, code in (
                ("prepare-runtime", setup),
                ("verify-inputs", prepare),
                ("run-suites", execute),
            )
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_ipynb.parent.mkdir(parents=True, exist_ok=True)
    out_ipynb.write_text(json.dumps(nb, indent=2) + "\n")
    print(f"Built {out_ipynb} ({out_ipynb.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--suite", choices=("paired", "full"), default="full")
    args = parser.parse_args()
    target = args.output or root / "kaggle_jobs" / f"unstructured_shortpose_{args.suite}_source_label.ipynb"
    build_notebook(root, target, suite=args.suite)
