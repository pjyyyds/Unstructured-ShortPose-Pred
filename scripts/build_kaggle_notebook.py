"""Pack verified local project into a clean, self-contained Kaggle GPU Jupyter Notebook."""

from pathlib import Path
import base64
import io
import json
import shutil
import tarfile


def build_notebook(project_root: Path, out_ipynb: Path) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in ("src", "tests", "scripts/cloud_experiment_runner.py"):
            full = project_root / rel
            for p in ([full] if full.is_file() else sorted(full.rglob("*.py"))):
                if "__pycache__" not in str(p):
                    tf.add(p, arcname=str(p.relative_to(project_root)))
    b64_payload = base64.b64encode(buf.getvalue()).decode("ascii")

    cell1 = f"""# Cell 1: Unpack verified project codebase, install GPU pose runtime, and run 12/12 unit tests
import base64, io, os, subprocess, sys, tarfile
from pathlib import Path

PROJ_DIR = Path('/kaggle/working/shortpose_proj') if Path('/kaggle/working').exists() else Path('/content/shortpose_proj')
PROJ_DIR.mkdir(parents=True, exist_ok=True)

PAYLOAD_B64 = "{b64_payload}"
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD_B64)), mode='r:gz') as tf:
    tf.extractall(PROJ_DIR)

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'onnxruntime-gpu', 'gdown'], check=True)
sys.path.insert(0, str(PROJ_DIR))
os.environ['PYTHONPATH'] = str(PROJ_DIR)

import torch
print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'GPU count:', torch.cuda.device_count(), flush=True)
for i in range(torch.cuda.device_count()):
    print(f'  GPU {{i}}:', torch.cuda.get_device_name(i), flush=True)

subprocess.run([sys.executable, str(PROJ_DIR / 'tests/test_invariants.py')], check=True)
"""

    cell2 = """# Cell 2: Stage RTMPose-m ONNX, PINNS (10Hz real video + homography), and JTA (2.5Hz) in /tmp
from pathlib import Path
from scripts.cloud_experiment_runner import stage_and_extract_all_datasets

PINNS_SAMPLES, JTA_SAMPLES, DATASET_DIAG = stage_and_extract_all_datasets(Path('/tmp/shortpose_raw'))
print('Dataset Diagnostics:', DATASET_DIAG, flush=True)
"""

    cell3 = """# Cell 3: Execute 5-seed controlled experiments (Suites A-E), Holm-Bonferroni stats, and export bundle
import shutil
from pathlib import Path
from scripts.cloud_experiment_runner import execute_suites_from_samples

OUT_ROOT = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/content/output')
BUNDLE_PATH = execute_suites_from_samples(
    PINNS_SAMPLES, JTA_SAMPLES, DATASET_DIAG, out_dir=OUT_ROOT, epochs_pinns=28, epochs_jta=24
)
rep_txt = (OUT_ROOT / 'results' / 'master_experiment_report.json').read_text(encoding='utf-8')
print('=== MASTER_REPORT_BYTES ===', len(rep_txt))
if (OUT_ROOT / 'shortpose_proj').exists():
    shutil.rmtree(OUT_ROOT / 'shortpose_proj', ignore_errors=True)
shutil.rmtree('/tmp/shortpose_raw', ignore_errors=True)
print('READY FOR DOWNLOAD:', BUNDLE_PATH)
try:
    from google.colab import files
    files.download(str(BUNDLE_PATH))
    files.download(str(OUT_ROOT / 'results' / 'master_experiment_report.json'))
except Exception as exc:
    print('Colab auto-download skipped:', exc)
"""

    def _mk_cell(src: str) -> dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in src.strip().splitlines()],
        }

    nb = {
        "cells": [_mk_cell(cell1), _mk_cell(cell2), _mk_cell(cell3)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_ipynb.parent.mkdir(parents=True, exist_ok=True)
    out_ipynb.write_text(json.dumps(nb, indent=2))
    print(f"Built {out_ipynb} ({out_ipynb.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    dst = root / "kaggle_jobs" / "unstructured_shortpose_exp_v1.ipynb"
    build_notebook(root, dst)
    stage_uploads = Path(
        "/Users/pengjiayu/.gemini/antigravity/brain/e1fb5464-dcdf-4c38-a8ef-b854fa310868/browser_sandbox/uploads"
    )
    stage_uploads.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dst, stage_uploads / "unstructured_shortpose_exp_v1.ipynb")
    print("Staged upload copy to:", stage_uploads / "unstructured_shortpose_exp_v1.ipynb")
