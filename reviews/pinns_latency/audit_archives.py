"""Inspect the historical ZIPs and notebook-embedded source. No execution of project/model code."""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile


RELEASES = {
    "V1": ("443e1c1bb19f2b7b1a194c16fcbff03b1453b695", "unstructured_shortpose_v1_results.zip",
           "431ef3a5f4af33f05b9b1df9a2e369b001dd555696cdefa1705b35c7ad0c00a3", "v1_initial_diagnosis/"),
    "V2": ("39e10135bc747af14e8b126671be342a13d6501a", "unstructured_shortpose_v1_results_1.zip",
           "f8a033b2ab5e42c677584906095b1ef445d4888becf58d071cae531f9dfaa021", ""),
}
REPORT = "results/master_experiment_report.json"
NOTEBOOK = "kaggle_jobs/unstructured_shortpose_exp_v1.ipynb"
TRAINER = "src/engine/trainer.py"
RUNNER = "scripts/cloud_experiment_runner.py"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_file(repo, commit, path):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{path}"])


def check_release(repo, archives, release):
    commit, name, expected, subdir = RELEASES[release]
    raw = (archives / name).read_bytes()
    assert sha(raw) == expected
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        assert z.testzip() is None and len(z.namelist()) == 10
        members = z.namelist()
        master_bytes = z.read(REPORT)
    master = json.loads(master_bytes)
    assert master_bytes == git_file(repo, commit, REPORT)
    assert master_bytes == (repo / "results" / subdir / "master_experiment_report.json").read_bytes()
    notebook_bytes = git_file(repo, commit, NOTEBOOK)
    nb = json.loads(notebook_bytes)
    cells = ["".join(cell["source"]) for cell in nb["cells"]]
    payload = base64.b64decode(re.search(r'PAYLOAD_B64 = "([A-Za-z0-9+/=]+)"', cells[0])[1], validate=True)
    assert "execute_suites_from_samples(" in cells[-1]
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        files = {x.name: tar.extractfile(x).read() for x in tar.getmembers() if x.isfile()}
    sources = [k for k in files if k.startswith("src/") or k == RUNNER]
    assert all(files[k] == git_file(repo, commit, k) for k in sources)
    scripted = {}
    for k in sources:
        for f in ast.walk(ast.parse(files[k])):
            if isinstance(f, ast.FunctionDef) and any(ast.unparse(d) == "torch.jit.script" for d in f.decorator_list):
                scripted[k + ":" + str(f.lineno)] = f.name
    trainer = files[TRAINER].decode()
    func = next(f for f in ast.parse(trainer).body if isinstance(f, ast.FunctionDef) and f.name == "benchmark_inference_latency")
    sampler = "\n".join(trainer.splitlines()[func.decorator_list[0].lineno-1:func.end_lineno])
    assert [ast.literal_eval(v) for v in func.args.defaults] == ["Ours_Full", "cuda:0", 5, 25, 100]
    assert "load_state_dict" not in sampler and "torch.load" not in sampler and "manual_seed" not in sampler
    assert "mod = mod.to(dev).eval()" in sampler and "for n_agents in (1, 4, 8, 12)" in sampler
    assert 'benchmark_inference_latency("Ours_Full", device_str="cuda:0", seq_t=t)' in files[RUNNER].decode()
    batch = next(n.value for n in ast.walk(func) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
                 and len(n.value.keys) == 10)
    inputs = {ast.literal_eval(k): ast.unparse(v) for k,v in zip(batch.keys, batch.values)}
    grid = master["latency_cuda"]
    assert list(grid) == ["T=5", "T=10", "T=20"]
    for rows in grid.values():
        assert list(rows) == ["N=1", "N=4", "N=8", "N=12"]
        assert all(set(r) == {"mean_ms", "p50_ms", "p95_ms", "peak_vram_mb"} for r in rows.values())
    assert master["gpu_name"] == "Tesla T4" and master["gpu_count"] == 2
    return {
        "archive_name": name, "archive_sha256": sha(raw), "archive_bytes": len(raw), "members": members,
        "master_report_sha256": sha(master_bytes), "historical_git_release": commit,
        "master_bytes_identical_to_release_and_present_preserved_report": True,
        "historical_notebook_sha256": sha(notebook_bytes), "notebook_embedded_tar_sha256": sha(payload),
        "embedded_runner_and_src_py_files_matched_to_git": len(sources),
        "sources_sha256": {k:sha(files[k]) for k in sorted(sources)},
        "historical_sampler_sha256_utf8_no_terminal_newline": sha(sampler.encode()),
        "torchscript_functions": scripted,
        "input_constructors": inputs,
        "reported_full_model_parameters": master["summaries"]["PINNS_T5_Ours_Full"]["params"],
        "gpu_name": master["gpu_name"], "gpu_count_visible": master["gpu_count"], "latency_cuda": grid,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", type=Path)
    ap.add_argument("archives", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    versions = {r: check_release(args.repo, args.archives, r) for r in RELEASES}
    v1, v2 = versions.values()
    assert v1["historical_sampler_sha256_utf8_no_terminal_newline"] == v2["historical_sampler_sha256_utf8_no_terminal_newline"]
    assert v1["input_constructors"] == v2["input_constructors"] and not v1["torchscript_functions"]
    assert set(v2["torchscript_functions"].values()) == {"_jit_masked_gru_scan", "_jit_decoder_rollout"}
    out = {
        "versions": versions,
        "protocol": {
            "device": "cuda:0", "model": "Ours_Full", "fresh_model_instances_per_version": 3,
            "checkpoint_loaded": False, "explicit_latency_seed": False, "model_eval": True, "torch_no_grad": True,
            "hidden_dim": 128, "environment_dim": 64, "modes": 6, "future_steps": 30, "dt_s": 0.1,
            "sampling": "A single constant synthetic batch per (T,N). Every target row identical. No real frame, scene, target ID or motion/pose dataset is sampled.",
            "blocks_per_cell": 1, "warmups_per_block": 25, "timed_per_block": 100,
            "timed_calls_per_version": 1200, "warmup_calls_per_version": 300,
            "timer": "time.perf_counter host wall time around one full model call plus cuda.synchronize(cuda:0). Preceded by warmups, CUDA sync and reset_peak_memory_stats.",
            "timing_outputs": "numpy.mean, median, percentile(95) over single block, rounded to 0.001 ms; no division by batch size.",
            "memory": "torch.cuda.max_memory_allocated(device) / 2**20 (MiB), rounded to 0.01; peak allocated PyTorch tensor bytes after reset, including live allocations, not whole GPU process or incremental activation usage.",
            "exclusions": ["input tensor construction", "video decode and image detection", "association and tracking", "RTMPose", "feature construction and host-to-device copy", "model initialization", "training/backward"],
            "unavailable_from_result_archives": ["individual durations and their order", "separate repeated blocks within each cell", "between-run uncertainty", "timestamps and physical GPU identity for each block", "package/driver/clock manifest or determinism/precision switch manifest", "executed notebook hash bound into result"],
        },
        "paper_table4": "All four selected rows checked against both raw reports; budget headroom uses 100*(25 - V2 mean)/25.",
    }
    path = args.repo / "paper/ShortPose_Unstructured_Pedestrian_Paper.md"
    matched = []
    for line in path.read_text().splitlines():
        m = re.match(r'^\| \$T=(\d+)\$ \([^|]+\) \| \$N=(\d+)\$ \| (.+?) \|$', line)
        if not m:
            continue
        t,n,rest = m.groups()
        if len(rest.split(" | ")) != 6:
            continue
        left, right = [v["latency_cuda"]["T="+t]["N="+n] for v in (v1, v2)]
        actual = [float(re.search(r'[\d.]+', s)[0]) for s in rest.split(" | ")]
        required = [left["mean_ms"], right["mean_ms"], right["p50_ms"], right["p95_ms"],
                    right["peak_vram_mb"], round(100*(25-right["mean_ms"])/25, 1)]
        assert actual == required, (t, n, actual, required)
        matched.append([t,n])
    assert matched == [["5","1"],["5","12"],["10","12"],["20","12"]]
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
