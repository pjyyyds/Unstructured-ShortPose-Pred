"""Run the registered causal PINNS image studies inside a fresh Kaggle GPU process."""

import argparse
import hashlib
from pathlib import Path

import torch

from scripts.cloud_experiment_runner import (
    execute_suites_from_samples,
    stage_and_extract_all_datasets,
    stage_pinns_data,
)
from src.data.pinns_bev import SOURCE_LABEL
from src.data.pinns_causal_input import (
    FULL_WINDOW_SHA256,
    MAP_SHA256,
    PROTOCOL,
    WINDOW_SHA256,
    attach_causal_image,
)


def main(map_path: Path, suite: str) -> Path:
    output = Path("/kaggle/working")
    if not output.is_dir() or not torch.cuda.is_available():
        raise RuntimeError("Use a Kaggle GPU notebook with Internet enabled.")
    if hashlib.sha256(map_path.read_bytes()).hexdigest() != MAP_SHA256:
        raise ValueError("The private causal image input differs from the registered file.")
    if suite == "paired":
        from reviews.pinns_map_axis_comparison.neural.run_neural import OUTPUT_ZIP, run

        samples, audit = stage_pinns_data(Path("/tmp/shortpose_pinns_causal_pair"))
        aligned = attach_causal_image(samples, map_path)
        print("PINNS extraction:", audit, flush=True)
        pair_output = output / "pinns_neural_image_coordinates"
        run(aligned, pair_output)
        result = pair_output / OUTPUT_ZIP
    elif suite == "full":
        scratch = Path("/tmp/shortpose_source_label_raw")
        samples, jta, diagnostics = stage_and_extract_all_datasets(
            scratch, cache_dir=scratch / "cache", bev_convention=SOURCE_LABEL
        )
        aligned = attach_causal_image(samples, map_path)
        diagnostics["pinns"].update(
            map_inputs=PROTOCOL,
            map_sha256=MAP_SHA256,
            window_sha256=WINDOW_SHA256,
            full_window_sha256=FULL_WINDOW_SHA256,
        )
        result = execute_suites_from_samples(
            aligned,
            jta,
            diagnostics,
            out_dir=output,
            epochs_pinns=28,
            epochs_jta=24,
            bev_convention=SOURCE_LABEL,
        )
        if not (result.parent / "results" / "master_experiment_report.json").is_file():
            raise RuntimeError("Missing the full suite's master report")
    else:
        raise ValueError(f"Unknown Kaggle suite: {suite}")
    if not result.is_file():
        raise RuntimeError(f"Missing the expected result: {result}")
    print("Kaggle result:", result, "bytes:", result.stat().st_size, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-map", type=Path, required=True)
    parser.add_argument("--suite", choices=("paired", "full"), required=True)
    args = parser.parse_args()
    main(args.input_map, args.suite)
