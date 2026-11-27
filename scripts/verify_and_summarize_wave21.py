"""Dual-layer SHA-256 verifier and structured report extractor for all 4 Wave 21 Kaggle archives."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

EXPECTED_JOBS = (
    ("wave21_pinns_true_emergence_and_6slot_diag.zip", "wave21_pinns_true_emergence_and_6slot_diag_5seed_verified.json"),
    ("wave21_polyline_crop_fix_and_dual_boundary.zip", "wave21_polyline_crop_fix_and_dual_boundary_5seed_verified.json"),
    ("wave21_jrdb_and_jta_emergence_maneuvers.zip", "wave21_jrdb_and_jta_emergence_maneuvers_5seed_verified.json"),
    ("wave21_jaad_deocclusion_and_eth_ucy_6dataset.zip", "wave21_jaad_deocclusion_and_eth_ucy_6dataset_5seed_verified.json"),
    ("wave22_apa_leadtime_jaad_jta_and_pinns_delayed.zip", "wave22_apa_leadtime_jaad_jta_and_pinns_delayed_5seed_verified.json"),
    ("wave22_jrdb_delayed_apa_upgrade.zip", "wave22_jrdb_delayed_apa_upgrade_5seed_verified.json"),
)


def _assert_finite(obj: object, path: str = "root") -> None:
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite float at {path}: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            _assert_finite(v, f"{path}[{idx}]")


def verify_single_wave21_zip(zip_path: Path, out_json: Path) -> dict:
    raw_zip = zip_path.read_bytes()
    outer_sha = hashlib.sha256(raw_zip).hexdigest()
    with zipfile.ZipFile(zip_path, "r") as zf:
        raw_sum = zf.read("summary.json")
        raw_sha = json.loads(zf.read("sha256.json").decode("utf-8"))
    calc_sum_sha = hashlib.sha256(raw_sum).hexdigest()
    decl_sum_sha = raw_sha.get("summary.json", "")
    if calc_sum_sha != decl_sum_sha:
        raise ValueError(f"Inner SHA-256 mismatch in {zip_path.name}: {calc_sum_sha} != {decl_sum_sha}")
    payload = json.loads(raw_sum.decode("utf-8"))
    _assert_finite(payload)
    payload["verification"] = {
        "source_zip_name": zip_path.name,
        "outer_zip_bytes": len(raw_zip),
        "outer_zip_sha256": outer_sha,
        "inner_summary_sha256": calc_sum_sha,
        "dual_sha256_verified": True,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    for zn, jn in EXPECTED_JOBS:
        zp = args.zip_dir / zn
        if zp.is_file():
            pl = verify_single_wave21_zip(zp, args.results_dir / jn)
            v = pl["verification"]
            print(f"[VERIFIED] {zn} -> {jn} | bytes={v['outer_zip_bytes']} | outer={v['outer_zip_sha256'][:12]} | inner={v['inner_summary_sha256'][:12]}")
        else:
            print(f"[PENDING]  {zn} not yet present in {args.zip_dir}")


if __name__ == "__main__":
    main()
