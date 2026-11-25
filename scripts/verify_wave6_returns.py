"""Independent SHA-256 & schema verification for Wave 6 Kaggle archives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


def verify_zip_manifest(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as outer:
        assert outer.testzip() is None, f"Corrupted zip: {zip_path}"
        inner_zips = [n for n in outer.namelist() if n.endswith(".zip")]
        if inner_zips:
            import io
            raw_inner = outer.read(inner_zips[0])
            zf = zipfile.ZipFile(io.BytesIO(raw_inner))
            assert zf.testzip() is None
            arch_name = inner_zips[0]
            arch_sha = hashlib.sha256(raw_inner).hexdigest()
        else:
            zf = outer
            arch_name = zip_path.name
            arch_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        manifest = json.loads(zf.read("sha256.json").decode())
        for name, exp_sha in manifest.items():
            got = hashlib.sha256(zf.read(name)).hexdigest()
            if got != exp_sha:
                raise ValueError(f"SHA-256 mismatch in {arch_name}:{name}: {got} != {exp_sha}")
        rep = json.loads(zf.read("report.json").decode())
    return {
        "archive_name": arch_name,
        "archive_sha256": arch_sha,
        "outer_download_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        "manifest_sha256": manifest,
        "report": rep,
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--occ-zip", type=Path, required=True)
    ap.add_argument("--conf-zip", type=Path, required=True)
    ap.add_argument("--graph-zip", type=Path, required=True)
    ap.add_argument("--horizon-zip", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "pinns_o2_occlusion_stress_5seed_verified.json": args.occ_zip,
        "pinns_conformal_and_events_5seed_verified.json": args.conf_zip,
        "o2_hetero_graph_pinns_and_dut_5seed_verified.json": args.graph_zip,
        "pinns_horizon_and_speed_regimes_5seed_verified.json": args.horizon_zip,
    }
    for out_name, zp in mapping.items():
        v = verify_zip_manifest(zp)
        out_p = args.out_dir / out_name
        out_p.write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")
        print("Verified:", out_name, v["archive_sha256"][:16])


if __name__ == "__main__":
    main()
