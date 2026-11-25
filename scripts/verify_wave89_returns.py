"""Independent SHA-256 verification for Wave 8 & Wave 9 Kaggle returns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.verify_wave6_returns import verify_zip_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bidir-zip", type=Path, required=True)
    ap.add_argument("--hybrid-zip", type=Path, required=True)
    ap.add_argument("--unified-zip", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "bidirectional_pinns_jta_o2_5seed_verified.json": args.bidir_zip,
        "pinns_hybrid_action_events_5seed_verified.json": args.hybrid_zip,
        "unified_o2_full_5seed_and_t4_latency_verified.json": args.unified_zip,
    }
    for out_name, zp in mapping.items():
        v = verify_zip_manifest(zp)
        out_p = args.out_dir / out_name
        out_p.write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")
        print("Verified:", out_name, v["archive_name"], v["archive_sha256"][:16])


if __name__ == "__main__":
    main()
