# Replaying the matched PINNS weights

The verifier runs CPU inference from a checkout of this repository with Python, NumPy and PyTorch. Run Python without optimization so that verification assertions are active. The originating Kaggle workers used PyTorch 2.11.0+cpu. No training or GPU is required for replay.

Supply the following **original returned files** in an external directory. The repository contains aggregates but does not include individual forecasts, source annotations or model weights.

| Files | Origin |
|---|---|
| `cpu_T5.zip`, `matched_fusion_cpu_T5.zip` | Seed 42, Kaggle versions 373601953 and 373604843 |
| `cpu_T10.zip`, `matched_fusion_cpu_T10.zip` | Seed 42, Kaggle versions 373601970 and 373604882 |
| `matched_same_worker_T20.zip` | Seed 42, same-worker rerun, version 373610670. Keep its two inner files together. |
| `multiseed_s1024_all.zip`, `multiseed_s2025_all.zip`, `multiseed_s3407_all.zip`, `multiseed_s4096_all.zip` | Versions 373615911, 373615928, 373615947 and 373617529 |
| `pinns_matched_histories.npz` | Original cohort manifest for 1,482 anchors, or one rebuilt from the published PINNS labels and calibration. |

To rebuild the manifest from a PINNS data root, run the following from the checkout, with the paths set to your local directories. The cohort builder additionally uses OpenCV. The replay extracts the verified 623-window observation cache directly from `cpu_T5.zip`.

```sh
python -m scripts.audit_pinns_matched_histories "$PINNS_DATA" \
  --manifest "$ARCHIVES/pinns_matched_histories.npz" \
  --output "$ARCHIVES/cohort_aggregate.json"

python reviews/pinns_matched_multiseed/verify_and_pool.py \
  "$ARCHIVES" "$ARCHIVES/verified.json" \
  "$ARCHIVES/multiseed_s1024_all.zip" "$ARCHIVES/multiseed_s2025_all.zip" \
  "$ARCHIVES/multiseed_s3407_all.zip" "$ARCHIVES/multiseed_s4096_all.zip"
```

The verifier rebuilds inputs for each duration, checks fold identities, and loads all 240 fitted states for the full comparison. Seed 42 base-model archives store top-ranked errors. Seed 42 fusion archives and all four new seed archives additionally store the six candidate trajectories and their probabilities. The output contains the overall and fixed five-frame quality groups, individual seed estimates, and the registered five-frame primary contrast with site estimates, seed range and conditional paired-clip interval. These fields can be compared with `group_results`, `five_seed_by_site` and `overall_paired_contrasts` in [the published aggregate](../../results/pinns_matched_fusion_five_seed.json). A partial set of new archives can be checked separately. The primary result is populated only when all five seeds supply five-frame predictions.
