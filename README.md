# Unstructured-ShortPose-Pred

**Joint Pedestrian Action and Trajectory Prediction on Unstructured Roads under Short-Term Variable-Quality Pose Observation**  
*短时姿态观测下的非结构化道路行人动作与轨迹联合预测*

---

## Benchmark status

The archived V1 and V2 Kaggle runs each cover 50 configurations and 242 training runs. In each version, 46 configurations use five seeds and four sensitivity settings use three. V2 reused the split on which V1 had already been inspected. Independent confirmation and the proposed JRDB evaluation are pending. Under the original proposal, the data do not establish A (improvement at all three window lengths) or B (improvement in both short-window quality groups). C applies to the comparisons without a net benefit, including M1 versus its implicit-gate control at five frames.

- On PINNS at five frames, mean top-mode displacement error (ADE, lower is better) was 0.5295 m for the full model and 0.4856 m for the motion-only baseline. On JTA the corresponding errors were 1.8637 m and 1.6640 m.
- There is a narrower result for pose fusion. In 193 JTA test windows with injected heavy noise, M1 reduced mean ADE from 2.3063 m (M1a) to 1.5138 m. The motion-only baseline reached 1.4397 m. M1a adds the motion and available pose embeddings elementwise with equal weight, then applies LayerNorm. M1 scales the pose residual by a quality-dependent gate.
- PINNS contains 47 clips from four sites. The 7,283 windows include 497 test windows at the same sites. JTA uses a separate within-dataset training run and simulated pose corruption. Three external model names in the tables identify local adaptations. Official implementations and independent camera sites still require evaluation.
- A crop check used one frame from each of three PINNS sites, with 13 person boxes of height at least 60 pixels. Moving the crop up by half its height increased model confidence. No manually labeled joint locations were used to measure localization accuracy.
- On one Tesla T4 the full predictor averaged 14.488 ms for a batch of 12 at five frames (p95 15.110 ms). The inputs were prebuilt GPU tensors. Video decoding, tracking, pose extraction and preprocessing require separate timing. At 20 frames and 12 targets, the training project's operator count reported 798,230,016 FLOPs. Including elementwise floating-point additions, subtractions and multiplications gives 807,563,628, exceeding the proposed 0.80 GFLOP limit under that convention.

## Additional PINNS checks

These analyses use existing public PINNS recordings. The original 484 training runs are archived separately.

- Reconstructing the archived split produced all 497 windows. The saved M0 checkpoint reproduced its seed-42 cloud ADE/FDE within 0.000001 m. All five history positions are available in every window. Only 233 windows have a labeled position at 3.0 s. The archived FDE uses the last available future position in each window. M0 includes the moving/static target-type input. [Split reconstruction and results](results/pinns_motion_audit.json)
- In 496 of the 497 archived windows, the same pedestrian had an annotation during the 15 sampling steps before the five-frame input. In the site-held-out cohort, 1,033 of 1,482 inputs followed at least 35 consecutive annotated steps. Three US clips use person frame numbers spanning 0 through the video frame count minus one, while the frozen extractor pairs label 1 with video frame 0. They contribute 597 of the 1,482 windows. [Track history and video index audit](results/pinns_observation_audit.json)
- On the original split, constant velocity from the final two positions scored 0.5041 m ADE and the explicit pose gate (M1) scored 0.4783 m. The US site contributes 316 of 497 windows and is the only site where M1 improves on constant velocity. Giving all four sites equal weight yields 0.3897 m for constant velocity and 0.4288 m for M1. A paired bootstrap of 21 clips gives a window-weighted M1-minus-constant-velocity difference of −0.0258 m (95% percentile interval −0.1297 to +0.0713 m). [Cluster bootstrap](results/pinns_clip_bootstrap.json)
- In a separate leave-one-site-out study, each person track contributed disjoint, complete five-history/30-future frame windows. Across 1,482 windows, five-point constant velocity and ridge regression trained on the other three sites had four-site macro ADE of 0.5557 and 0.7189 m. The neural pilot below uses the same protocol. [Fold definitions, counts and errors](results/pinns_heldout_sites.json)

On the same 1,482 complete windows, extrapolation from the last two observations had lower ADE and FDE at each of the four sites than the five-point fit or the coordinatewise median of four interframe velocities. Four-site macro ADE/FDE was 0.4839/1.1046, 0.5557/1.2103 and 0.5555/1.2104 m respectively. These are descriptive comparisons on previously examined locations. [Per-site and per-clip results](results/pinns_velocity_estimators.json)

A prespecified single-coefficient speed reduction fit on the three training locations raised four-site macro ADE/FDE to 0.6068/1.2338 m. It improved on unscaled two-point velocity only at the US location. [Coefficients and held-out errors](results/pinns_velocity_scale.json)

The original scenario model received neighbors in 480 of 497 archived test windows. Its 2,724 directed edges comprised 2,444 pedestrian edges and 280 car edges. All edge tensors, masks and type vectors were reconstructed from observed data. The release distinguishes people and cars at source. [Input coverage and temporal reconstruction](results/pinns_neighbor_inputs.json)

On the outdoor JRDB training recording Huang Lane, projecting 6,430 person positions into the panorama gave a median horizontal discrepancy of 8.53 pixels from the matching box center. HST supplied 33-joint 3D records for 554 of these person-frame pairs. The derived tracks run at 2.5 Hz and the joint records lack image-plane confidence. The planned 10 Hz RTMPose evaluation still requires suitable source frames and annotations. [Outdoor JRDB development check](reviews/jrdb_development/README.md)

The five-frame action development probe fits three regularized logistic heads to motion, motion plus final-frame pose, and motion plus final-frame pose and observed joint changes. The five cross-validation folds group whole clips, with all four known locations represented in each fold. The program records held-out probabilities, class supports, AP and fixed-threshold F1. Run `python -m scripts.evaluate_pinns_action_probes "$PINNS_ROOT" "$PILOT_DIR/pose.npz" --output action_probes.json` from the repository root.

## Project files

- [V2 benchmark tables](results/EMPIRICAL_BENCHMARK_REPORT.md) and [V1 archive](results/v1_initial_diagnosis/EMPIRICAL_BENCHMARK_REPORT.md)
- [Review and proposal criteria](reviews/PROJECT_REVIEW_SUMMARY.md) and [decision history](CONTEXT_MEMORY.md)
- [Prediction models](src/models/), [data preparation](src/data/), [evaluation](src/metrics/) and [invariant tests](tests/test_invariants.py)
- [V2 checkpoints](checkpoints/), [figures](docs/figures/) and [research manuscript](paper/ShortPose_Unstructured_Pedestrian_Paper.md)

## Quick Verification

```bash
PYTHONPATH=. python3 tests/test_invariants.py
PYTHONPATH=. python3 scripts/plot_paper_figures.py results/master_experiment_report.json docs/figures
PYTHONPATH=. python3 scripts/generate_manuscript_and_review.py results/master_experiment_report.json results/EMPIRICAL_BENCHMARK_REPORT.md
```

The five checked-in PNGs were generated with Matplotlib 3.9.4 at 300 DPI. The supplemental checks use Python, NumPy, SciPy, OpenCV and PyTorch. The arithmetic count below was verified with PyTorch 2.10.0. From the repository root, point `PINNS_ROOT` at the extracted dataset directory containing `labels/` and `videos/`. [PINNS dataset and download instructions](https://github.com/Songan-Lab/PINNS)

```bash
export PINNS_ROOT="/path/to/extracted/PINNS"
python3 -m scripts.audit_pinns_short_window "$PINNS_ROOT"
python3 -m scripts.audit_pinns_type_input "$PINNS_ROOT"
python3 -m scripts.audit_pinns_observations "$PINNS_ROOT"
python3 -m scripts.audit_pinns_projection "$PINNS_ROOT" --output results/pinns_projection_audit.json
python3 -m scripts.cluster_pinns_errors
python3 -m scripts.benchmark_heldout_sites "$PINNS_ROOT"
python3 -m scripts.compare_pinns_velocity_estimators "$PINNS_ROOT"
python3 -m scripts.fit_pinns_velocity_scale "$PINNS_ROOT"
python3 -m scripts.count_forward_ops
```

### Neural held-out site pilot

With a CUDA runtime and Internet access, run the following from the repository root. The runner verifies both archives, applies frozen RTMPose to the five observed frames, fits M0 and M1 independently on the other three sites, and saves per-window test errors with clip, person and time identifiers. The initial seed is 42 and the final epoch is fixed at 28. Each method uses the same 1,482 windows. Before inspecting the neural results, we fixed M1 minus M0 as the primary paired comparison. Five-point and two-point constant velocity are secondary comparisons.

```bash
python -m scripts.run_heldout_cloud --scratch-dir /tmp/shortpose_heldout_cache --output-dir /kaggle/working/shortpose_heldout
```

The Kaggle pilot completed all eight fits. With equal weight per site, ADE/FDE in metres was 0.5044/1.0896 for motion-only M0, 0.5463/1.1638 for quality-gated pose M1, 0.5557/1.2103 for five-point velocity and 0.4839/1.1046 for two-point velocity. The M1 minus M0 ADE difference was +0.0419 m (within-site clip-bootstrap 95% interval +0.0230 to +0.0620 m). The US site was the only one with a lower mean ADE for M1 than M0. The 47 clips come from four previously examined sites, and the intervals condition on one seed. The verified weights reproduce each window's cloud ADE and FDE on CPU within 0.000003 m. [Checkpoint replay](results/pinns_heldout_replay.json) · [Paired comparisons](results/pinns_heldout_paired.json)

The frozen pose-quality rules assigned 409 windows to clear, 691 to partial and 382 to heavy or low quality. Within each bucket, the four-site mean M1-minus-M0 ADE/FDE differences were +0.0135/+0.0152, +0.0375/+0.0682 and +0.0776/+0.1446 m, respectively. These groups use keypoint availability, confidence and relative box height. The dataset supplied no manual physical occlusion labels for this comparison. [Quality definitions, per-site counts and errors](results/pinns_heldout_pose_quality.json)

Applying the same event rules to annotated and forecast trajectories gave four-site macro-F1/mAP of 0.2376/0.3709 for M0 and 0.2419/0.3795 for M1. These scores describe events implied by six probability-weighted trajectory modes. Neither network trained an action head in this pilot. [Trajectory-implied event scores](results/pinns_heldout_implied_actions.json)

With the four trained M1 checkpoints fixed, masking every observed joint raised the four-site macro ADE by 0.0130 m and FDE by 0.0230 m. Reversing the order of the five observed pose frames while keeping motion and quality fixed changed macro ADE by less than 0.00002 m. All 1,482 windows retained the same top-mode index under the order reversal. These interventions assess input sensitivity for seed 42. [Matched fixed-weight controls](results/pinns_heldout_pose_input_controls.json)

To reproduce the checkpoint verification, run `python -m scripts.verify_heldout_bundle /path/to/shortpose_heldout.zip "$PINNS_ROOT" --output replay.json`. With the archive extracted, run `python -m scripts.analyze_heldout_pairs /path/to/folds "$PINNS_ROOT" --output paired.json`.

The runner writes `shortpose_heldout.zip` to the output directory. The scratch directory keeps verified downloads and per-clip extraction checkpoints. Reusing the same scratch and output directories resumes completed clips and models.

### Five-seed extension

The extension reuses the eight verified seed-42 fits and schedules 32 more across the four sites, two models and seeds 1024, 2025, 3407 and 4096. The reported four-site results above use seed 42. Training requires a CUDA device and the pilot's PyTorch build (`2.11.0+cu130`). Supply the official label/calibration directory with its video inventory, the extracted seed-42 bundle containing `pose.npz` and `folds/`, and a separate writable output directory. The first command verifies the inputs on CPU. The second resumes matching checkpoints and writes `shortpose_heldout_five_seeds.zip` when all 40 fits are present. [Five-seed runner](scripts/run_heldout_seed_sensitivity.py)

```bash
python -m scripts.run_heldout_seed_sensitivity --data-root "$PINNS_ROOT" --pilot-dir "$PILOT_DIR" --check-only
python -m scripts.run_heldout_seed_sensitivity --data-root "$PINNS_ROOT" --pilot-dir "$PILOT_DIR" --output-dir "$SEED_OUTPUT_DIR"
```

### Test sensitivity to a one-frame image offset

After extracting the pilot's `shortpose_heldout.zip`, point `PILOT_DIR` at the directory containing `pose.npz` and `folds/`, and `RTMPOSE_ONNX` at the original RTMPose model. The first command re-extracts 2,985 person observations in three US videos one decoded video frame later. It checks sampled zero-offset batches against the frozen cache on the same ONNX execution providers. Choose a new output cache path.

```bash
python -m scripts.cache_pinns_frame_offset "$PINNS_ROOT" "$PILOT_DIR/pose.npz" "$RTMPOSE_ONNX" shifted_pose.npz
python -m scripts.evaluate_pinns_frame_offset "$PINNS_ROOT" "$PILOT_DIR" shifted_pose.npz --output frame_offset_evaluation.json
```

The second command replays all four saved M1 checkpoints and verifies each window's archived errors. It then measures the test-input change on 597 US windows and reports the three clips individually, the US fold and the equal-site mean. Model weights stay fixed. Estimating the effect on training requires a separate refit.
