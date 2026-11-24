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
- On the original split, constant velocity from the final two positions scored 0.5041 m ADE and the explicit pose gate (M1) scored 0.4783 m. The US site contributes 316 of 497 windows and is the only site where M1 improves on constant velocity. Giving all four sites equal weight yields 0.3897 m for constant velocity and 0.4288 m for M1. A paired bootstrap of 21 clips gives a window-weighted M1-minus-constant-velocity difference of −0.0258 m (95% percentile interval −0.1297 to +0.0713 m). [Cluster bootstrap](results/pinns_clip_bootstrap.json)
- In a separate leave-one-site-out study, each person track contributed disjoint, complete five-history/30-future frame windows. Across 1,482 windows, five-point constant velocity and ridge regression trained on the other three sites had four-site macro ADE of 0.5557 and 0.7189 m. The pose networks still require evaluation with this protocol. [Fold definitions, counts and errors](results/pinns_heldout_sites.json)

On the same 1,482 complete windows, extrapolation from the last two observations had lower ADE and FDE at each of the four sites than the five-point fit or the coordinatewise median of four interframe velocities. Four-site macro ADE/FDE was 0.4839/1.1046, 0.5557/1.2103 and 0.5555/1.2104 m respectively. These are descriptive comparisons on previously examined locations. [Per-site and per-clip results](results/pinns_velocity_estimators.json)

A prespecified single-coefficient speed reduction fit on the three training locations raised four-site macro ADE/FDE to 0.6068/1.2338 m. It improved on unscaled two-point velocity only at the US location. [Coefficients and held-out errors](results/pinns_velocity_scale.json)

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
python3 -m scripts.cluster_pinns_errors
python3 -m scripts.benchmark_heldout_sites "$PINNS_ROOT"
python3 -m scripts.compare_pinns_velocity_estimators "$PINNS_ROOT"
python3 -m scripts.fit_pinns_velocity_scale "$PINNS_ROOT"
python3 -m scripts.count_forward_ops
```

### Neural held-out site pilot

With a CUDA runtime and Internet access, run the following from the repository root. The runner verifies both archives, applies frozen RTMPose to the five observed frames, fits M0 and M1 independently on the other three sites, and saves per-window test errors with clip, person and time identifiers. The initial seed is 42 and the final epoch is fixed at 28. Each method uses the same 1,482 windows. The five-point constant-velocity baseline is the primary prespecified reference. The two-point baseline is reported as a separate sensitivity check.

```bash
python -m scripts.run_heldout_cloud --scratch-dir /tmp/shortpose_heldout_cache --output-dir /kaggle/working/shortpose_heldout
```

The final archive is `shortpose_heldout.zip` in the output directory. On Colab, choose an output path under `/content`. The scratch directory keeps verified downloads and per-clip extraction checkpoints. Reusing the same scratch and output directories resumes completed clips and models.
