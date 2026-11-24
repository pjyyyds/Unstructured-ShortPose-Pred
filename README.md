# Unstructured-ShortPose-Pred

**Joint Pedestrian Action and Trajectory Prediction on Unstructured Roads under Short-Term Variable-Quality Pose Observation**  
*短时姿态观测下的非结构化道路行人动作与轨迹联合预测*

---

## Highlights

- **Ultra-Lightweight Edge Architecture**: **0.471 M** parameters (`< 2.5 M` budget), **0.409–0.798 GFLOPs** at $N=12$ concurrent targets (`< 0.80 GFLOPs` budget), **14.49 ms** end-to-end latency and **31.5 MB** peak VRAM on `NVIDIA Tesla T4` (**zero future skeleton prediction**).
- **Core Algorithmic Modules**:
  1. **Balanced Quality-Bottleneck & Biomechanical Part-Decomposed Gate (`M1` / `BPD-Gate`)**: Eliminates occluded-pose negative transfer (`-34.36%` error vs equal-weight pose concat on `JTA` Heavy, Holm $p = 8.62 \times 10^{-24}$) and beats `MCENET`, `HST`, and `Social-Transmotion` across all observation windows $T \in \{5, 10, 20\}$ on `PINNS`.
  2. **Ego-Preserving Trunk & Null-Sink Directed GATv2 (`M2`)**: Couples $20\text{ m}$ non-lane local polylines with $10\text{ m}$ radial-approaching-speed ($v_{\text{rad}}$) heterogeneous graphs while allowing non-conflicting bystander attention to decay naturally via a learnable Null-Sink token (`-14.77%` error reduction from V1 to V2).
  3. **Stop-Grad 5-Class Action Conditioning & Kinematic Mode Calibration (`M3` / `ACMC`)**: Preserves objective trajectory-derived action `Macro-F1` (`+5.96%` to `+6.44%` over end-to-end backprop) with TorchScript C++ fused rollout.
- **Benchmark Calibration Discovery**: Identifies and corrects the ground-contact foot-anchor `(cx, y_bottom)` coordinate convention in `PINNS`, doubling `RTMPose-m` keypoint confidence (`0.20–0.29 -> 0.47–0.58`) across 47 multi-country roadside sequences (`7,283` zero-leakage windows).

---

## Repository Structure

```text
├── src/
│   ├── data/        # Anti-leakage 6.0s splitter, PINNS 10Hz RTMPose-m extractor, JTA 2.5Hz builder
│   ├── models/      # M0-M3, BPD-Gate (M1p), Null-Sink GATv2, Stop-Grad ACMC (M3p), MCENET/HST/Social-Transmotion
│   ├── losses/      # Shared-winner Smooth-L1 + distance soft-ranked mode CE + m_act-masked BCE
│   ├── metrics/     # 3x3 bucket metrics, Shapiro-Wilk -> Paired-t/Wilcoxon + Holm-Bonferroni tester
│   └── engine/      # Deterministic multi-GPU 5-seed trainer & CUDA latency profiler
├── scripts/         # Cloud GPU orchestrator, Kaggle notebook builder, 300-DPI figure & table generators
├── tests/           # 13 rigorous mathematical invariant, zero-leakage, gradient & FLOPs unit tests
├── results/         # V2 242-run 5-seed JSON/NPZ/Markdown tables + archived V1 diagnostic baseline
├── checkpoints/     # Trained seed-42 PyTorch weights (M0, M1, M2, M3, Ours_Full)
├── docs/figures/    # 5 publication-grade 300-DPI figures
├── paper/           # Complete academic paper manuscript (ShortPose_Unstructured_Pedestrian_Paper.md)
└── reviews/         # Executive morning review report (PROJECT_REVIEW_SUMMARY.md)
```

## Quick Verification

```bash
PYTHONPATH=. python3 tests/test_invariants.py
PYTHONPATH=. python3 scripts/plot_paper_figures.py results/master_experiment_report.json docs/figures
PYTHONPATH=. python3 scripts/generate_manuscript_and_review.py results/master_experiment_report.json results/EMPIRICAL_BENCHMARK_REPORT.md
```
