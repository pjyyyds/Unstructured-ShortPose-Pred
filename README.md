# Unstructured-ShortPose-Pred

**Unstructured Road Pedestrian Action & Trajectory Prediction under Short-Term Pose Observation**  
*(短时姿态观测下的非结构化道路行人动作轨迹预测)*

---

## Overview

This repository contains the modular PyTorch implementation, anti-leakage data pipeline, mathematical invariant test suite, and controlled multi-seed benchmark suite for short-horizon ($0.4\text{--}1.9\text{ s}$) pedestrian observable action and multimodal ground trajectory forecasting ($3.0\text{ s}$, $K=6$ modes) on non-channelized mixed-traffic roads (`PINNS` & `JTA` / `JRDB`).

### Key Methodological Contributions
1. **Explicit & Biomechanical Part-Decomposed Quality Gating (`M1` / `BPD-Gate` `M1p`)**: Suppresses noisy/occluded 2D skeleton negative transfer while preserving head-shoulder anticipatory turning cues when lower limbs are occluded by roadside vehicles or barriers, with exact bit-level identity fallback ($q_1 = 0 \Rightarrow g = 0, \mathbf{h}_p = \mathbf{0}$).
2. **Non-Lane Local Polyline & Radial-Approaching Directed `GATv2` (`M2`)**: Encodes irregular physical boundaries/portals ($\mathbf{P}_{\text{env}} \in \mathbb{R}^{16 \times 10 \times 11}$, distinguishing unmapped scenes $m_{\text{map}}=0$ from open space $m_{\text{map}}=1$) alongside asymmetric vehicle-pedestrian radial approaching speed $v_{\text{rad}}$.
3. **Stop-Gradient Observable Action Conditioning & Kinematic-Action Mode Calibration (`M3` / `ACMC` `M3p`)**: Predicts 5 observable dynamic events (`Start`, `Stop`, `Turn-Left`, `Turn-Right`, `Maintain` gated by label validity $m_{\text{act}}$), isolates continuous regression gradients via `stop_grad`, and calibrates Top-1 mode ranking (`ADE_1`, `NLL`, `Brier`).

---

## Repository Structure

```text
Unstructured-ShortPose-Pred/
├── CONTEXT_MEMORY.md              # Persistent architectural & audit log
├── src/
│   ├── data/                      # 10Hz/2.5Hz extractors, 6.0s buffer splits, Eq.(3)-(5) builders
│   ├── models/                    # Masked GRU, ST-GCN, BPD-Gate, Polyline+GATv2, ACMC, Baselines
│   ├── losses/                    # Eq.(10) shared-winner Smooth-L1 + Mode CE + m_act BCE
│   ├── metrics/                   # 3x3 bucket metrics & Shapiro-Wilk/Paired-t/Wilcoxon/Holm-Bonferroni
│   └── engine/                    # Deterministic multi-seed trainer & latency profiler
├── scripts/                       # Extraction, cloud/local orchestrators, 300-DPI figure plotter
├── kaggle_jobs/                   # Self-contained cloud GPU notebook
└── tests/
    └── test_invariants.py         # 12-item mathematical, gradient, FLOPs (<0.8G), and smoke tests
```

## Quick Verification

```bash
PYTHONPATH=. python3 tests/test_invariants.py
```
