# Orthogonal-Equivariant Pedestrian Action and Trajectory Forecasting on Unstructured Roads Under Short-Term Variable-Quality Pose Observations

**Jiayu Peng and Danya Yao, *Member, IEEE***  
*Department of Automation, Tsinghua University, Beijing 100084, China*

---

## Abstract

Unmanned delivery and shuttle vehicles operating in campuses, parks, and residential streets face three coupled forecasting challenges: short causal observation windows ($0.4\text{ s}$, 5 frames at $10\text{ Hz}$) upon pedestrian emergence, variable-quality 2D/3D skeletal detections under roadside occlusion, and lane-free mixed pedestrian-vehicle interactions. While clean skeletal cues aid trajectory prediction on synthetic benchmarks, when and why short-term poses, passive bird's-eye-view (BEV) road contours, heterogeneous neighbor graphs, and observable dynamic action heads improve **same-decoder preferred (`Top-1`)** ground trajectories across unseen cameras remains rigorously under-tested. We present an $O(2)$ reflection- and rotation-equivariant multi-modal forecasting framework (`O2-M0` through `O2-M3`, down to $20,739$ parameters and $10.50\text{ MFLOPs}$) and conduct a strictly controlled, cross-scene five-seed empirical audit across six benchmarks (`PINNS` 4-country leave-one-site-out with $1,482$ non-overlapping windows, `JRDB` 9-location leave-one-location-out with $5,836$ windows, `JTA` $960$ windows, `DUT` $3,066$ mixed-traffic windows, `IMPTC`, and `DAIR-V2X`). Our audit establishes five principal findings, reporting positive gains, null intervals, and mechanistic failure modes on equal footing: (1) Canonicalizing short-term motion into a local Frenet frame with inner-fold low-displacement constant-velocity (`CV`) anchoring lowers 4-site equal `Top-1 ADE_1 / FDE_1` on `PINNS` from `0.4839 / 1.1046 m` (`Last-2 CV`) to **`0.4522 / 1.0194 m`** (`−6.03 / −15.62 cm` on `449` first-sampled windows); however, on the `323` first-sampled low-displacement (`< 0.5 m`) windows, parameter-free `Last-2 CV` (`0.3545 / 0.8491 m`) remains unrefuted—even after rank-3 `3×3` homography verification (`6,930 / 7,410` nonzero coronal facing frames, `2.24 µm` median footpoint residual) and `< 0.5 m` regime-isolated training (`0.3538 / 0.8462 m`, 95% clip bootstrap CI vs. `CV` `[−0.20, +0.26] cm` crossing zero). (2) A `2 × 2` (`quality vector q × speed suppression`) factorial across $7,318$ `PINNS + JRDB` windows proves that unconstrained trajectory-supervised `MLP([h_m, h_p, q])` gating exhibits negative rank correlation with valid-joint ratio ($\rho = -0.188$ on `PINNS`, $-0.218$ on `JRDB`) due to high-speed hidden-state confounding, whereas variable-quality training (`−2.41` to `−4.28 cm` vs. clean-only training), `m_pose = 0` hard shutoff (`−3.27` to `−3.68 cm` under full pose dropout), and high-speed residual damping (`−0.82 cm` on `JRDB`) provide the true robustness gains. (3) Centering `20 m` passive BEV contours at true pedestrian world coordinates (`pt0_ego`) reveals a `2.36%` ground-truth crossing rate (`4.19%` on `835` map-eligible windows) and a `+1.58 cm` `Top-1 ADE_1` penalty (`O2-M2a`), whereas `10 m` directed neighbor graphs with radial approaching rate `v_rad` reduce 6-mode `minFDE_6` to **`0.6513–0.6539 m`** on `PINNS` and **`0.3078 m`** (`−44.3%` vs. `CV`) on `JRDB`. (4) Five-class observable action supervision (`Start, Stop, Turn-Left, Turn-Right, Maintain`) significantly lowers `Top-1 ADE_1` on `JRDB` (`−0.80 cm`, 95% location CI `[−1.57, −0.20] cm`) and, via `Stop-Grad` conditioning (`M3`), mitigates 3.0 s endpoint drift on `< 0.5 m` `PINNS` windows (`−0.78 cm` vs. end-to-end `M3b`). (5) **Without retraining after calibration**, fixed-predictor split-conformal tubes achieve **`90.60% / 94.94%`** (`PINNS`) and **`88.37% / 94.48%`** (`JRDB`) within-scene Union-6 coverage at `90% / 95%` nominal levels while shrinking safety tube radii by **`36.4%–46.9%`** vs. `CV`, and quantify `86.48% / 91.74%` (`12` cross-country pairs) and `86.77% / 92.81%` (`72` cross-location pairs) coverage under strict geographic shift.

**Index Terms**—Pedestrian trajectory prediction, unstructured mixed traffic, short-term pose observation, $O(2)$ equivariance, quality-aware fusion, split-conformal calibration.

---

## I. Introduction

Low-speed autonomous delivery, shuttle, and sanitation vehicles have rapidly expanded from structured arterial roads into shared-right-of-way spaces such as university campuses, industrial parks, residential compounds, and urban plazas [1]–[3]. Unlike signalized multi-lane roadways with painted crosswalks and rigid curb channelization, unstructured mixed-traffic environments exhibit three simultaneous operational constraints:
1. **Short-Horizon Emergence Under Occlusion**: Parked vehicles, roadside vegetation, and building corners frequently occlude pedestrians until they step into the active field of view, leaving roadside or onboard trackers with as little as $0.4\text{ s}$ ($T = 5$ frames at $10\text{ Hz}$ or $T = 2$ steps at $2.5\text{ Hz}$) of causal observation before planning must commit.
2. **Variable-Quality Skeletal Cues**: While 2D/3D human keypoints can in principle disambiguate body orientation and gait initiation before finite-difference displacement accumulates [14], [15], real detector outputs degrade continuously with camera distance, motion blur, and partial limb truncation.
3. **Weak Geometric Channelization and Heterogeneous Conflicts**: Absence of lane markings yields frequent trajectory crossings among pedestrians, cyclists, and cars, creating genuine multi-modal intent (`Start, Stop, Turn-Left, Turn-Right, Maintain`) from identical short histories.

Despite rapid progress in pose-assisted trajectory transformers [14], [15], vectorized map encoders [6], [13], and action-conditioned decoders [8], [16], three methodological confounds often obscure empirical claims in short-horizon pedestrian forecasting:
- **Overlapping Sliding Windows and Mixed Decoders**: Evaluating overlapping sliding windows drawn from shared clips inflates statistical significance due to up to $26/30$-frame future target autocorrelation, while comparing probability-weighted (`Bayes`) expectations of a multi-modal model against single-path (`Top-1`) baselines conflates estimator shrinkage with model accuracy [4].
- **Coordinate-Frame Non-Invariance and Silent Feature Bugs**: Global $(X, Y)$ world coordinates rotate arbitrarily across camera installations, causing severe cross-site mode collapse; meanwhile, subtle coordinate bugs—such as mistaking a handedness boolean for a $3\times 3$ homography or centering local map polylines at an ego-subtracted $(0, 0)$ origin—can silently invalidate physical ablations unless guarded by runtime geometric assertions.
- **Post-Calibration Retraining in Conformal Safety Claims**: Calibrating conformal quantiles on a held-out site and subsequently retraining the predictor on all sites violates exchangeability with the deployed predictor, turning nominal finite-sample guarantees into empirical-only observations.

To resolve these gaps with complete methodological transparency, this paper formulates a strictly controlled $O(2)$-equivariant model hierarchy (`O2-M0 -> M1a -> M1b -> M1 -> M2a / M2b -> M3a / M3b / M3`) and evaluates every component under unseen-site (`PINNS` 4-country LOSO) and unseen-location (`JRDB` 9-location LOLO) protocols using 5 random seeds, cluster/clip/location/person bootstrap confidence intervals, and runtime coordinate assertions. Crucially, we report **null findings and negative architectural boundaries** with the same prominence as positive gains—demonstrating where simple `Last-2 CV` remains unbeaten (`< 0.5 m` emergence), why unconstrained soft quality gates invert rank correlation with keypoint completeness, why passive BEV contours fail as rigid curbs, and how physical radial-approaching gating and fixed-predictor split-conformal calibration yield reliable real-time (`~4.3 ms` on `Tesla T4`) safety envelopes.


---

## II. Related Work

### A. Short-Horizon Pose-Assisted Trajectory Prediction
Constant-velocity (`CV`) extrapolation remains a remarkably strong baseline for pedestrian motion when velocity estimates are uncorrupted [4]. To improve short-horizon or near-stationary intent estimation, Human Scene Transformer (`HST`) [14] interleaves skeletal keypoints with 3D positions in a spatio-temporal Transformer, while `Social-Transmotion` [15] introduces modality-masking across ground tracks, 2D boxes, and 2D/3D poses. `Social-Pose` [17] further showed on synthetic `JTA` sequences that models trained solely on clean 3D skeletons suffer severe degradation under Gaussian keypoint jitter ($\sigma = 0.20$), whereas noisy pose training partially recovers accuracy. However, existing studies rarely isolate **explicit quality statistics (`q`)** from implicit feature gating or velocity-dependent damping under strict cross-camera splits, nor do they audit whether learned gates preserve positive monotonicity with respect to keypoint completeness.

### B. Unstructured Scene Context and Mixed-Traffic Interaction
Vectorized HD-map encoders (`VectorNet` [6]) and mixed-traffic CVAE architectures (`MCENET` [13]) aggregate lane centerlines or polar social neighborhoods. In unmarked shared spaces lacking semantic HD maps, roadside systems often extract passive BEV visual contours, yet shadow edges and pavement seams frequently violate non-traversable boundary assumptions. For heterogeneous pedestrian-vehicle conflicts, dynamic risk-field models demonstrate that inter-agent distance and positive radial approaching speed govern yielding and evasive maneuvers [18]. We explicitly separate passive `20 m` BEV polylines (`O2-M2a`) from `10 m` directed heterogeneous graphs (`O2-M2b` with and without radial approaching rate $v_{\text{rad}}$, plus causal radial-conflict gating) to test each cue independently.

### C. Action-Conditioned Multi-Modal Decoding and Conformal Safety
Joint activity-trajectory models such as `NEXT` [16] and switching-dynamics forecasters [8] supervise discrete behavioral states alongside continuous paths. When multi-label kinematic events (`Start, Stop, Turn-Left, Turn-Right, Maintain`) are injected back into trajectory decoding, trajectory regression gradients can distort action calibration unless gradient flow is controlled (`Stop-Grad`). Furthermore, downstream motion planners require calibrated spatial safety tubes [19]; standard split-conformal guarantees hold only when the predictor is **held strictly fixed** between calibration and test sets without post-calibration retraining.

---

## III. Problem Formulation and Geometric Verification

### A. Causal Observation Inputs and Prediction Targets
At trigger frame $t_0$, let a target pedestrian be observed over a strictly causal window $\mathcal{T}_{\text{obs}} = \{t_0 - (T-1)\Delta t, \dots, t_0\}$ ($\Delta t = 0.1\text{ s}, T = 5$ on $10\text{ Hz}$ `PINNS`; $\Delta t = 0.4\text{ s}, T = 2$ with $5$ sub-step $15\text{ fps}$ skeletons on `JRDB`), with no future skeleton inputs permitted:
1. **Ground Trajectory History**: $P_{\text{obs}} = \{p_t \in \mathbb{R}^2\}_{t \in \mathcal{T}_{\text{obs}}}$ in meters and binary validity mask $m^{\text{pos}}_t \in \{0, 1\}$ ($m^{\text{pos}}_{t_0} = 1$). Note that defining "first-sampled windows" solely by absence of a prior $10\text{ Hz}$ annotation (`449` windows on `PINNS`, `261` raw first-frame annotations) reflects annotation onset rather than verified physical de-occlusion.
2. **Variable-Quality Skeletal Sequence**: 2D image keypoints $\{(u_{t,j}, v_{t,j}, c_{t,j})\}_{j=1}^{17}$ with bounding box $b_t = (u^c_t, v^c_t, w_t, h_t)$ (`PINNS`) or 3D BlazePose landmarks $\mathbb{R}^{33\times 3}$ (`JRDB`), paired with a 4-D causal window quality summary $q = [q_1, q_2, q_3, q_4]^\top \in [0, 1]^4$ (valid-joint ratio above $\tau_c = 0.20$, mean valid confidence, normalized box-scale / anatomical plausibility, and track completeness) and window pose-availability indicator $m_{\text{pose}} = \mathbb{I}(\text{any valid torso/limb joint in } \mathcal{T}_{\text{obs}})$.
3. **Local Environment and Neighbors**: Up to $16$ arc-length-resampled ($10$ points/polyline) BEV contours $P_{\text{env}} \in \mathbb{R}^{16\times 10\times 11}$ centered at the **pedestrian's true world coordinate $p_{t_0}$** (with map-availability flag $m_{\text{map}} \in \{0, 1\}$), and up to $12$ directed neighbor edges $e_{j\to i} \in \mathbb{R}^9$ within $10\text{ m}$ encoding relative offset $\Delta p_{ij}$, relative velocity $\Delta v_{ij}$, Euclidean distance $d_{ij}$, and **radial approaching speed** $v_{\text{rad}, ij} = -(\Delta p_{ij}^\top \Delta v_{ij}) / (d_{ij} + \epsilon)$ ($v_{\text{rad}, ij} > 0$ when closing).

Given causal inputs, the model outputs $K = 6$ future candidate trajectories $\{\hat{Y}^{(k)} \in \mathbb{R}^{T_f \times 2}\}_{k=1}^6$ ($T_f = 30$ over $3.0\text{ s}$ on `PINNS`; $T_f = 7$ over $2.8\text{ s}$ on `JRDB`) with mode probabilities $\{\hat{\pi}_k\}_{k=1}^6$, alongside five observable event probabilities $\hat{a} \in [0, 1]^5$ (`Start, Stop, Turn-Left, Turn-Right, Maintain`, evaluated strictly under per-class kinematic eligibility mask $m_{\text{act}} \in \{0, 1\}^5$). Primary comparisons strictly align **same-decoder preferred (`Top-1`, $k^\star = \arg\max_k \hat{\pi}_k$) `ADE_1 / FDE_1`**, reporting 6-mode `minADE_6 / minFDE_6` separately as candidate coverage metrics.

### B. Runtime Homography and World-Center Assertions
To prevent silent coordinate bugs, our pipeline enforces two mandatory runtime assertions prior to training (`verify_pinns_ground_and_polyline_diagnostics`):
1. **Rank-3 $3\times 3$ Ground Homography Verification**: Rather than passing a handedness boolean in place of a camera matrix, each recording's four ground control pairs are fitted via `cv2.findHomography` to a non-singular $H \in \mathbb{R}^{3\times 3}$ ($|\det H| > 10^{-12}$) with axis reflection $y_{\text{world}} = (y_{\min} + y_{\max}) - \tilde{y}$. Across all $1,482 \times 5 = 7,410$ `PINNS` person-frames ($12,460$ frames in the true $20$-frame cohort), runtime assertions confirm **$6,930$ ($93.5\%$) nonzero coronal shoulder/hip facing vectors** (median ground shoulder span $0.2616\text{ m}$, mean cosine $+0.4935$ with walking velocity; $+0.5505$ over $20$ frames), **$6,404$ ($86.4\%$) nonzero torso-lean vectors**, and a **$2.24\ \mu\text{m}$ median reprojection residual** between projected bbox footpoints and published world coordinates.
2. **True Ego World-Origin Polyline Extraction**: Local $20\text{ m}$ polylines are queried around the pedestrian's unshifted world position $p_{t_0}$ (median $\|p_{t_0}\|_2 = 21.13\text{ m}$) rather than ego-subtracted $(0, 0)$, altering $835 / 835$ ($100\%$) map-eligible windows ($830$ containing nonempty $20\text{ m}$ polylines).


---

## IV. Orthogonal-Equivariant Forecasting Methodology

### A. $O(2)$ Parity-Equivariant Frenet Backbone and Pose Fusion (`M0 -> M1`)
To eliminate cross-camera rotational drift while preserving exact left-right reflection equivariance, we construct a causal local Frenet basis $B = [\mathbf{u}, \mathbf{u}_\perp] \in \mathbb{R}^{2\times 2}$ at $p_{t_0}$ using the most recent displacement exceeding $0.05\text{ m}$ (with ground coronal facing $\bar{f}_{\text{cor}}$ as fallback when `use_pose = True`). Transforming relative history into $(x_{\parallel}, x_{\perp})$ yields positive and reflected parity streams $F_+, F_-$ encoded by a shared GRU into $(c_+, c_-)$.

For biomechanical pose fusion (`M1a, M1b, M1`), we extract 11 even-parity gait invariants (normalized ankle span, knee flexion cosines, leg extension, torso verticality, and causal finite differences), project ground coronal facing $f_{\text{cor}}$ and sagittal lean $f_{\text{lean}}$ onto $B$, and append odd-parity left-minus-right knee asymmetry $\Delta \cos\theta_{\text{knee}}$ (negated in the reflected stream). A shared pose GRU produces $(h^p_+, h^p_-)$ with even summary $h^p_{\text{even}} = \frac{1}{2}(h^p_+ + h^p_-)$. Crucially, to isolate explicit quality gating without velocity-damping confounds:
- **`O2-M1a` (Direct Fusion)**: $g = m_{\text{pose}}$.
- **`O2-M1b` (Implicit Feature Gate)**: $g = m_{\text{pose}} \cdot \sigma(\text{MLP}([c_{\text{even}}, h^p_{\text{even}}]))$.
- **`O2-M1` (Single-Factor Explicit Quality Gate)**: $g = m_{\text{pose}} \cdot \sigma(\text{MLP}([c_{\text{even}}, h^p_{\text{even}}, q]))$.
- **`O2-M1_mono_q` (Structurally Monotonic Quality Remedy)**: Motivated by our empirical finding that unconstrained trajectory-trained MLPs can correlate $g$ negatively with $q_1$ due to hidden-state confounding, we also test $g_{\text{mono}} = m_{\text{pose}} \cdot (q_1 q_2)^{\gamma} \sigma(b_0 + w_+^\top q)$ with $\gamma = \text{softplus}(\theta_\gamma) > 0$ and $w_+ = \text{softplus}(\theta_w) \ge 0$, guaranteeing $\partial g_{\text{mono}} / \partial q_1 \ge 0$ and $\partial g_{\text{mono}} / \partial q_2 \ge 0$ everywhere.

When $m_{\text{pose}} = 0$ (complete keypoint loss), $g \equiv 0$ shuts off the pose residual identically.

### B. Local Polylines and Conflict-Gated Neighbor Attention (`M2a / M2b / M2`)
Local polylines (`O2-M2a`) are projected onto $B$ across both parity streams and aggregated via masked point pooling plus null-sink attention gated by $m_{\text{map}}$. Directed $10\text{ m}$ neighbor edges (`O2-M2b`) transform $(\Delta p_{ij}, \Delta v_{ij})$ into $B$ alongside $O(2)$-invariant $[d_{ij}/10, v_{\text{rad}, ij}, m^{\text{spd}}_{ij}, \mathbb{I}_{\text{car}}]$ and attend with a learnable null sink $\ell_\varnothing$:
$$w_{ij} = \frac{\exp(a_{ij})}{\sum_{m=1}^{12} m^{\text{nbr}}_{im}\exp(a_{im}) + \exp(\ell_\varnothing)}.$$
Because open-road or receding neighbors ($v_{\text{rad}} \le 0$) can still perturb frozen `Top-1` mode logits, we further formulate a **Causal Radial-Conflict Gate** (`M2b_ttc_conflict_gate`):
$$w_{\text{risk}, i} = \max_{1 \le j \le 12} \left[ m^{\text{nbr}}_{ij} \, \sigma\!\left(\frac{v_{\text{rad}, ij} - 0.20}{0.25}\right) \exp\!\left(-\frac{d_{ij}}{5.0}\right) \right],$$
which equals $0$ identically on isolated pedestrians ($m^{\text{nbr}}_i = \mathbf{0}$), suppresses receding background traffic, and activates selectively under close positive radial approach, with scalar gain $\alpha^\star \in \{0, 0.25, 0.5, 1.0\}$ selected strictly via inner source-fold cross-validation.

### C. Parity-Consistent Action Bridge, Reranking, and Split-Conformal Tubes (`M3`)
Even-parity contexts predict `Start`, `Stop`, and `Maintain`, while $(c_+, c_-)$ symmetrically score `Turn-Left` and `Turn-Right` (guaranteeing exact left-right swap under reflection). We compare `M3a` (auxiliary supervision only), `M3b` (end-to-end probability injection), and `M3` (`Stop-Grad` injection truncating $\partial \mathcal{L}_{\text{traj}} / \partial \hat{a}$). All $K=6$ trajectories are parameterized as residuals around `Last-2 CV` (Mode $0$ anchored to `CV` with low-displacement bonus; Modes $1,2,5$ central even/odd; Modes $3,4$ symmetric lateral pair), trained with site-balanced shared-winner Charbonnier/Smooth-$L_1$ regression, soft-rank mode cross-entropy, and $m_{\text{act}}$-masked BCE.

For downstream safety certification **without retraining after calibration**, let $\hat{f}_\mathcal{D}$ be a fixed predictor trained strictly on source folds $\mathcal{D}$. On a disjoint calibration fold $\mathcal{C}$ ($|\mathcal{C}| = n$), we compute nonconformity scores $s_i = \min_{k \in \{1..6\}} \max_{t \in \{1..T_f\}} \|\hat{Y}^{(k)}_{i,t} - Y_{i,t}\|_2$ (`Union-6`) and $s^{(1)}_i = \max_t \|\hat{Y}^{(k^\star)}_{i,t} - Y_{i,t}\|_2$ (`Top-1`), set $\hat{r}_{1-\alpha}$ to the $\lceil (n+1)(1-\alpha) \rceil / n$ empirical quantile on $\mathcal{C}$, and evaluate coverage directly on unseen test fold $\mathcal{E}$ using the **unaltered** $\hat{f}_\mathcal{D}$.


---

## V. Experimental Evaluation

### A. Benchmarks, Strict Cross-Scene Splits, and Statistical Protocol
All experiments use five fixed random seeds (`42, 1024, 2025, 3407, 4096`) and site/location-balanced loss weights:
1. **`PINNS` Cross-Country Mixed Traffic ($1,482$ Non-Overlapping Windows, 4-Site LOSO)**: Extracted across $47$ recordings in four countries (`USA` intersections, `China` urban roads, `India` mixed streets, `Switzerland` pedestrian spaces) at $10\text{ Hz}$ ($T = 5$ frames $= 0.4\text{ s}$, $T_f = 30$ frames $= 3.0\text{ s}$). Paired differences use $20,000$-draw 4-site clip-stratified bootstrap 95% confidence intervals (`paired_site_clip_ci`). An additional `623`-window cohort provides verified `20`-frame RTMPose tracks ($12,460$ person-frames) for true $T \in \{5, 10, 20\}$ duration $\times$ quality evaluation. (Results on the legacy `497` sliding-window split from `21` clips are reported strictly with clip/person cluster bootstrap limits: only `16 / 25` preferred-mode comparisons exclude zero at the clip level.)
2. **`JRDB` Campus Shared Spaces ($5,836$ Non-Overlapping Windows, 9-Location LOLO)**: Extracted across $10$ stationary-robot recordings spanning $9$ distinct Stanford campus locations ($2.5\text{ Hz}$ ground tracks, $T = 2$ steps $= 0.4\text{ s}$, $5$ causal $15\text{ fps}$ BlazePose 3D skeletons, $T_f = 7$ steps $= 2.8\text{ s}$, $800$ unique actors), evaluated with 9-location equal means and both location-level and actor-cluster 95% bootstrap CIs.
3. **Additional External Benchmarks (`JTA`, `DUT`, `IMPTC`, `DAIR-V2X`)**: `JTA` ($960$ test windows, $2.5\text{ Hz}$, $T \in \{2, 8\}$, $T_f = 12$); `DUT` ($3,066$ campus pedestrian-vehicle windows); `IMPTC` ($485$ LiDAR-camera intersection windows); and `DAIR-V2X` ($54$ Beijing roadside windows, $31$ strict-clock windows).

### B. `PINNS` Controlled `M0–M3` Chain and `< 0.5 m` Emergence Regime
Table I reports the verified 5-seed 4-site LOSO ablation chain on `PINNS` alongside the `< 0.5 m` low-displacement emergence audit (`first_short_323`: $323$ first-sampled windows with cumulative 5-frame displacement $< 0.5\text{ m}$).

**Table I: `PINNS` 4-Site LOSO Controlled `M0–M3` and `< 0.5 m` Regime Evaluation (5 Seeds, Same-Decoder `Top-1` Primary)**

| Model / Regime Configuration | All `1,482` `ADE_1 / FDE_1` (m) ↓ | All `minFDE_6` (m) ↓ | `first_short_323` (`<0.5 m`) `ADE_1 / FDE_1` (m) ↓ | Polyline Crossing ↓ | 5-Action Macro-AP / F1 ↑ | Paired 47-Clip 95% CI Key Contrast (`ADE_1`; `FDE_1` noted) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Last-2 CV` (Parameter-Free Reference) | 0.4839 / 1.1046 | 1.1046 | **0.3545 / 0.8491** | 2.01% | — | Primary `< 0.5 m` reference (`OLS5 CV`: `0.3833 / 0.8986 m`) |
| **`M0_anchor` (Inner-LOSO CV-Anchored Motion)** | **0.4522 / 1.0194** | 0.6850 | 0.3564 / 0.8559 | — | — | All vs. `CV`: **`[−4.42, −0.54] cm`**; `<0.5m` `FDE_1` vs. `CV`: **`[+0.23, +0.89] cm`** |
| `M1_ground_only` (`M0_anchor` + $3\times 3$ Ground Facing/Lean) | 0.4523 / 1.0196 | 0.6835 | 0.3562 / 0.8555 | — | — | `<0.5m` vs. `M0_anchor`: `[−0.04, +0.02] cm` (null); `FDE_1` vs. `CV`: `[+0.19, +0.85] cm` |
| `Regime_M0` (`< 0.5 m` Source-Only Pure Motion) | — | — | 0.3559 / 0.8497 | — | — | `<0.5m` vs. `CV`: `[−0.11, +0.38] cm` (`FDE_1`: `[−0.84, +1.03] cm`, bias removed) |
| `Regime_M1_ground` (`< 0.5 m` Source-Only + Ground Pose) | — | — | **0.3538 / 0.8462** | — | — | vs. `Regime_M0`: **`[−0.31, −0.03] cm`**; **vs. `CV`: `[−0.20, +0.26] cm` (null)** |
| `O2-M0` (Unanchored Pure Motion Trunk) | 0.4618 / 1.0310 | 0.6843 | 0.3684 / 0.8782 | 1.83% | — | All `FDE_1` vs. `CV`: **`[−10.83, −0.17] cm`** |
| `O2-M1a` (Direct Pose Fusion, $g = m_{\text{pose}}$) | 0.4660 / 1.0381 | 0.6699 | 0.3768 / 0.8994 | 1.81% | — | `Heavy` (`382`) vs. `M0`: **`[+0.51, +4.27] cm`** (unweighted pose degrades) |
| `O2-M1b` (Implicit Gate Without `q`, $\bar{g}=0.765$) | 0.4668 / 1.0384 | 0.6699 | 0.3765 / 0.8987 | 1.82% | — | `Clear` (`409`) vs. `M1a`: **`[−0.71, −0.03] cm`** (`−0.39 cm`) |
| `O2-M1` (Single-Factor `q` Gate, $\bar{g}=0.800$) | 0.4672 / 1.0403 | 0.6707 | 0.3769 / 0.9008 | 1.82% | — | All vs. `M1b`: `[−0.17, +0.24] cm` (null; `T=5 Heavy` 20f cohort: **`[−1.29, −0.07] cm`**) |
| `O2-M2a` (`M1` + True-Ego `20 m` Polylines) | 0.4829 / 1.0689 | 0.6788 | 0.3805 / 0.9042 | 1.78% | — | All vs. `M1`: **`[+0.69, +2.16] cm`** (GT crosses BEV contours in `2.36%` / `4.19%`) |
| `O2-M2b` (`M1` + `10 m` Graph + $v_{\text{rad}}$, No Polyline) | 0.4817 / 1.0589 | 0.6545 | 0.3807 / 0.8960 | 1.74% | — | `Heavy` $+v_{\text{rad}}$ vs. $-v_{\text{rad}}$ (with map): **`[−3.56, −0.09] cm`** |
| `O2-M3` (`Stop-Grad` 5-Action Bridge, No Polyline) | 0.4810 / 1.0628 | **0.6521** | 0.3836 / 0.9031 | 1.85% | **0.3994** / 0.3606 | `<0.5m` `FDE_1` vs. end-to-end `M3b`: **`[−1.59, −0.04] cm`** (`−0.78 cm`) |

Three key conclusions emerge from Table I:
1. **Pure Motion vs. `< 0.5 m` Emergence Bound**: Across all $1,482$ windows (and the $449$ first-sampled windows: `0.5014 / 1.0886 m` vs. `0.5617 / 1.2448 m`), `M0_anchor` substantially outperforms `Last-2 CV`. However, on `first_short_323` (`< 0.5 m`), full-cohort `M0_anchor` incurs a `+0.69 cm` 3.0 s `FDE_1` penalty due to high-speed intersection gradient dominance. Restricting training strictly to `< 0.5 m` source-site windows (`Regime_M0`, `Regime_M1_ground`) eliminates that high-speed bias and yields a statistically significant `−0.22 cm` `ADE_1` gain from verified $3\times 3$ ground orientation over `Regime_M0` (`[−0.31, −0.03] cm`; `−0.35 cm` on `first_short_clear_81`), **yet the `−0.07 / −0.29 cm` difference against parameter-free `Last-2 CV` (`0.3545 / 0.8491 m`) still crosses zero (`[−0.20, +0.26] cm`), leaving `Last-2 CV` unrefuted as the practical `< 0.5 m` baseline**.
2. **Passive BEV Contours vs. Directed Neighbor Graphs**: Ground-truth pedestrian trajectories cross the `20 m` passive BEV contours in `2.36%` of all windows (`4.19%` of map-eligible windows), confirming that unlabeled visual edges capture shadows and surface seams rather than physical curbs; consequently, `O2-M2a` degrades `Top-1 ADE_1` by `+1.58 cm`. Removing noisy contours while retaining the `10 m` neighbor graph (`O2-M2b`, `O2-M3`) drives 6-mode `minFDE_6` down to `0.6513–0.6545 m`.


### C. `2 × 2` Quality-vs.-Speed Factorial and Controlled Keypoint-Degradation Audit
To disentangle explicit quality statistics (`q`) from velocity-dependent pose damping ($\exp(-(v/v_{\text{ref}})^2)$) and training-distribution coverage, Table II synthesizes our `2 × 2` factorial (`374375632`) and controlled `Social-Pose` degradation matrix (`374403419`, where synthetic keypoint perturbations `Test_Jitter020`, `Test_LowerOcc`, and `Test_FullDrop` are strictly separated from natural detector strata).

**Table II: Orthogonal `2 × 2` (`q × Speed Damping`) Factorial and Controlled Keypoint-Stress Audit (`PINNS` 4-Site LOSO & `JRDB` 9-Location LOLO, 5 Seeds)**

| Diagnostic Dimension / Comparison | `PINNS` (`n = 1,482`) `Top-1 ADE_1` Effect (95% Clip CI) | `JRDB` (`n = 5,836`) `Top-1 ADE_1` Effect (95% Location / Person CI) | Mechanistic Interpretation |
| :--- | :---: | :---: | :--- |
| Pure `q` effect without speed damping (`M1_q − M1b`) | `+0.10 cm` (`[−0.16, +0.35] cm`) | `−0.02 cm` (loc `[−0.16, +0.13] cm`) | Null on natural full cohorts (`−0.88 cm` `[−1.29, −0.07]` only on `T=5 Heavy` 20f) |
| Pure `q` effect with speed damping (`M1_qv − M1_v`) | `+0.07 cm` (`[−0.29, +0.36] cm`) | `+0.05 cm` (loc `[−0.04, +0.13] cm`) | Confirms `q` alone did not drive prior coupled-gate gains |
| **Pure speed damping without `q` (`M1_v − M1b`)** | `−0.07 cm` (`[−0.51, +0.49] cm`) | **`−0.82 cm` (loc `[−1.30, −0.39] cm`)** | High-speed residual shrinkage drives `JRDB` `Top-1` gain (`FDE_1`: **`−1.66 cm`**) |
| **Pure speed damping with `q` (`M1_qv − M1_q`)** | `−0.09 cm` (`[−0.54, +0.43] cm`) | **`−0.76 cm` (loc `[−1.21, −0.33] cm`)** | Consistent across both levels of `q` (`FDE_1`: **`−1.51 cm`**) |
| **`VarQTrain − CleanTrain` (`M1`, `Test_Natural`)** | **`−2.41 cm` (`[−3.29, −1.24] cm`)** | **`−3.78 cm` (pers `[−5.37, −2.23] cm`)** | Variable-quality training prevents gate saturation ($\bar{g}$ drops `0.931 -> 0.803`) |
| **`VarQTrain − CleanTrain` (`M1`, `Test_Jitter020`)** | `+0.55 cm` (`[−0.67, +2.14] cm`) | **`−4.28 cm` (pers `[−5.88, −2.74] cm`)** | `JRDB` 3D $\sigma=0.20\text{ m}$ jitter + 35% drop shrinks `VarQ` $\bar{g}$ to `0.088` |
| **`m_pose = 0` Shutoff vs. No-Shutoff (`Test_FullDrop`)** | **`−3.27 cm` (`[−4.11, −2.49] cm`)** | **`−3.68 cm` (pers `[−4.48, −2.91] cm`)** | Hard shutoff guarantees $\max |g| = 0$ (`M1 − M0` crosses zero: `+0.10 / −0.15 cm`) |
| **Out-of-Site Spearman $\rho(g, q_1)$ (`VarQTrain` `M1`)** | **`−0.1883`** (`Clear 0.782`, `Heavy 0.839`) | **`−0.2184`** (`Clear 0.451`, `Partial 0.594`) | **Negative discovery**: unconstrained `MLP([h_m, h_p, q])` inverts quality rank via $h_m$ |

### D. `JRDB` 9-Location `M0–M3` Chain and Fixed-Predictor Split-Conformal Coverage
On `JRDB` ($5,836$ windows across $9$ campus locations, `374304954` & `374375632`), `Last-2 CV` (`0.2812 / 0.5521 m`) retains the lowest single-mode `Top-1 ADE_1` (`O2-M0`: `0.2947 / 0.5739 m`), reflecting predominantly smooth campus corridor walking at $2.5\text{ Hz}$. Nevertheless, 6-mode `minFDE_6` improves monotonically by **`−44.3%`** along `CV -> M0 -> M1 -> M2b -> M3` (`0.5521 -> 0.3369 -> 0.3288 -> 0.3116 -> 0.3078 m`), radial approaching rate $v_{\text{rad}}$ lowers partial-occlusion `Top-1 ADE_1` by **`−0.83 cm`** (`[−1.92, −0.07] cm`), and **5-class observable action supervision (`Macro-AP = 0.3498`) enables `M3a / M3b` to significantly outperform `M2b` on `Top-1 ADE_1` by `−0.80 / −0.78 cm` (`95% location CI [−1.57, −0.20]` / `[−1.65, −0.08] cm`; `−1.69 / −1.78 cm` on `Clear` and `−2.95 / −3.19 cm` on `Partial`)**.

**Table III: Fixed-Predictor Split-Conformal Safety Coverage and Tube Radius Without Post-Calibration Retraining**

| Benchmark & Strict Zero-Retrain Split-Conformal Protocol | Nominal $1-\alpha$ | `Last-2 CV` Cov. / Radius (m) | `O2-M3` `Top-1` Cov. | `O2-M3` (`M2b`) `Union-6` Cov. / Radius (m) | Radius Reduction vs. `CV` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`PINNS` Within-Site 50/50 Clip Split** (Fixed 3-Site `O2-M3`) | 90% / 95% | 90.35% (`2.655`) / 95.33% (`3.751`) | 89.99% / 95.02% | **90.60% (`1.593`) / 94.94% (`2.384`)** | **`−40.0% / −36.4%`** |
| **`PINNS` Cross-Country 12 Ordered Pairs** (Fixed 2-Site `O2-M3`) | 90% / 95% | 87.31% (`2.704`) / 92.22% (`3.816`) | 87.24% / 92.33% | **86.48% (`1.632`) / 91.74% (`2.428`)** | **`−39.6% / −36.4%`** |
| **`JRDB` Within-Scene Temporal Split** (Fixed 8-Loc `O2-M2b / M3`) | 90% / 95% | 87.62% (`1.364`) / 94.31% (`1.912`) | 88.99% / 94.64% | **88.37% (`0.724`) / 94.48% (`1.051`)** | **`−46.9% / −45.0%`** |
| **`JRDB` Cross-Location 72 Ordered Pairs** (Fixed 7-Loc `O2-M3`) | 90% / 95% | 87.14% (`1.426`) / 92.77% (`1.928`) | 86.30% / 92.81% | **86.77% (`0.744`) / 92.81% (`1.056`)** | **`−47.8% / −45.2%`** |

![Figure 1: Controlled O(2) M0-M3 ablation, ground-facing <0.5 m evaluation, and PINNS fixed-predictor split-conformal coverage](../docs/figures/fig17_ppt_mainline_conformal_and_short_disp.png)

![Figure 2: Orthogonal 2x2 (q x speed) factorial, <0.5 m regime bound, Slide-5 keypoint-degradation & gate rank-inversion audit, and JRDB 72-pair split-conformal coverage](../docs/figures/fig18_wave17_factorial_slide5_and_jrdb_conformal.png)

### E. Causal Radial-Conflict Gated Interaction and Zero-Leakage Action Reranking
To resolve why un-gated `O2-M2b` improved `minFDE_6` while perturbing `Top-1 ADE_1` on `PINNS` (`+1.38 cm` over `M1`), Table IV decomposes neighbor-graph variants across physical approaching strata (`374440205`) alongside zero-leakage inner-fold Platt action calibration and candidate reranking.

**Table IV: Causal Radial-Conflict Gating (`M2b_ttc_conflict_gate`) and Zero-Leakage Action Calibration / Reranking (5 Seeds)**

| Evaluation Stratum / Metric | `PINNS` (`n = 1,482`, 4-Site LOSO) | `JRDB` (`n = 5,836`, 9-Location LOLO) | Paired 95% Clip / Location CI & Boundary Note |
| :--- | :---: | :---: | :--- |
| `S1_non_approaching` (`M2b_frozen − M1`) | `+2.57 / +4.43 cm` (`n = 527`) | `+0.02 / −0.42 cm` (`n = 2,860`) | Confirms receding/distant traffic perturbs un-gated `PINNS` mode logits (`[+0.90, +5.81] cm`) |
| **`All` Windows: `M2b_ttc_conflict_gate`** | **`0.4572 / 1.0223 m`** (`M1`: `0.4679 / 1.0424`) | **`0.2987 / 0.5806 m`** (`M2b`: `0.3036 / 0.5845`) | `PINNS` vs. `M1`: **`[−1.52, −0.58] cm`** (vs. `M2b`: **`−2.45 cm`**); `JRDB` vs. `M2b`: **`[−0.93, −0.05] cm`** |
| **Close Conflict (`TTC <= 4 s` / `d <= 6 m`)** | **`−1.96 / −3.56 cm`** vs. `M1` (`TTC <= 4s`) | **`−0.71 / −1.79 cm`** vs. `M1` (`d <= 6m`) | `PINNS` `TTC<=4s` CI **`[−2.78, −0.60] cm`**; `PINNS` Car-Approach (`315`) vs. `M2b`: **`−1.98 cm`** (`[−4.90, −0.25]`) |
| **5-Action ECE (`Raw 0.5 -> Inner Platt`)** | **`0.1717 -> 0.0349`** (`Macro-F1`: `0.3606 ->` **`0.3911`**) | **`0.0933 -> 0.0040`** (`Macro-AP`: **`0.3567`**) | Cuts `PINNS` `Left / Right` false positives from `236 / 231` to `76 / 46` (zero test leakage) |
| **Action-Consistent `Top-1` Reranking (`M3`)** | `0.4810 -> 0.4769 m` (`−0.41 / −0.82 cm`) | **`0.2973 -> 0.2930 m` (`−0.43 / −0.81 cm`)** | `JRDB` CI **`[−0.66, −0.25] cm`** (driven by `Maintain` `−0.53 cm`; `PINNS` CI `[−0.96, +0.14] cm` crosses zero) |

---

## VI. Discussion, Limitations, and Conclusion

1. **Boundary of Short-Term Pose on Near-Stationary Emergence**: Even with verified $3\times 3$ ground homographies ($2.24\ \mu\text{m}$ footpoint residual) and `< 0.5 m` regime-specific training, single-camera 2D keypoint back-projection yields only sub-centimeter `ADE_1` gains (`−0.22 cm`) over neural pure-motion models and remains statistically indistinguishable from `Last-2 CV` (`[−0.20, +0.26] cm`) on `< 0.5 m` windows. Claims of "irreplaceable" 2D pose utility on near-stationary road emergence are therefore unsupported by current monocular roadside tolerances (`~0.28 m` ground calibration error in `PINNS` [20]).
2. **Annotation Onset vs. Visual De-Occlusion**: First-sampled cohorts (`449` / `261` windows) identify track initialization in published annotations rather than annotated physical occluders; pixel-level occluder segmentation with synchronized hardware timestamps remains necessary for future de-occlusion benchmarks.
3. **Summary**: Across six real and synthetic benchmarks, $O(2)$ parity-equivariant local canonicalization, `m_pose = 0` hard shutoff, variable-quality pose training, `10 m` radial-approaching neighbor interaction, `Stop-Grad` action supervision, and fixed-predictor split-conformal calibration deliver compact (`~21k` parameters), real-time (`~4.3 ms` on `Tesla T4`), and statistically transparent multi-modal trajectory prediction for unstructured mixed traffic.

---

## References

[1] A. Rasouli and J. K. Tsotsos, "Autonomous vehicles that interact with pedestrians: A survey of theory and practice," *IEEE Trans. Intell. Transp. Syst.*, vol. 21, no. 3, pp. 900–918, Mar. 2020.  
[2] A. Rudenko *et al.*, "Human motion trajectory prediction: A survey," *Int. J. Robot. Res.*, vol. 39, no. 8, pp. 895–935, Jul. 2020.  
[3] R. Korbmacher and A. Tordeux, "Review of pedestrian trajectory prediction methods: Comparing deep learning and knowledge-based approaches," *IEEE Trans. Intell. Transp. Syst.*, vol. 23, no. 12, pp. 24126–24144, Dec. 2022.  
[4] C. Schöller, V. Aravantinos, F. Lay, and A. Knoll, "What the constant velocity model can teach us about pedestrian motion prediction," *IEEE Robot. Autom. Lett.*, vol. 5, no. 2, pp. 1696–1703, Apr. 2020.  
[6] J. Gao *et al.*, "VectorNet: Encoding HD maps and agent dynamics from vectorized representation," in *Proc. IEEE/CVF CVPR*, 2020, pp. 11525–11533.  
[7] R. Martín-Martín *et al.*, "JRDB: A dataset and benchmark of egocentric robot visual perception of humans in built environments," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 45, no. 6, pp. 6748–6765, Jun. 2023.  
[8] J. F. P. Kooij *et al.*, "Context-based path prediction for targets with switching dynamics," *Int. J. Comput. Vis.*, vol. 127, no. 3, pp. 239–262, Mar. 2019.  
[13] H. Cheng, W. Liao, M. Y. Yang, B. Rosenhahn, and M. Sester, "MCENET: Multi-context encoder network for homogeneous agent trajectory prediction in mixed traffic," in *Proc. IEEE ITSC*, 2020, pp. 1–8.  
[14] T. Salzmann, H.-T. L. Chiang, M. Ryll, D. Sadigh, C. Parada, and A. Bewley, "Robots that can see: Leveraging human pose for trajectory prediction," *IEEE Robot. Autom. Lett.*, vol. 8, no. 11, pp. 7090–7097, Nov. 2023.  
[15] S. Saadatnejad, Y. Gao, K. Messaoud, and A. Alahi, "Social-Transmotion: Promptable human trajectory prediction," in *Proc. ICLR*, 2024.  
[16] J. Liang, L. Jiang, J. C. Niebles, A. G. Hauptmann, and L. Fei-Fei, "Peeking into the future: Predicting future person activities and locations in videos," in *Proc. IEEE/CVF CVPR*, 2019, pp. 5725–5734.  
[17] Y. Gao *et al.*, "Social-Pose: Enhancing trajectory prediction with noisy 3D human pose," 2025.  
[18] W. Wu *et al.*, "Pedestrian crossing azimuth prediction under dynamic vehicle-pedestrian risk fields," *J. Mech. Eng.*, 2024.  
[19] L. Lindemann, M. Cleaveland, G. Shim, and G. J. Pappas, "Safe planning in dynamic environments using conformal prediction," *IEEE Robot. Autom. Lett.*, vol. 8, no. 8, pp. 5116–5123, Aug. 2023.  
[20] Songan-Lab, "PINNS: Multi-country unstructured pedestrian-vehicle interaction dataset," arXiv:2605.25947, 2026.
