# Five observed frames, six residual trajectories

Draft for a single-seed exploratory experiment. The candidate architecture was selected after seeing earlier PINNS results on these four sites. No real-data optimization or new-model scores were used to prepare this draft.

## Controlled comparison

Fit the full 2 × 2 crossing a zero or observed constant-velocity base with motion-only M0 or explicit pose-quality-gated M1. Fit all sixteen models (4 sites × 4 arms) on a single Kaggle CPU worker. The primary outcome is top-probability ADE of velocity-anchored M0 minus zero-base M0, in metres, giving each site equal weight. Negative values favor the anchor. Report each site, the macro estimate and a paired 95% whole-clip resampling interval, regardless of sign. Prespecified secondary comparisons are listed in the runner. Both 3.0 s FDE and true-future minimum ADE/FDE over six candidates are reported; the two minima can come from different candidates. Mode selection frequencies, top-versus-oracle agreement, average pairwise path separation and probability entropy describe how the modes are used. Archived original seed-42 M0/M1 and last-two CV are external comparators.

All arms use identical shared initial parameter tensors (including queries, probability head and recurrent transitions), final epoch 28, seed 42 and identical within-fold shuffled batches. Zero-initialized displacement weight and bias make all six paths start exactly at their respective base. On the CV arms, the base is computed from the measured final two of five ground positions at 10 Hz; a 30-step path is compared bit-for-bit against the prior float64 CV implementation. The GRU's returned cumulative residual is added to this path. After the first transition the GRU feeds its previous residual increment back into the next transition, identically in both base conditions. Zero and CV arms both retain the original initial velocity side input to the GRU. Both learn corrections, with no fixed CV candidate retained once training begins.

The old winner-take-all indexing would send every initial tied window to query zero. Instead, resolve exact best-ADE ties by the direction of the greatest ground-truth departure from the *current arm's fixed base* over valid training steps, earliest in case of equal distance. Six directions at 60-degree intervals are anchored to the last observed ground step (world x/y for exact standstill). Among tied queries maximize the directional dot product; distribute the hard target and regression credit evenly across any exactly tied directional optima. If the target equals its base, each tied winner receives equal credit. Unique best-ADE winners receive the original loss, up to numerical precision. Test labels never enter inference. Changing the base therefore changes the initial path and, as intended, can also change the training-time tie assignments. The experiment estimates these combined consequences of the base under a fixed rule. Epoch logs record the number of ties and mode credit among tied and all training windows, including the first minibatch separately. Six distinct geometrical sectors need not have equal real-data support.

Use the original 1,482 complete label-derived windows, five 10 Hz ground observations over 0.4 s, frozen pose features, and thirty future points through 3.0 s. Four train/test counts are 666/816, 1,398/84, 1,050/432 and 1,332/150. Test clip counts are 14/9/11/13, respectively, for America, Italy, Japan and Thailand. Train on the three other physical sites; no validation selection. New-site transfer requires separate evaluation. Confidence intervals are conditional on one seed and these four repeatedly studied sites.

## Running after approval

Make the original project importable with PYTHONPATH. Use the official labels/calibration staging directory and the accepted five-seed ZIP (42,019,120 bytes, previously archived). The ZIP already contains the frozen pose and all four-site seed-42 reference weights. No video or pose extractor is required. Place these scripts together.

Run the first two commands before launching any fits. Check only reconstructs identities, histories and the exact CV base. Register creates a deterministic local plan; retain/publish its digest before reading any new results. Transfer the fixed plan and scripts intact to the single target worker.

    python run_anchor.py DATA FIVE_SEEDS.zip plan.json --mode check
    python run_anchor.py DATA FIVE_SEEDS.zip plan.json --mode register
    CUDA_VISIBLE_DEVICES='' python run_anchor.py DATA FIVE_SEEDS.zip plan.json --mode fit

A fresh worker must run all four arms and sites. Within the worker, completed arm checkpoints are reusable with verified registration, site, worker identity and epoch count. The default private output is /kaggle/working/anchored_six/anchored_six_candidates.zip. It contains sixteen weights, separately readable training logs, compact per-site six-candidate predictions and scores for six neural comparators, last-two CV errors, registration, software and aggregate report. Only the aggregate needs public distribution. Wall time for this exact design has not been measured.

Local synthetic tests run with "python -m unittest discover -s . -p 'test*.py'".
