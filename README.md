# Unstructured-ShortPose-Pred

**Joint Pedestrian Action and Trajectory Prediction on Unstructured Roads under Short-Term Variable-Quality Pose Observation**  
*短时姿态观测下的非结构化道路行人动作与轨迹联合预测*

---

## Predict from five observations

Run the CPU predictor with a five-frame held-out PINNS checkpoint and an observation JSON file. Use `python -m scripts.predict_five_frame_pedestrian M1.pt five_observations.json --output forecast.json` from the repository root. The inference entry uses NumPy and PyTorch. It accepts the original five-frame M0/M1 weights or a checkpoint from the registered individual joint-action experiment.

Supply five ascending `timestamps_s` at 10 Hz and five measured, consistently oriented, metric `ground_xy_m` positions for one tracked person. Pose models also take `image_height_px` and five `pose_frames`. Each pose frame contains `bbox_center_xywh_px` (box centre x/y, width and height in original image pixels) and `keypoints_xyc` (17 COCO joints with image x/y and confidence). A null pose frame denotes missing image observations. A measured box may be provided with null keypoints when joint estimation is unavailable. Camera calibration and pixel-to-ground conversion are upstream of this entry.

The output contains six future paths of 30 positions at 0.1 s intervals in the input ground coordinate frame, the six probabilities, the zero-based selected path and the last-pair constant-velocity reference. A checkpoint trained with an action head additionally returns five independent event probabilities for start, stop, left turn, right turn and maintain. These targets describe movement events derived from ground trajectories. The four sites have already participated in research development.

For input directly from five camera images, supply timestamps, five tracked center-width-height boxes for one pedestrian and a camera-to-ground homography in metres. The image runner reads the original pixels, estimates 17 RTMPose joints and projects the bottom centre of each box into the supplied coordinate frame. A registered joint-action checkpoint produces six trajectories and five event probabilities. The input schema and time, calibration and image checks are in [the image runner](scripts/predict_five_frame_images.py).

```sh
python -m scripts.predict_five_frame_images detached.pt camera_input.json end2end.onnx --output forecast.json --observations-out observations.json
```

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

The 40 reconstructed maps contained 400 contours extracted from bird’s-eye imagery and 25 convex hulls fitted to training pedestrians’ positions. All 497 original test windows had contour input, and 495 had a hull. The wall, planter, temporary-barrier and parked-vehicle channels were empty. In 618 training windows a scored future position matched a vertex of an available hull; 349 validation windows had a same-clip background image later than the prediction origin. Both counts were zero in test windows. The separate seed-42 M0/M1 site-held-out runs used no maps. [Map sources and timing by split](results/pinns_map_sources.json)

A separate residual-ridge experiment reused the original splits. Maps drew on earlier training frames from the same clip, and the occupancy hull excluded the target pedestrian. Across 497 test windows, adding the nearest image contour to motion and availability flags changed mean ADE from 0.6002 to 0.6234 m (paired difference +0.0232 m, 95% clip-bootstrap interval −0.0126 to +0.0417 m). The hull gave 0.6098 m, motion-only ridge 0.5842 m, and five-frame constant velocity 0.5430 m. Only 233 windows had a scored endpoint at exactly 3 s. Two Linux reconstructions agreed with each other but differed from the frozen macOS map tensors. The first Kaggle run stopped before fitting. The reported run used the verified frozen tensors. [Geometry controls and independent replay](results/pinns_causal_geometry.json)

The original scenario model received neighbors in 480 of 497 archived test windows. Its 2,724 directed edges comprised 2,444 pedestrian edges and 280 car edges. All edge tensors, masks and type vectors were reconstructed from observed data. The release distinguishes people and cars at source. [Input coverage and temporal reconstruction](results/pinns_neighbor_inputs.json)

A separate single-trajectory development experiment held out whole clips in five folds across the same four PINNS sites (47 clips, 1,482 windows). With five observed positions and a 3 s forecast, adding annotated neighbor availability and nearest-person/car geometry to a motion-only ridge predictor reduced equal-site ADE/FDE from 0.7290/1.4998 m to 0.6818/1.3907 m. The paired ADE difference was −0.0472 m (95% within-site clip bootstrap interval −0.0713 to −0.0192 m). Availability alone gave 0.7319/1.5049 m. Five-point constant velocity achieved the lowest equal-site errors, 0.5557/1.2103 m. The comparison uses annotation-supplied identities and positions, and all four sites contribute training clips in every fold. [Neighbor experiment and independent aggregation checks](results/pinns_neighbor_residuals.json)

On the outdoor JRDB training recording Huang Lane, projecting 6,430 person positions into the panorama gave a median horizontal discrepancy of 8.53 pixels from the matching box center. HST supplied 33-joint 3D records for 554 of these person-frame pairs. The derived tracks run at 2.5 Hz and the joint records lack image-plane confidence. The planned 10 Hz RTMPose evaluation still requires suitable source frames and annotations. [Outdoor JRDB development check](reviews/jrdb_development/README.md)

A registered within-model intervention held the first and last poses and every mask and score fixed. Swapping eligible joint coordinates in the second and fourth observed frames changed 1,324 of 1,482 windows. Across five seeds the ADE change was below 0.001 mm and its paired interval included zero. Disabling the whole pose residual raised ADE by 6.22 mm (95% interval 2.30–8.94 mm). [Protocol](reviews/pinns_pose_order/PROTOCOL.md) · [Kaggle results](reviews/pinns_pose_order/pose_order_scores.json) · [Independent replay](reviews/pinns_pose_order/verified_provenance.json)

On the same 623 held-out forecast origins, the seed-42 Intel CPU fits gave top-ranked ADE of 0.5329, 0.5357 and 0.5276 m for motion alone (5, 10 and 20 frames), versus 0.5896, 0.6283 and 0.6445 m for the explicit pose gate. Last-pair constant velocity scored 0.4751 m at every duration. All 449 windows starting at the first sampled annotation are excluded by the matched-history requirement. Relative to five frames, twenty-frame motion ADE changed by −0.0054 m (95% paired clip interval −0.0319 to +0.0231 m) and explicit-gate ADE rose by 0.0549 m (+0.0096 to +0.1061). [Matched cohort and original runs](results/pinns_matched_cpu_durations.json) · [Four-model common-worker repeat at 20 frames](results/pinns_matched_same_worker_t20.json)

With five frames, four-site ADE was 0.5805 m for the implicit gate and 0.5679 m for projected concatenation. Both paired ADE intervals against the explicit gate included zero. An initial 20-frame comparison used AMD for the explicit gate and Intel for concatenation. Their common-Intel-worker difference (concat minus explicit) was +0.0528 m with a 95% interval of −0.0199 to +0.1240 m. The original 48 fits and all 16 common-worker fits were independently replayed. [Initial fusion fits and CPU models](results/pinns_matched_cpu_fusion.json)

At one external IMPTC intersection in Aschaffenburg, Germany, a frozen protocol retained 564 non-overlapping windows within pedestrian tracks across four public demonstration recordings. With the recordings weighted equally, two-point constant velocity using actual timestamps achieved 0.2199/0.5418 m ADE/FDE. The four seed-42 PINNS M0 checkpoints retained their nominal 0.1 s step and averaged 0.3944/0.8657 m. Each checkpoint had higher ADE and FDE than constant velocity in every recording. IMPTC supplies filtered tracking positions and lacks verified per-person image poses for this comparison. [Frozen protocol](reviews/imptc_external/external_protocol.md) · [Per-recording, per-model and pooled results](reviews/imptc_external/external_motion_scores.json)

Three linear action classifiers used five observed frames to anticipate events derived from the next 3 seconds of ground tracks. Five-fold development cross-validation kept each of the 47 videos in one evaluation fold and included all four previously examined locations in every fold. All comparisons used the same 1,482 windows, regularization and 0.5 decision threshold. With motion alone, with final-frame pose added, and with ordered joint changes also added, pooled start AP was 0.2131, 0.3439 and 0.3375. At the fixed threshold the models detected 0, 3 and 4 of 49 starts and missed all 53 stops. Adding ordered changes to final-frame pose also lowered pooled AP for both turn directions. These descriptive results concern trajectory-derived events at known sites. [Per-event scores, fold counts and independent verification](results/pinns_action_probes.json)

A second development experiment held out each previously studied site and selected a decision threshold using only the other three sites’ out-of-fold scores. On the combined 1,482 windows, the model with ordered pose changes detected 39 of 49 starts with 89 false alarms (F1 0.4407). Motion alone detected 49 with 129 false alarms (F1 0.4317), matching the rule that flags every eligible start. Equal-site start F1 was 0.4458 with ordered pose changes and 0.4613 with motion alone. The pose model detected no stops at the fixed 0.5 threshold. At the inner-selected threshold it detected 24 of 53 stops with 487 false alarms (F1 0.0851), compared with 8 detections and 35 false alarms for motion alone (F1 0.1667). [Nested site-level metrics and independent score replay](results/pinns_nested_start_stop.json)

On the same 1,482 complete windows, a separately registered one-seed, individual-only action/trajectory experiment trained 16 checkpoints. Equal-site ADE was 0.5885 m without an action head, 0.5161 m with auxiliary action labels, 0.5068 m with end-to-end action conditioning and 0.5081 m with the conditioning gradient detached. The primary detached-minus-end-to-end difference was +0.0014 m (paired 95% interval −0.0087 to +0.0144). Their mean binary Brier difference was −0.00134 (−0.00298 to −0.00024). Each action head detected 40 of 49 starts at the fixed threshold, with 149–155 false alarms. They detected zero or one of the 53 stops. [Preregistered analysis](reviews/pinns_individual_actions/preregistration.json) · [Independent 16-weight verification and per-class results](results/pinns_individual_joint_actions.json) · [First-annotation subgroup](results/pinns_joint_action_history_groups.json)

Run `python -m scripts.evaluate_pinns_action_probes "$PINNS_ROOT" "$PILOT_DIR/pose.npz" --output action_probes.json` from the repository root to reproduce the out-of-fold probabilities.

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

Kaggle completed all 32 additional fits for seeds 1024, 2025, 3407 and 4096. Independent CPU inference reproduced all 40 checkpoints on every held-out window to within 0.000004 m of the cloud errors. With sites and seeds weighted equally, M0 reached 0.5126/1.0977 m ADE/FDE and M1 reached 0.5375/1.1455 m. The mean paired difference was +0.0250/+0.0477 m. One of the five seeds favored M1. Across the five seeds, the mean ADE differences for the clear, partial and low-quality buckets were +0.0253, +0.0245 and +0.0227 m. [Per-seed paired results, conditional clip intervals and weight checks](results/pinns_heldout_five_seed_replay.json)

After examining constant velocity and six-candidate coverage on these four sites, we tested a fixed rule on all 40 checkpoints. It selects the candidate closest on average over 30 steps to an extrapolation from the last two observed positions. Exact ties go to the higher network probability, then the lower candidate index. With sites and seeds weighted equally, this changed the selected candidate in 36.31% of M0 and 51.76% of M1 predictions. ADE/FDE fell to 0.4963/1.0685 m for M0 and 0.4961/1.0664 m for M1. Direct velocity had lower ADE and higher FDE at 0.4839/1.1046 m. The paired 95% whole-clip intervals for the change in ADE were [−0.0325, −0.0070] m for M0 and [−0.0605, −0.0298] m for M1, conditional on these development sites and fits. [Per-site and per-seed reranking results](results/pinns_velocity_reranking.json)

A separate single-trajectory ridge model predicted corrections to last-two-point constant velocity on the same 1,482 windows. The three training sites selected one shared correction multiplier for the motion, last-pose and ordered-pose models before evaluation at each held-out site. Equal-site ADE/FDE was 0.4839/1.1046 m for constant velocity, 0.4735/1.0742 m for motion residuals, 0.4805/1.0851 m with last-frame pose, and 0.4851/1.0892 m with ordered pose changes. The prespecified ordered-pose minus constant-velocity ADE difference was +0.0013 m (95% paired clip-bootstrap interval −0.0056 to +0.0103 m). These four development sites showed no clear ADE benefit from the ordered-pose correction. [Nested residual experiment](results/pinns_pose_residuals.json)

The same five-frame inputs also allow a comparison by track history. Of 1,482 windows, 449 began at the person's first sampled annotation in that clip and 1,033 followed at least 35 consecutive earlier annotations. First/later counts were 240/576 in 14 US clips, 28/56 in 9 Italian clips, 117/315 in 11 Japanese clips and 64/86 in 13 Thai clips. With sites and seeds weighted equally, M0/M1 top-candidate ADE was 0.5968/0.6261 m for first annotations and 0.4780/0.5009 m for later windows. Historical reranking gave 0.5746/0.5697 m and 0.4646/0.4665 m, respectively. Direct velocity gave 0.5617 and 0.4517 m. These strata describe annotation timing within previously examined recordings. Physical visibility at track onset was unlabelled. [Subgroup ADE/FDE, future-informed reference and paired clip intervals](results/pinns_annotation_history_scores.json)

The extension reuses the eight verified seed-42 fits. The original pilot results use seed 42. Training requires a CUDA device and the pilot's PyTorch build (`2.11.0+cu130`). Supply the official label/calibration directory with its video inventory, the extracted seed-42 bundle containing `pose.npz` and `folds/`, and a separate writable output directory. The first command verifies the inputs on CPU. The second resumes matching checkpoints and writes `shortpose_heldout_five_seeds.zip` when all 40 fits are present. [Five-seed runner](scripts/run_heldout_seed_sensitivity.py)

```bash
python -m scripts.run_heldout_seed_sensitivity --data-root "$PINNS_ROOT" --pilot-dir "$PILOT_DIR" --check-only
python -m scripts.run_heldout_seed_sensitivity --data-root "$PINNS_ROOT" --pilot-dir "$PILOT_DIR" --output-dir "$SEED_OUTPUT_DIR"
python -m scripts.verify_heldout_five_seeds "$SEED_OUTPUT_DIR/shortpose_heldout_five_seeds.zip" "$PINNS_ROOT" "$PILOT_DIR" --output five_seed_replay.json
```

### Test sensitivity to a one-frame image offset

After extracting the pilot's `shortpose_heldout.zip`, point `PILOT_DIR` at the directory containing `pose.npz` and `folds/`, and `RTMPOSE_ONNX` at the original RTMPose model. The first command re-extracts 2,985 person observations in three US videos one decoded video frame later. It checks sampled zero-offset batches against the frozen cache on the same ONNX execution providers. Choose a new output cache path.

```bash
python -m scripts.cache_pinns_frame_offset "$PINNS_ROOT" "$PILOT_DIR/pose.npz" "$RTMPOSE_ONNX" shifted_pose.npz
python -m scripts.evaluate_pinns_frame_offset "$PINNS_ROOT" "$PILOT_DIR" shifted_pose.npz --output frame_offset_evaluation.json
```

Using the GPU-extracted shifted poses and the four frozen seed-42 M1 checkpoints, the CPU replay passed the original per-window checks. Across the 597 shifted US windows, mean ADE changed by −0.0000730 m and mean FDE by −0.0001085 m. The 95th percentile absolute per-window FDE change was 0.0100 m, with a maximum of 0.0306 m. All 219 other US windows produced exactly the same trajectories. Across the four equally weighted sites, the changes were −0.0000134 m and −0.0000198 m. The Kaggle CUDA replay stopped at the exact-equality check for the 219 untouched windows. Its maximum discrepancy and cause remain unknown. These results measure sensitivity to test image selection. Temporal alignment and the effect of refitting on shifted images remain open. [Frame-offset results and device checks](results/pinns_heldout_frame_offset.json)
