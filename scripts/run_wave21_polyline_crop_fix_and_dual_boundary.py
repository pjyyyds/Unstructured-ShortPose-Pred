"""Wave 21A-GPU1: Fix repeated-crop polyline missing-lines bug & compare No-Map vs Auto vs Dual-Confirmed Barriers."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.benchmark_heldout_sites import load_site_windows
from scripts.ppt_mainline_o2_models import PPTMainlineO2Forecaster, polyline_crossing_rate_np
from scripts.run_pinns_matched_o2_multiduration import make_site_weights
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import load_ppt_cohort_with_causal_maps
from scripts.run_wave18_conflict_gated_interaction_5seed import safe_site_clip_ci
from scripts.run_wave21_pinns_true_emergence_and_6slot_diag import SEEDS, blind_pinns_physical_emergence_audit
from src.data.pinns_extractor import _check_right_handed_homography, _parse_pinns_label_file
from src.data.scene_graph import build_local_polylines
from src.models.bioeq_o2_joint import o2_joint_loss


def _dedup_world_polylines(polys: list[tuple[np.ndarray, np.ndarray]], tol_m: float = 0.12) -> list[tuple[np.ndarray, np.ndarray]]:
    kept: list[tuple[np.ndarray, np.ndarray]] = []
    sigs: list[np.ndarray] = []
    for xy, attr in polys:
        if len(xy) < 2:
            continue
        sig = np.concatenate([xy[0], xy[-1], xy.mean(axis=0)])
        rev_sig = np.concatenate([xy[-1], xy[0], xy.mean(axis=0)])
        if any(min(np.linalg.norm(sig - s), np.linalg.norm(rev_sig - s)) <= tol_m for s in sigs):
            continue
        kept.append((xy, attr))
        sigs.append(sig)
    return kept


def build_uncropped_and_dual_confirmed_maps(
    meta_dir: Path,
    joint_zip: Path,
    causal_map_bin: Path,
) -> tuple[dict[str, np.ndarray], dict]:
    """Fix repeated-cropping bug and classify segments via two independent blind barrier criteria."""
    groups, _, _ = load_site_windows(meta_dir)
    rows = [r for s in sorted(groups) for r in groups[s]]
    n = len(rows)
    with zipfile.ZipFile(causal_map_bin) as mz:
        if "p_env_image.npy" in mz.namelist():
            raw_map = causal_map_bin.read_bytes()
        else:
            nm = next(x for x in mz.namelist() if x.endswith(".npz"))
            raw_map = mz.read(nm)

    clip_snaps: dict[str, dict[int, dict[int, np.ndarray]]] = {}
    clip_parked: dict[str, dict[int, list[tuple[np.ndarray, np.ndarray]]]] = {}
    clip_train_pts: dict[str, np.ndarray] = {}
    clip_hull_3d: dict[str, np.ndarray] = {}
    for c in sorted({r.clip for r in rows}):
        cond, _ = c.rsplit("_", 1)
        cal = json.loads((meta_dir / "videos" / cond / "calib.json").read_text())
        fl = _check_right_handed_homography(cal)
        ymn, ymx = float(cal["y_min"]), float(cal["y_max"])
        p3 = np.asarray(cal["pts_3d"], dtype=np.float64)
        if fl:
            p3[:, 1] = ymx - (p3[:, 1] - ymn)
        clip_hull_3d[c] = p3
        parsed = _parse_pinns_label_file(meta_dir / "labels" / f"{c}.txt", fl, ymn, ymx)
        clip_snaps[c] = {t: {int(o["id"]): np.array([o["wx"], o["wy"]], dtype=np.float64) for o in v if o["type"] == "Person"} for t, v in parsed.items()}
        pk_t: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for t, v in parsed.items():
            boxes = []
            for o in v:
                if o["type"].lower().startswith(("car", "bus", "tru")) and int(o.get("fixed", 0)) == 1:
                    wx, wy = float(o["wx"]), float(o["wy"])
                    rect = np.array([[wx - 1.9, wy - 0.9], [wx + 1.9, wy - 0.9], [wx + 1.9, wy + 0.9], [wx - 1.9, wy + 0.9], [wx - 1.9, wy - 0.9]], dtype=np.float64)
                    at = np.zeros(8, dtype=np.float32)
                    at[[4, 6]] = 1.0  # Explicit solid parked vehicle closed contour (ch 4 & 6)
                    boxes.append((rect, at))
            if boxes:
                pk_t[t] = boxes
        clip_parked[c] = pk_t

        all_p = [xy for td in clip_snaps[c].values() for xy in td.values()]
        clip_train_pts[c] = np.asarray(all_p, dtype=np.float64) if all_p else np.zeros((0, 2), dtype=np.float64)

    pt0_ego = np.stack([clip_snaps[r.clip][int(r.t0)][int(r.person_id)] for r in rows], axis=0)
    p_bug = np.zeros((n, 16, 10, 11), dtype=np.float32)
    p_fix = np.zeros((n, 16, 10, 11), dtype=np.float32)
    p_solid = np.zeros((n, 16, 10, 11), dtype=np.float32)
    p_trav = np.zeros((n, 16, 10, 11), dtype=np.float32)
    m_map = np.zeros(n, dtype=np.float32)
    m_solid = np.zeros(n, dtype=np.float32)

    rater_a_lbls: list[int] = []  # 0=solid_barrier, 1=traversable_edge, 2=uncertain
    rater_b_lbls: list[int] = []

    with np.load(io.BytesIO(raw_map), allow_pickle=False) as md:
        m_clips, m_pids, m_t0s = md["clip"].tolist(), md["person_id"].tolist(), md["t0"].tolist()
        m_lines, m_elig = md["p_env_image"], md["source_eligible"]
        by_clip: dict[str, list[tuple[int, int, int]]] = {}
        for j, (mc, mp, mt) in enumerate(zip(m_clips, m_pids, m_t0s)):
            if m_elig[j]:
                by_clip.setdefault(str(mc), []).append((int(mt), int(mp), j))
        for mc in by_clip:
            by_clip[mc].sort()

        for i, r in enumerate(rows):
            cands = [x for x in by_clip.get(r.clip, []) if x[0] <= int(r.t0)]
            if not cands:
                continue
            # 1) Legacy buggy double-crop: only took last single person's 16-line local crop
            mt1, mp1, mj1 = cands[-1]
            ref1 = clip_snaps[r.clip].get(mt1, {}).get(mp1)
            if ref1 is not None:
                lk1 = m_lines[mj1]
                pl1 = [(lk1[pk, lk1[pk, :, 10] > 0.5, 0:2].astype(np.float64) + ref1[None, :], lk1[pk, 0, 2:10]) for pk in range(lk1.shape[0]) if (lk1[pk, :, 10] > 0.5).sum() >= 2]
                p_bug[i], _ = build_local_polylines(pl1, pt0_ego[i], map_available=True)

            # 2) Fixed un-cropped clip scene union across ALL causal <= t0 entries + causal parked cars
            raw_union: list[tuple[np.ndarray, np.ndarray]] = []
            for mt, mp, mj in cands:
                ref = clip_snaps[r.clip].get(mt, {}).get(mp)
                if ref is None:
                    continue
                lk = m_lines[mj]
                for pk in range(lk.shape[0]):
                    vp = lk[pk, :, 10] > 0.5
                    if vp.sum() >= 2:
                        raw_union.append((lk[pk, vp, 0:2].astype(np.float64) + ref[None, :], lk[pk, 0, 2:10]))
            for pk_box in clip_parked[r.clip].get(int(r.t0), []):
                raw_union.append(pk_box)
            dedup = _dedup_world_polylines(raw_union)
            p_fix[i], m_map[i] = build_local_polylines(dedup, pt0_ego[i], map_available=True)

            # 3) Dual independent blind classification (0=solid_barrier, 1=traversable_edge, 2=uncertain)
            causal_pts_l = [xy for t_k, d_k in clip_snaps[r.clip].items() if t_k <= int(r.t0) for xy in d_k.values()]
            causal_pts = np.asarray(causal_pts_l, dtype=np.float64) if causal_pts_l else np.zeros((0, 2), dtype=np.float64)
            # Build strictly causal (<= t0) historical pedestrian step segments in this clip
            ped_hist_tracks: dict[int, list[np.ndarray]] = {}
            for t_k in sorted(k for k in clip_snaps[r.clip] if k <= int(r.t0)):
                for pid_k, xy_k in clip_snaps[r.clip][t_k].items():
                    ped_hist_tracks.setdefault(pid_k, []).append(xy_k)
            causal_segs_a, causal_segs_b = [], []
            for pts_tr in ped_hist_tracks.values():
                if len(pts_tr) >= 2:
                    arr_tr = np.asarray(pts_tr, dtype=np.float64)
                    step_d = np.linalg.norm(arr_tr[1:] - arr_tr[:-1], axis=-1)
                    ok_s = step_d <= 1.5
                    if ok_s.any():
                        causal_segs_a.append(arr_tr[:-1][ok_s])
                        causal_segs_b.append(arr_tr[1:][ok_s])
            cs_a = np.concatenate(causal_segs_a, axis=0) if causal_segs_a else np.zeros((0, 2), dtype=np.float64)
            cs_b = np.concatenate(causal_segs_b, axis=0) if causal_segs_b else np.zeros((0, 2), dtype=np.float64)

            h_box = clip_hull_3d[r.clip]
            x_lo, x_hi = float(h_box[:, 0].min()) - 2.0, float(h_box[:, 0].max()) + 2.0
            y_lo, y_hi = float(h_box[:, 1].min()) - 2.0, float(h_box[:, 1].max()) + 2.0
            solid_list, trav_list = [], []
            for xy, attr in dedup:
                seg_len = float(np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum())
                is_parked = (float(attr[4]) > 0.5) or (float(attr[1]) > 0.5)
                is_hull = float(attr[5]) > 0.5
                in_cal = bool((xy[:, 0] >= x_lo).all() and (xy[:, 0] <= x_hi).all() and (xy[:, 1] >= y_lo).all() and (xy[:, 1] <= y_hi).all())
                near_ped_interior = False
                if len(causal_pts) >= 5:
                    d_mat = np.linalg.norm(xy[:, None, :] - causal_pts[None, :, :], axis=-1)
                    near_ped_interior = int((d_mat < 0.80).sum(axis=1).max()) >= 3
                crossed_causally = False
                if len(cs_a) > 0 and len(xy) >= 2:
                    pa, pb = xy[:-1][:, None, :], xy[1:][:, None, :]
                    qa, qb = cs_a[None, :, :], cs_b[None, :, :]
                    da, db, dp_c = pb - pa, qb - qa, qa - pa
                    den = da[..., 0] * db[..., 1] - da[..., 1] * db[..., 0]
                    ok_d = np.abs(den) > 1e-9
                    t_c = np.where(ok_d, (db[..., 0] * dp_c[..., 1] - db[..., 1] * dp_c[..., 0]) / np.where(ok_d, den, 1.0), -1.0)
                    u_c = np.where(ok_d, (da[..., 0] * dp_c[..., 1] - da[..., 1] * dp_c[..., 0]) / np.where(ok_d, den, 1.0), -1.0)
                    crossed_causally = bool(np.any(ok_d & (t_c >= 0.0) & (t_c <= 1.0) & (u_c >= 0.0) & (u_c <= 1.0)))
                # Rater A: physical parked-obstacle / point-cloud occupancy clearance
                ca = 0 if (is_parked or (not is_hull and not near_ped_interior and seg_len >= 2.0)) else (1 if (is_hull or near_ped_interior) else 2)
                # Rater B: strict calibration-domain support & zero causal (<= t0) pedestrian trajectory crossings
                cb = 1 if crossed_causally else (0 if (in_cal and (is_parked or (not is_hull and seg_len >= 2.5 and not near_ped_interior))) else (1 if near_ped_interior else 2))
                rater_a_lbls.append(ca)
                rater_b_lbls.append(cb)
                if ca == 0 and cb == 0:
                    solid_list.append((xy, attr))
                elif ca == 1 or cb == 1:
                    trav_list.append((xy, attr))
            if solid_list:
                p_solid[i], m_solid[i] = build_local_polylines(solid_list, pt0_ego[i], map_available=True)
            else:
                m_solid[i] = 1.0  # Scene audited, 0 solid barriers within 20 m
            if trav_list:
                p_trav[i], _ = build_local_polylines(trav_list, pt0_ego[i], map_available=True)

    ra, rb = np.asarray(rater_a_lbls), np.asarray(rater_b_lbls)
    po = float((ra == rb).mean()) if len(ra) else 1.0
    pe = float(sum(((ra == k).mean() * (rb == k).mean()) for k in (0, 1, 2))) if len(ra) else 1.0
    kappa = (po - pe) / max(1.0 - pe, 1e-8)
    cnt_bug = (p_bug[:, :, :, 10].sum(axis=2) >= 2).sum(axis=1)
    cnt_fix = (p_fix[:, :, :, 10].sum(axis=2) >= 2).sum(axis=1)
    cnt_sol = (p_solid[:, :, :, 10].sum(axis=2) >= 2).sum(axis=1)
    map_diag = {
        "map_eligible_windows": int((m_map > 0.5).sum()),
        "windows_gaining_polylines_after_uncrop_fix": int((cnt_fix > cnt_bug).sum()),
        "mean_polylines_buggy_double_crop": float(cnt_bug[m_map > 0.5].mean()),
        "mean_polylines_fixed_uncropped": float(cnt_fix[m_map > 0.5].mean()),
        "windows_with_dual_confirmed_solid_barriers": int((cnt_sol > 0).sum()),
        "dual_rater_agreement_rate": po,
        "dual_rater_cohen_kappa": float(kappa),
        "segment_counts_consensus": {
            "solid_non_traversable": int(((ra == 0) & (rb == 0)).sum()),
            "traversable_edge": int(((ra == 1) | (rb == 1)).sum()),
            "uncertain_excluded": int(((ra == 2) & (rb == 2)).sum()),
        },
    }
    return {"p_bug": p_bug, "p_fix": p_fix, "p_solid": p_solid, "p_trav": p_trav, "m_map": m_map, "m_solid": m_solid}, map_diag



def _fwd_poly_arm(net: PPTMainlineO2Forecaster, gd: dict, ids_g: torch.Tensor, p_env_g: torch.Tensor | None, m_map_g: torch.Tensor | None) -> dict[str, torch.Tensor]:
    return net(
        gd["history"][ids_g], gd["bio_even"][ids_g], gd["pose_vec_world"][ids_g], gd["signed_knee"][ids_g],
        gd["quality"][ids_g], gd["m_pose"][ids_g], p_env=p_env_g, m_map=m_map_g,
        edges=gd["edges"][ids_g], is_car=gd["is_car"][ids_g], nbr_mask=gd["nbr_mask"][ids_g],
        obs_dt=gd["obs_dt"][ids_g], fut_times_s=gd["fut_s"][ids_g],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--causal-map-bin", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = load_ppt_cohort_with_causal_maps(args.meta_dir, args.pose_npz, args.joint_zip, args.causal_map_bin)
    blind = blind_pinns_physical_emergence_audit(args.meta_dir, args.pose_npz, dp)
    maps, map_diag = build_uncropped_and_dual_confirmed_maps(args.meta_dir, args.joint_zip, args.causal_map_bin)

    hp_np, tp_np = dp["history"].numpy(), dp["truth"].numpy()
    n = len(hp_np)
    # Ground-truth crossing rates on each boundary category (before any model training!)
    m_elig = maps["m_map"] > 0.5
    tp_rel = tp_np - hp_np[:, -1:, :]
    map_diag["gt_crossing_rate_all_1482"] = {
        "buggy_double_crop_auto": polyline_crossing_rate_np(tp_rel, maps["p_bug"]),
        "fixed_uncropped_auto": polyline_crossing_rate_np(tp_rel, maps["p_fix"]),
        "dual_confirmed_solid_barriers": polyline_crossing_rate_np(tp_rel, maps["p_solid"]),
        "traversable_edges_only": polyline_crossing_rate_np(tp_rel, maps["p_trav"]),
    }

    gdp = {k: dp[k].to(dev) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask")}
    gdp["obs_dt"] = torch.full((n, 4), 0.1, dtype=torch.float64, device=dev)
    gdp["fut_s"] = torch.arange(1, 31, dtype=torch.float64, device=dev)[None].expand(n, -1) * 0.1
    p_gpu = {k: torch.from_numpy(v).to(dev) for k, v in maps.items()}

    arms = ("Arm0_No_Map", "Arm1_Buggy_DoubleCrop_Auto", "Arm2_Fixed_Uncropped_Auto", "Arm3_DualConfirmed_Solid_Barriers")
    err_s = {a: np.zeros((len(args.seeds), n, 3), dtype=np.float64) for a in arms}  # [ade1, fde1, minfde6]
    cross_s = {a: np.zeros((len(args.seeds), n), dtype=np.float64) for a in arms}
    site_arr, clip_arr = dp["site"], dp["clip"]

    for si, seed in enumerate(args.seeds):
        for g in sorted(set(site_arr.tolist())):
            tr, te = np.flatnonzero(site_arr != g), np.flatnonzero(site_arr == g)
            w_tr = make_site_weights(site_arr, tr).to(dev)
            tr_g, te_g = torch.from_numpy(tr).to(dev), torch.from_numpy(te).to(dev)

            # Train shared base M2b (motion + pose + neighbor interaction, no map) first
            torch.manual_seed(seed)
            base = PPTMainlineO2Forecaster("M2b").to(dev)
            opt0 = torch.optim.AdamW(base.parameters(), lr=1.6e-3, weight_decay=2e-4)
            base.train()
            for ep in range(42):
                perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed * 701 + ep)).to(dev)
                for cg in perm.split(64):
                    l0 = o2_joint_loss(_fwd_poly_arm(base, gdp, tr_g[cg], None, None), gdp["truth"][tr_g[cg]], w_tr[cg])
                    if l0 is not None:
                        opt0.zero_grad(set_to_none=True); l0.backward(); opt0.step()
            base.eval()

            for arm_nm, pk, mk in (
                ("Arm0_No_Map", None, None),
                ("Arm1_Buggy_DoubleCrop_Auto", "p_bug", "m_map"),
                ("Arm2_Fixed_Uncropped_Auto", "p_fix", "m_map"),
                ("Arm3_DualConfirmed_Solid_Barriers", "p_solid", "m_solid"),
            ):
                if pk is None:
                    net = base
                else:
                    torch.manual_seed(seed + 17)
                    net = PPTMainlineO2Forecaster("M2").to(dev)
                    net.load_state_dict(base.state_dict(), strict=True)
                    for pn, pp in net.named_parameters():
                        pp.requires_grad = pn.startswith("poly_") or pn in (
                            "central_dx.weight", "central_dx.bias", "central_dy.weight",
                            "pair_knots.weight", "pair_knots.bias",
                            "central_scores.weight", "central_scores.bias",
                            "pair_score.weight", "pair_score.bias",
                        )
                    optp = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=9e-4, weight_decay=3e-4)
                    net.train()
                    for ep in range(24):
                        perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed * 911 + ep)).to(dev)
                        for cg in perm.split(64):
                            ids_c = tr_g[cg]
                            lp = o2_joint_loss(_fwd_poly_arm(net, gdp, ids_c, p_gpu[pk][ids_c], p_gpu[mk][ids_c]), gdp["truth"][ids_c], w_tr[cg])
                            if lp is not None:
                                optp.zero_grad(set_to_none=True); lp.backward(); optp.step()
                    net.eval()
                with torch.inference_mode():
                    ot = _fwd_poly_arm(net, gdp, te_g, None if pk is None else p_gpu[pk][te_g], None if mk is None else p_gpu[mk][te_g])
                    pw = ot["paths"].cpu().numpy()
                    i1 = ot["logits"].argmax(1).cpu().numpy()
                d_all = np.linalg.norm(pw - tp_np[te, None], axis=-1)
                d1 = d_all[np.arange(len(te)), i1]
                err_s[arm_nm][si, te] = np.column_stack((d1.mean(1), d1[:, -1], d_all[:, :, -1].min(1)))
                for ii, ti in enumerate(te):
                    cross_s[arm_nm][si, ti] = polyline_crossing_rate_np(pw[ii : ii + 1, i1[ii]] - hp_np[ti : ti + 1, -1:], maps["p_solid"][ti : ti + 1])
        print(f"[W21-MAP] Finished seed={seed}", flush=True)

    em = {a: err_s[a].mean(0) for a in arms}
    near_solid = np.zeros(n, dtype=bool)
    for i in range(n):
        vp = maps["p_solid"][i, :, :, 10] > 0.5
        if vp.any():
            near_solid[i] = float(np.linalg.norm(maps["p_solid"][i, :, :, 0:2][vp], axis=-1).min()) <= 6.0

    strata_out = {}
    for sk, sm in (("all_1482", np.ones(n, dtype=bool)), ("map_eligible_835", m_elig), ("near_confirmed_solid_le6m", near_solid), ("true_occluder_emergence", blind["e_true_all"])):
        if int(sm.sum()) < 5:
            continue
        gs = sorted(set(site_arr[sm].tolist()))
        mods = {a: {"ade1_m": float(np.mean([em[a][(site_arr == g) & sm, 0].mean() for g in gs])),
                    "fde1_m": float(np.mean([em[a][(site_arr == g) & sm, 1].mean() for g in gs])),
                    "minfde6_m": float(np.mean([em[a][(site_arr == g) & sm, 2].mean() for g in gs])),
                    "solid_barrier_cross_rate": float(cross_s[a].mean(0)[sm].mean())} for a in arms}
        strata_out[sk] = {
            "n_windows": int(sm.sum()),
            "models": mods,
            "paired_top1_ci": {
                "fixed_auto_minus_buggy_auto": safe_site_clip_ci(em["Arm2_Fixed_Uncropped_Auto"][sm, :2] - em["Arm1_Buggy_DoubleCrop_Auto"][sm, :2], site_arr[sm], clip_arr[sm]),
                "fixed_auto_minus_no_map": safe_site_clip_ci(em["Arm2_Fixed_Uncropped_Auto"][sm, :2] - em["Arm0_No_Map"][sm, :2], site_arr[sm], clip_arr[sm]),
                "dual_solid_minus_no_map": safe_site_clip_ci(em["Arm3_DualConfirmed_Solid_Barriers"][sm, :2] - em["Arm0_No_Map"][sm, :2], site_arr[sm], clip_arr[sm]),
                "dual_solid_minus_fixed_auto": safe_site_clip_ci(em["Arm3_DualConfirmed_Solid_Barriers"][sm, :2] - em["Arm2_Fixed_Uncropped_Auto"][sm, :2], site_arr[sm], clip_arr[sm]),
            },
        }

    summary = {"experiment": "wave21_polyline_crop_fix_and_dual_boundary", "map_and_rater_diagnostics": map_diag, "pinns_4site_loso": strata_out}
    raw = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = (json.dumps({"summary.json": hashlib.sha256(raw).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw); zf.writestr("sha256.json", sh)
    print("Saved:", args.zip_path, flush=True)


if __name__ == "__main__":
    main()
