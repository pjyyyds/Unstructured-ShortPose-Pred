"""Wave 18 GPU1 (Slides 8, 11, 13, 19): Zero-leakage 5-action Platt calibration, F1 thresholds, & candidate reranking."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ppt_mainline_o2_models import PPTMainlineO2Forecaster
from scripts.run_dair_o2_bioeq_external import SEEDS
from scripts.run_jrdb_rc1_pose_quality_and_initiation_5seed import _loc_bootstrap_ci
from scripts.run_jrdb_rc2_rc3_graph_and_actions_5seed import _fwd_gd, _stage_gpu_dict, fast_person_bootstrap_ci, load_jrdb_10scene_rc2_rc3
from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_pinns_hybrid_action_events_5seed import score_action_matrix
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import _copy_matching, load_ppt_cohort_with_causal_maps
from scripts.run_wave18_conflict_gated_interaction_5seed import safe_site_clip_ci
from src.data.actions import compute_action_labels
from src.models.bioeq_o2_joint import o2_joint_loss


def compute_ece_binary(y_true: np.ndarray, probs: np.ndarray, valid: np.ndarray, n_bins: int = 10) -> float:
    m = valid.astype(bool)
    if not m.any():
        return 0.0
    yt, pr = y_true[m].astype(np.float64), np.clip(probs[m].astype(np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        bm = (pr >= edges[b]) & (pr <= edges[b + 1] if b == n_bins - 1 else pr < edges[b + 1])
        if bm.any():
            ece += (bm.sum() / len(yt)) * abs(float(yt[bm].mean() - pr[bm].mean()))
    return float(ece)


def fit_platt_and_thresholds(oof_logits: np.ndarray, y_act: np.ndarray, valid_act: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit per-class Platt (a_c * logit + b_c) and optimal F1 threshold strictly on source OOF validation logits."""
    a_arr = np.ones(5, dtype=np.float64)
    b_arr = np.zeros(5, dtype=np.float64)
    thr_arr = np.full(5, 0.5, dtype=np.float64)
    for c in range(5):
        vm = valid_act[:, c].astype(bool)
        if vm.sum() < 10 or y_act[vm, c].sum() < 2 or (~y_act[vm, c]).sum() < 2:
            continue
        x = torch.from_numpy(oof_logits[vm, c].astype(np.float32))
        y = torch.from_numpy(y_act[vm, c].astype(np.float32))
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)
        opt = torch.optim.LBFGS([a, b], lr=0.25, max_iter=40)

        def _closure() -> torch.Tensor:
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(a.clamp(0.1, 5.0) * x + b, y) + 1e-3 * ((a - 1.0) ** 2 + b * b)
            loss.backward()
            return loss

        opt.step(_closure)
        ac = float(a.detach().clamp(0.1, 5.0).item())
        bc = float(b.detach().clamp(-8.0, 8.0).item())
        a_arr[c], b_arr[c] = ac, bc
        p_cal = 1.0 / (1.0 + np.exp(-np.clip(ac * oof_logits[vm, c] + bc, -30.0, 30.0)))
        yt = y_act[vm, c].astype(bool)
        best_t, best_f1 = 0.5, -1.0
        for th in np.linspace(0.08, 0.88, 41):
            pr = p_cal >= th
            tp = int((pr & yt).sum())
            denom = 2 * tp + int((pr & (~yt)).sum()) + int(((~pr) & yt).sum())
            f1 = (2.0 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(th)
        thr_arr[c] = best_t
    return a_arr, b_arr, thr_arr


def score_grouped_actions(y_true: np.ndarray, v_mask: np.ndarray, probs: np.ndarray, grp_arr: np.ndarray, thresh: np.ndarray) -> dict:
    from scripts.run_pinns_o2_joint_actions_5seed import ACTIONS
    from src.metrics.metrics_and_stats import _binary_average_precision

    pooled_cls, eces = {}, []
    for c_i, c_name in enumerate(ACTIONS):
        vm = v_mask[:, c_i].astype(bool)
        yt, ps = y_true[vm, c_i].astype(bool), probs[vm, c_i]
        pr = ps >= thresh[c_i]
        tp, fp, fn = int((pr & yt).sum()), int((pr & (~yt)).sum()), int(((~pr) & yt).sum())
        f1 = (2.0 * tp) / max(2 * tp + fp + fn, 1)
        ap = float(_binary_average_precision(yt.astype(np.float64), ps)) if yt.sum() > 0 else 0.0
        br = float(((ps - yt.astype(np.float64)) ** 2).mean()) if len(yt) else 0.0
        ece = compute_ece_binary(y_true[:, c_i], probs[:, c_i], vm)
        eces.append(ece)
        pooled_cls[c_name] = {"pos": int(yt.sum()), "ap": ap, "f1": float(f1), "tp": tp, "fp": fp, "brier": br, "ece": ece, "thresh": float(thresh[c_i])}

    g_aps, g_f1s, g_brs = [], [], []
    for g in sorted(set(grp_arr.tolist())):
        gm = grp_arr == g
        aps, f1s, brs = [], [], []
        for c_i in range(5):
            vm = gm & v_mask[:, c_i].astype(bool)
            if not vm.any():
                continue
            yt, ps = y_true[vm, c_i].astype(bool), probs[vm, c_i]
            if yt.sum() > 0:
                aps.append(float(_binary_average_precision(yt.astype(np.float64), ps)))
            pr = ps >= thresh[c_i]
            tp, fp, fn = int((pr & yt).sum()), int((pr & (~yt)).sum()), int(((~pr) & yt).sum())
            f1s.append((2.0 * tp) / max(2 * tp + fp + fn, 1))
            brs.append(float(((ps - yt.astype(np.float64)) ** 2).mean()))
        if aps:
            g_aps.append(float(np.mean(aps)))
        if f1s:
            g_f1s.append(float(np.mean(f1s)))
            g_brs.append(float(np.mean(brs)))
    return {
        "group_equal": {"macro_ap": float(np.mean(g_aps)), "macro_f1": float(np.mean(g_f1s)), "mean_brier": float(np.mean(g_brs)), "mean_ece": float(np.mean(eces))},
        "pooled_classes": pooled_cls,
    }


def candidate_implied_actions_np(hist_np: np.ndarray, paths_np: np.ndarray, dt: float) -> np.ndarray:
    """Deterministic causal kinematic action signature [N, 6, 5] for each candidate path."""
    n, k_modes, fut_t, _ = paths_np.shape
    hm = np.ones(hist_np.shape[1], dtype=np.float32)
    fm = np.ones(fut_t, dtype=np.float32)
    out = np.zeros((n, k_modes, 5), dtype=np.float64)
    for i in range(n):
        hp = hist_np[i] - hist_np[i, -1:]
        for k in range(k_modes):
            yk, _ = compute_action_labels(hp, hm, paths_np[i, k], fm, dt=dt)
            out[i, k] = yk
    return out


def rerank_top1_modes(mode_logits: np.ndarray, act_probs: np.ndarray, cand_acts: np.ndarray, beta: float) -> np.ndarray:
    log_pi = mode_logits - np.log(np.exp(mode_logits - mode_logits.max(1, keepdims=True)).sum(1, keepdims=True)) - mode_logits.max(1, keepdims=True)
    if beta <= 1e-12:
        return log_pi.argmax(axis=1)
    p = np.clip(act_probs[:, None, :], 1e-4, 1.0 - 1e-4)
    ll_act = (cand_acts * np.log(p) + (1.0 - cand_acts) * np.log(1.0 - p)).sum(axis=-1)
    return (log_pi + float(beta) * ll_act).argmax(axis=1)


def _fit_m3_gd(seed: int, grp_arr: np.ndarray, gd: dict, tr_ids: np.ndarray, bsz: int, ep_s: tuple[int, int, int, int], lr_s: tuple[float, float, float, float], device: torch.device) -> PPTMainlineO2Forecaster:
    w_tr = make_site_weights(grp_arr, tr_ids).to(device)
    tr_t = torch.from_numpy(tr_ids).to(device)
    truth_tr, ya_tr, va_tr = gd["truth"][tr_t], gd["y_act"][tr_t], gd["valid_act"][tr_t]
    pw = (((~ya_tr) & va_tr).sum(0).float().clamp_min(1.0) / (ya_tr & va_tr).sum(0).float().clamp_min(1.0)).clamp(1.0, 12.0)

    def _tr(net: PPTMainlineO2Forecaster, prefs: tuple[str, ...], ep: int, lr: float, wd: float, aw: float, soff: int) -> PPTMainlineO2Forecaster:
        for nm, p in net.named_parameters():
            p.requires_grad = nm.startswith(prefs) or nm in prefs
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        gen = torch.Generator().manual_seed(seed + soff)
        net.train()
        for _ in range(ep):
            for ch in torch.randperm(len(tr_ids), generator=gen).split(bsz):
                cg = ch.to(device)
                loss = o2_joint_loss(_fwd_gd(net, gd, tr_t[cg]), truth_tr[cg], w_tr[cg], ya_tr[cg] if aw > 0 else None, va_tr[cg] if aw > 0 else None, pw if aw > 0 else None, act_weight=aw)
                if loss is not None:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return net.eval()

    torch.manual_seed(seed)
    m0 = _tr(PPTMainlineO2Forecaster("M0").to(device), ("motion.", "ctx_proj.", "cv_low_bonus", "central_", "pair_"), ep_s[0], lr_s[0], 1e-4, 0.0, 0)
    torch.manual_seed(seed); m1 = PPTMainlineO2Forecaster("M1").to(device); _copy_matching(m1, m0)
    m1 = _tr(m1, ("pose_", "gate_", "central_", "pair_", "cv_low_bonus"), ep_s[1], lr_s[1], 3e-4, 0.0, 10)
    torch.manual_seed(seed); m2b = PPTMainlineO2Forecaster("M2b").to(device); _copy_matching(m2b, m1)
    m2b = _tr(m2b, ("edge_", "graph_proj", "null_logit"), ep_s[2], lr_s[2], 3e-4, 0.0, 21)
    torch.manual_seed(seed); m3 = PPTMainlineO2Forecaster("M3").to(device); _copy_matching(m3, m2b)
    return _tr(m3, ("ctx_proj.", "act_", "bridge_", "central_", "pair_", "cv_low_bonus"), ep_s[3], lr_s[3], 2e-4, 0.25, 32)


def run_action_rerank_benchmark(
    tag: str, grp_arr: np.ndarray, unit_arr: np.ndarray, is_pinns: bool, gd: dict, hist_np: np.ndarray,
    truth_np: np.ndarray, y_act_np: np.ndarray, v_act_np: np.ndarray, dt: float, hor_steps: tuple[int, int, int],
    bsz: int, ep_s: tuple[int, int, int, int], lr_s: tuple[float, float, float, float], device: torch.device,
) -> tuple[dict, dict[str, np.ndarray]]:
    n = len(truth_np)
    groups = sorted(set(grp_arr.tolist()))
    betas = (0.0, 0.05, 0.10, 0.20, 0.35)
    raw_p5 = np.zeros((len(SEEDS), n, 5))
    cal_p5 = np.zeros((len(SEEDS), n, 5))
    thr_list, beta_list = [], []
    err_base = np.zeros((len(SEEDS), n, 6))  # [ade_h1, fde_h1, ade_h2, fde_h2, ade_h3, fde_h3]
    err_rerank = np.zeros((len(SEEDS), n, 6))
    err_fixed015 = np.zeros((len(SEEDS), n, 6))

    def _hor_err(paths_k: np.ndarray, idx_top: np.ndarray, te_ids: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(paths_k[np.arange(len(te_ids)), idx_top] - truth_np[te_ids], axis=-1)
        h1, h2, h3 = hor_steps
        return np.column_stack((d[:, :h1].mean(1), d[:, h1 - 1], d[:, :h2].mean(1), d[:, h2 - 1], d[:, :h3].mean(1), d[:, h3 - 1]))

    for si, seed in enumerate(SEEDS):
        for g in groups:
            tr_ids = np.flatnonzero(grp_arr != g)
            te_ids = np.flatnonzero(grp_arr == g)
            src_g = sorted(set(grp_arr[tr_ids].tolist()))
            n_in = min(4, len(src_g))
            val_sets = [src_g[k::n_in] for k in range(n_in)]
            oof_al = np.zeros((len(tr_ids), 5), dtype=np.float64)
            oof_ml = np.zeros((len(tr_ids), 6), dtype=np.float64)
            oof_pk = np.zeros((len(tr_ids), 6, truth_np.shape[1], 2), dtype=np.float64)

            for v_locs in val_sets:
                vm_loc = np.isin(grp_arr[tr_ids], v_locs)
                in_tr = tr_ids[~vm_loc]
                in_va = tr_ids[vm_loc]
                m_in = _fit_m3_gd(seed, grp_arr, gd, in_tr, bsz, ep_s, lr_s, device)
                with torch.inference_mode():
                    oo = _fwd_gd(m_in, gd, torch.from_numpy(in_va).to(device))
                    oof_al[vm_loc] = oo["act_logits"].cpu().numpy()
                    oof_ml[vm_loc] = oo["logits"].cpu().numpy()
                    oof_pk[vm_loc] = oo["paths"].cpu().numpy()

            a_cal, b_cal, thr_cal = fit_platt_and_thresholds(oof_al, y_act_np[tr_ids], v_act_np[tr_ids])
            thr_list.append(thr_cal.tolist())
            oof_pcal = 1.0 / (1.0 + np.exp(-np.clip(oof_al * a_cal[None, :] + b_cal[None, :], -30.0, 30.0)))
            oof_ca = candidate_implied_actions_np(hist_np[tr_ids], oof_pk, dt=dt)
            b_scores = {}
            for bt in betas:
                idx_b = rerank_top1_modes(oof_ml, oof_pcal, oof_ca, bt)
                b_scores[bt] = float(np.linalg.norm(oof_pk[np.arange(len(tr_ids)), idx_b] - truth_np[tr_ids], axis=-1).mean())
            beta_star = float(min(betas, key=lambda bt: b_scores[bt] + (1e-7 if bt > 0 else 0.0)))
            beta_list.append(beta_star)

            m_full = _fit_m3_gd(seed, grp_arr, gd, tr_ids, bsz, ep_s, lr_s, device)
            with torch.inference_mode():
                oo_te = _fwd_gd(m_full, gd, torch.from_numpy(te_ids).to(device))
                te_al = oo_te["act_logits"].cpu().numpy()
                te_ml = oo_te["logits"].cpu().numpy()
                te_pk = oo_te["paths"].cpu().numpy()
            p_raw = 1.0 / (1.0 + np.exp(-np.clip(te_al, -30.0, 30.0)))
            p_cal = 1.0 / (1.0 + np.exp(-np.clip(te_al * a_cal[None, :] + b_cal[None, :], -30.0, 30.0)))
            raw_p5[si, te_ids], cal_p5[si, te_ids] = p_raw, p_cal
            te_ca = candidate_implied_actions_np(hist_np[te_ids], te_pk, dt=dt)
            err_base[si, te_ids] = _hor_err(te_pk, rerank_top1_modes(te_ml, p_cal, te_ca, 0.0), te_ids)
            err_rerank[si, te_ids] = _hor_err(te_pk, rerank_top1_modes(te_ml, p_cal, te_ca, beta_star), te_ids)
            err_fixed015[si, te_ids] = _hor_err(te_pk, rerank_top1_modes(te_ml, p_cal, te_ca, 0.15), te_ids)
        print(f"[W18-ACT {tag}] seed={seed} base_ade={err_base[si, :, 4].mean():.4f} rerank_ade={err_rerank[si, :, 4].mean():.4f}", flush=True)

    eb, er, ef = err_base.mean(0), err_rerank.mean(0), err_fixed015.mean(0)
    thr_mean = np.mean(np.asarray(thr_list), axis=0)
    act_raw = score_grouped_actions(y_act_np, v_act_np, raw_p5.mean(0), grp_arr, np.full(5, 0.5))
    act_cal05 = score_grouped_actions(y_act_np, v_act_np, cal_p5.mean(0), grp_arr, np.full(5, 0.5))
    act_cal_opt = score_grouped_actions(y_act_np, v_act_np, cal_p5.mean(0), grp_arr, thr_mean)

    def _pair_ci(d2: np.ndarray, msk: np.ndarray) -> dict:
        if int(msk.sum()) < 5:
            return {"n": int(msk.sum())}
        if is_pinns:
            return safe_site_clip_ci(d2[msk], grp_arr[msk], unit_arr[msk])
        p_locs = sorted(set(grp_arr[msk].tolist()))
        loc_ci = _loc_bootstrap_ci(d2[msk], grp_arr[msk]) if len(p_locs) >= 2 else {}
        return {"location_equal": loc_ci, **fast_person_bootstrap_ci(d2, msk, unit_arr, n_boot=10000)}

    man_m = (y_act_np[:, :4] & v_act_np[:, :4]).any(axis=1)
    mnt_m = y_act_np[:, 4] & v_act_np[:, 4]
    h_names = ("h1_short", "h2_mid", "h3_full")
    strata_out = {}
    for sn, msk in (("all", np.ones(n, dtype=bool)), ("active_maneuver_start_stop_turn", man_m), ("maintain_only", mnt_m)):
        if not msk.any():
            continue
        ug = sorted(set(grp_arr[msk].tolist()))
        cell = {"n_windows": int(msk.sum())}
        for hi, hn in enumerate(h_names):
            c0, c1 = 2 * hi, 2 * hi + 2
            cell[hn] = {
                "M3_argmax_prior": {"group_equal_ade_m": float(np.mean([eb[msk & (grp_arr == g), c0].mean() for g in ug])), "group_equal_fde_m": float(np.mean([eb[msk & (grp_arr == g), c0 + 1].mean() for g in ug]))},
                "M3_rerank_inner_beta": {"group_equal_ade_m": float(np.mean([er[msk & (grp_arr == g), c0].mean() for g in ug])), "group_equal_fde_m": float(np.mean([er[msk & (grp_arr == g), c0 + 1].mean() for g in ug]))},
                "M3_rerank_fixed_beta015": {"group_equal_ade_m": float(np.mean([ef[msk & (grp_arr == g), c0].mean() for g in ug])), "group_equal_fde_m": float(np.mean([ef[msk & (grp_arr == g), c0 + 1].mean() for g in ug]))},
                "rerank_inner_minus_prior_ci": _pair_ci(er[:, c0:c1] - eb[:, c0:c1], msk),
                "rerank_beta015_minus_prior_ci": _pair_ci(ef[:, c0:c1] - eb[:, c0:c1], msk),
            }
        strata_out[sn] = cell
    npz = {f"{tag}_raw_p5": raw_p5.mean(0), f"{tag}_cal_p5": cal_p5.mean(0), f"{tag}_err_base": eb, f"{tag}_err_rerank": er}
    return {
        "n_windows": n, "mean_inner_thresholds": thr_mean.tolist(), "inner_selected_betas": beta_list,
        "action_metrics_raw_thr05": act_raw, "action_metrics_platt_thr05": act_cal05, "action_metrics_platt_inner_opt_thr": act_cal_opt,
        "horizon_rerank_strata": strata_out,
    }, npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--causal-map-bin", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, required=True)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dp = load_ppt_cohort_with_causal_maps(args.meta_dir, args.pose_npz, args.joint_zip, args.causal_map_bin)
    np_w = len(dp["history"])
    gdp = {k: dp[k].to(dev) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask", "y_act", "valid_act")}
    gdp["obs_dt"] = torch.full((np_w, 4), 0.1, dtype=torch.float64, device=dev)
    gdp["fut_s"] = torch.arange(1, 31, dtype=torch.float64, device=dev)[None].expand(np_w, -1) * 0.1
    gdp["p_env"] = torch.zeros((np_w, 16, 10, 11), dtype=torch.float32, device=dev)
    gdp["m_map"] = torch.zeros(np_w, dtype=torch.float32, device=dev)
    p_res, p_npz = run_action_rerank_benchmark("pinns", dp["site"], dp["clip"], True, gdp, dp["history"].numpy(), dp["truth"].numpy(), dp["y_act"].numpy(), dp["valid_act"].numpy(), 0.1, (10, 20, 30), 64, (45, 25, 26, 28), (2e-3, 1.2e-3, 1.5e-3, 1.2e-3), dev)

    dj = load_jrdb_10scene_rc2_rc3(args.repo_root)
    gdj = _stage_gpu_dict(dj, dev)
    j_res, j_npz = run_action_rerank_benchmark("jrdb", dj["location"], dj["actors"], False, gdj, dj["history"].numpy(), dj["truth"].numpy(), dj["y_act"].numpy(), dj["valid_act"].numpy(), 0.4, (3, 5, 7), 512, (45, 28, 28, 30), (2.5e-3, 1.5e-3, 1.8e-3, 1.5e-3), dev)

    summary = {"experiment": "wave18_action_calibration_and_rerank_5seed", "device": str(dev), "seeds": list(SEEDS), "pinns_4site_loso_1482": p_res, "jrdb_9location_lolo_5836": j_res}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(args.out_dir / "per_window_action_rerank_5seed.npz", **p_npz, **j_npz)
    dig = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out_dir.iterdir()) if p.is_file()}
    (args.out_dir / "sha256.json").write_text(json.dumps(dig, indent=2), encoding="utf-8")
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(args.out_dir.iterdir()):
            zf.write(p, arcname=p.name)
    print(f"DONE: {args.zip_path} sha256={hashlib.sha256(args.zip_path.read_bytes()).hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
