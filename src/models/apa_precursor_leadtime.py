"""Wave 22: Causal Pre-Onset Lead-Time (APA) Detector & Delayed-Onset Maneuver Trajectory Selector."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAPAPrecursorLeadNet(nn.Module):
    """Lightweight causal 5-frame O(2)/2D pose+kinematics+interaction precursor detector for Start/Stop/Turn."""

    def __init__(self, kin_dim: int, pose_dim: int, nbr_dim: int, use_pose: bool = True, use_nbr: bool = True) -> None:
        super().__init__()
        self.use_pose = use_pose
        self.use_nbr = use_nbr
        self.kin_gru = nn.GRU(kin_dim, 40, batch_first=True)
        self.pose_gru = nn.GRU(pose_dim, 32, batch_first=True)
        self.delta_mlp = nn.Sequential(nn.Linear(pose_dim * 2, 32), nn.SiLU(), nn.Linear(32, 32))
        self.q_gate = nn.Linear(2, 1)
        nn.init.constant_(self.q_gate.weight, 0.8)
        nn.init.constant_(self.q_gate.bias, 0.0)
        self.nbr_mlp = nn.Sequential(nn.Linear(nbr_dim, 24), nn.SiLU(), nn.Linear(24, 24))
        self.head = nn.Sequential(
            nn.LayerNorm(40 + 32 + 24),
            nn.Linear(40 + 32 + 24, 48),
            nn.SiLU(),
            nn.Linear(48, 4),  # [Start, Stop, Turn_Any, onset_delay_norm]
        )

    def forward(
        self,
        kin_seq: torch.Tensor,
        pose_seq: torch.Tensor,
        q_pair: torch.Tensor,
        nbr_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = kin_seq.size(0)
        _, hk_t = self.kin_gru(kin_seq)
        hk = hk_t[0]
        hp = hk.new_zeros((bsz, 32))
        if self.use_pose:
            _, hp_t = self.pose_gru(pose_seq)
            # Explicit biomechanical APA velocity & net 5-frame postural shift (torso lean + coronal rotation + ankle sep)
            d_net = pose_seq[:, -1] - pose_seq[:, 0]
            d_last = pose_seq[:, -1] - pose_seq[:, -2]
            apa = self.delta_mlp(torch.cat((d_net, d_last), dim=-1))
            w_q = torch.sigmoid(F.linear(q_pair, F.softplus(self.q_gate.weight), self.q_gate.bias))
            hp = w_q * (hp_t[0] + apa)
        hn = self.nbr_mlp(nbr_vec) if self.use_nbr else hk.new_zeros((bsz, 24))
        raw = self.head(torch.cat((hk, hp, hn), dim=-1))
        logits3 = raw[:, :3]
        delay_s = 2.0 * torch.sigmoid(raw[:, 3])  # Predicted maneuver onset delay in [0, 2.0] seconds
        return logits3, delay_s


def calibrate_matched_fpr_thresholds(
    probs_src_neg: np.ndarray,
    target_fpr: float = 0.05,
) -> np.ndarray:
    """Calibrate per-class score thresholds strictly on source-only non-maneuver negative windows for exact FPR."""
    n_cls = probs_src_neg.shape[1]
    thr = np.full(n_cls, 0.50, dtype=np.float64)
    if len(probs_src_neg) >= 5:
        q = float(np.clip(1.0 - target_fpr, 0.50, 0.995))
        for c in range(n_cls):
            thr[c] = float(np.quantile(probs_src_neg[:, c], q))
    return thr


def score_event_lead_times_at_matched_fpr(
    probs_test: np.ndarray,
    rel_tau_s: np.ndarray,
    event_id: np.ndarray,
    event_cls: np.ndarray,
    is_neg_ctrl: np.ndarray,
    thr_src: np.ndarray,
) -> dict[str, dict]:
    """Evaluate empirical test FPR, pre-onset recall (tau <= 0), and earliest lead time (-tau_first_hit in s)."""
    names = ("Start", "Stop", "Turn")
    out: dict[str, dict] = {}
    neg_p = probs_test[is_neg_ctrl]
    for c, nm in enumerate(names):
        emp_fpr = float((neg_p[:, c] >= thr_src[c]).mean()) if len(neg_p) > 0 else 0.0
        ev_mask = (~is_neg_ctrl) & (event_cls == c)
        eids = sorted(set(event_id[ev_mask].tolist()))
        leads_pre: list[float] = []
        leads_any: list[float] = []
        pre_hits = 0
        any_hits = 0
        for eid in eids:
            m_e = ev_mask & (event_id == eid)
            taus = rel_tau_s[m_e]
            ps = probs_test[m_e, c]
            order = np.argsort(taus)
            taus, ps = taus[order], ps[order]
            hit_idx = np.flatnonzero(ps >= thr_src[c])
            if len(hit_idx) > 0:
                any_hits += 1
                t_first = float(taus[int(hit_idx[0])])
                leads_any.append(-t_first)  # Positive = detected BEFORE kinematic onset!
                if t_first <= 1e-6:
                    pre_hits += 1
                    leads_pre.append(-t_first)
            else:
                leads_any.append(-0.60)  # Censored miss penalty after +0.6s
        n_ev = len(eids)
        out[nm] = {
            "n_events": n_ev,
            "empirical_test_fpr": emp_fpr,
            "pre_onset_hit_count": pre_hits,
            "pre_onset_recall": float(pre_hits / max(n_ev, 1)),
            "any_window_recall": float(any_hits / max(n_ev, 1)),
            "mean_lead_s_given_pre_hit": float(np.mean(leads_pre)) if leads_pre else 0.0,
            "mean_lead_s_censored_all": float(np.mean(leads_any)) if leads_any else 0.0,
            "per_event_lead_s": leads_any,
        }
    return out


def build_delayed_onset_start_stop_candidates(
    hist_xy: np.ndarray,
    coronal_facing_xy: np.ndarray,
    pred_delay_s: np.ndarray,
    fut_s: np.ndarray,
    dt_obs: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct delayed-onset APA Start (along body coronal facing) and Stop braking curves."""
    n = len(hist_xy)
    v_vec = (hist_xy[:, -1] - hist_xy[:, -2]) / max(dt_obs, 1e-4)
    spd = np.linalg.norm(v_vec, axis=-1, keepdims=True)
    # Direction: coronal facing when stationary, blended with velocity when moving
    fn = np.linalg.norm(coronal_facing_xy, axis=-1, keepdims=True)
    f_dir = np.where(fn > 1e-4, coronal_facing_xy / np.maximum(fn, 1e-6), np.array([[1.0, 0.0]]))
    v_dir = np.where(spd > 0.25, v_vec / np.maximum(spd, 1e-6), f_dir)
    d_s = np.clip(pred_delay_s, 0.0, 1.6)[:, None]
    tau_eff = np.maximum(fut_s - d_s, 0.0)  # [N, T_fut]: zero motion before predicted onset delay!
    # Smooth human gait initiation: a = 1.1 m/s^2 capped at 1.15 m/s cruise
    s_dist = np.where(tau_eff < 1.05, 0.55 * (tau_eff ** 2), 0.606 + 1.15 * (tau_eff - 1.05))
    start_paths = hist_xy[:, -1:, :] + s_dist[:, :, None] * f_dir[:, None, :]
    # Delayed braking to standstill after d_s: cruise until d_s, then exponential stop over 0.45 s
    pre_d = np.minimum(fut_s, d_s) * spd
    post_t = np.maximum(fut_s - d_s, 0.0)
    post_d = spd * 0.45 * (1.0 - np.exp(-post_t / 0.45))
    stop_paths = hist_xy[:, -1:, :] + (pre_d + post_d)[:, :, None] * v_dir[:, None, :]
    return start_paths, stop_paths


def evaluate_iso_test_fpr_and_unconflated_leadtime(
    probs_test: np.ndarray,
    rel_tau_s: np.ndarray,
    event_id: np.ndarray,
    event_cls: np.ndarray,
    is_neg_ctrl: np.ndarray,
    target_test_fpr: float = 0.05,
    explicit_thr: np.ndarray | None = None,
) -> dict[str, dict]:
    """Evaluate unconflated pre-onset recall, miss count, and hit-only alarm lead time at exact test FPR.

    Never mixes censored miss penalties into alarm timing: reports hit count, miss count, recall,
    and conditional mean/median alarm lead time (ms before onset) strictly on detected events.
    """
    names = ("Start", "Stop", "Turn")
    neg_p = probs_test[is_neg_ctrl]
    if explicit_thr is not None:
        thr = np.asarray(explicit_thr, dtype=np.float64)
    else:
        thr = np.full(probs_test.shape[1], 0.50, dtype=np.float64)
        if len(neg_p) >= 5:
            q = float(np.clip(1.0 - target_test_fpr, 0.50, 0.998))
            for c in range(probs_test.shape[1]):
                thr[c] = float(np.quantile(neg_p[:, c], q))
                # Tie-break upward if discrete ties push empirical FPR > target_test_fpr + 0.015
                if (neg_p[:, c] >= thr[c]).mean() > target_test_fpr + 0.015:
                    thr[c] = float(np.nextafter(thr[c], np.inf))

    out: dict[str, dict] = {}
    for c, nm in enumerate(names):
        emp_fpr = float((neg_p[:, c] >= thr[c]).mean()) if len(neg_p) > 0 else 0.0
        ev_mask = (~is_neg_ctrl) & (event_cls == c)
        eids = sorted(set(event_id[ev_mask].tolist()))
        pre_lead_by_eid: dict[int, float] = {}
        any_hits = 0
        for eid in eids:
            m_e = ev_mask & (event_id == eid)
            order = np.argsort(rel_tau_s[m_e])
            taus = rel_tau_s[m_e][order]
            ps = probs_test[m_e, c][order]
            hits = np.flatnonzero(ps >= thr[c])
            if len(hits) > 0:
                any_hits += 1
                t_first = float(taus[int(hits[0])])
                if t_first <= 1e-6:
                    pre_lead_by_eid[int(eid)] = -t_first * 1000.0  # ms BEFORE kinematic onset (>= 0)
        n_ev = len(eids)
        pre_hits = len(pre_lead_by_eid)
        pre_vals = list(pre_lead_by_eid.values())
        out[nm] = {
            "n_events": n_ev,
            "threshold": float(thr[c]),
            "empirical_test_fpr": emp_fpr,
            "pre_onset_hit_count": pre_hits,
            "pre_onset_miss_count": n_ev - pre_hits,
            "pre_onset_recall": float(pre_hits / max(n_ev, 1)),
            "any_window_hit_count": any_hits,
            "any_window_miss_count": n_ev - any_hits,
            "mean_lead_ms_given_pre_hit": float(np.mean(pre_vals)) if pre_vals else 0.0,
            "median_lead_ms_given_pre_hit": float(np.median(pre_vals)) if pre_vals else 0.0,
            "pre_hit_lead_ms_by_eid": {str(k): v for k, v in pre_lead_by_eid.items()},
            "event_ids_ordered": [int(x) for x in eids],
        }
    return out


def paired_iso_fpr_comparison(
    res_b: dict[str, dict],
    res_a: dict[str, dict],
    seed: int = 2026,
    n_boot: int = 2000,
) -> dict[str, dict]:
    """Compare Model B vs Baseline A at matched FPR without any censored miss penalty.

    Separates:
      1) Paired pre-onset recall gain (hit_B - hit_A) across all N events with bootstrap 95% CI;
      2) Paired alarm-time gain (lead_B - lead_A in ms) strictly on events pre-hit by BOTH models.
    """
    rng = np.random.default_rng(seed)
    comp: dict[str, dict] = {}
    for nm in ("Start", "Stop", "Turn"):
        eb, ea = res_b[nm], res_a[nm]
        eids = eb["event_ids_ordered"]
        n_ev = len(eids)
        hb = np.array([1.0 if str(e) in eb["pre_hit_lead_ms_by_eid"] else 0.0 for e in eids])
        ha = np.array([1.0 if str(e) in ea["pre_hit_lead_ms_by_eid"] else 0.0 for e in eids])
        d_rec = hb - ha
        rec_mean = float(d_rec.mean()) if n_ev > 0 else 0.0
        if n_ev >= 2:
            idx = rng.integers(0, n_ev, size=(n_boot, n_ev))
            rec_ci = [float(x) for x in np.quantile(d_rec[idx].mean(axis=1), [0.025, 0.975])]
        else:
            rec_ci = [rec_mean, rec_mean]

        common = [str(e) for e in eids if (str(e) in eb["pre_hit_lead_ms_by_eid"] and str(e) in ea["pre_hit_lead_ms_by_eid"])]
        only_b = sum(1 for e in eids if (str(e) in eb["pre_hit_lead_ms_by_eid"] and str(e) not in ea["pre_hit_lead_ms_by_eid"]))
        only_a = sum(1 for e in eids if (str(e) not in eb["pre_hit_lead_ms_by_eid"] and str(e) in ea["pre_hit_lead_ms_by_eid"]))
        if len(common) >= 1:
            d_common = np.array([eb["pre_hit_lead_ms_by_eid"][k] - ea["pre_hit_lead_ms_by_eid"][k] for k in common])
            c_mean = float(d_common.mean())
            if len(common) >= 2:
                c_idx = rng.integers(0, len(common), size=(n_boot, len(common)))
                c_ci = [float(x) for x in np.quantile(d_common[c_idx].mean(axis=1), [0.025, 0.975])]
            else:
                c_ci = [c_mean, c_mean]
        else:
            c_mean, c_ci = 0.0, [0.0, 0.0]

        comp[nm] = {
            "n_events": n_ev,
            "fpr_model_b": eb["empirical_test_fpr"],
            "fpr_baseline_a": ea["empirical_test_fpr"],
            "recall_diff_mean": rec_mean,
            "recall_diff_ci95": rec_ci,
            "n_both_pre_hit": len(common),
            "n_only_model_b_pre_hit": int(only_b),
            "n_only_baseline_a_pre_hit": int(only_a),
            "common_hit_lead_diff_ms_mean": c_mean,
            "common_hit_lead_diff_ms_ci95": c_ci,
        }
    return comp
