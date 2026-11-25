"""Five-seed 4-site LOSO O(2) BioEq benchmark across matched T in {5, 10, 20} (623 windows)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from src.models.bioeq_neural import biomech_sequence_torch
from src.models.bioeq_o2_joint import O2BioEqJointForecaster, o2_joint_loss

SEEDS = (42, 1024, 2025, 3407, 4096)
DURATIONS = (5, 10, 20)
SITES = ("america_crossroad", "italy_park", "japan_crossroad", "thailand_street")
BIO_TRAINABLE = (
    "bio_in.weight", "bio_in.bias",
    "bio_gru.weight_ih_l0", "bio_gru.weight_hh_l0", "bio_gru.bias_ih_l0", "bio_gru.bias_hh_l0",
    "bio_par_central.weight", "bio_par_central.bias",
    "bio_par_pair.weight", "bio_par_pair.bias",
    "gate_logit",
)


def load_matched_623(matched_npz: Path, pose_npz: Path) -> dict:
    with np.load(matched_npz, allow_pickle=False) as m_ar:
        m = {k: m_ar[k] for k in m_ar.files}
    with np.load(pose_npz, allow_pickle=False) as p_ar:
        p = {k: p_ar[k] for k in p_ar.files}
    p_map = {(str(c), int(pid), int(t)): i for i, (c, pid, t) in enumerate(zip(p["clip"], p["person_id"], p["t0"]))}
    sel = sorted(
        np.flatnonzero(m["strict_disjoint_20"]).tolist(),
        key=lambda i: (str(m["clip"][i]), int(m["person_id"][i]), int(m["t0"][i])),
    )
    if len(sel) != 623:
        raise ValueError(f"Expected 623 strict matched windows, found {len(sel)}")
    p_idx = np.array([p_map[(str(m["clip"][i]), int(m["person_id"][i]), int(m["t0"][i]))] for i in sel], dtype=int)
    hist20 = m["history_20_xy"][sel].astype(np.float64)
    fut30 = m["future_30_xy"][sel].astype(np.float64)
    origin = hist20[:, -1:, :]
    hist20_rel = hist20 - origin
    fut30_rel = fut30 - origin
    np.testing.assert_allclose(hist20_rel[:, -5:], p["hist_pos_relative"][p_idx], atol=1e-6, rtol=0.0)
    kpts5 = torch.from_numpy(p["hist_kpts"][p_idx].astype(np.float32))
    bbox5 = torch.from_numpy(p["hist_bbox"][p_idx].astype(np.float32))
    bio_seq, q_bio = biomech_sequence_torch(kpts5, bbox5)
    return {
        "clip": m["clip"][sel].astype(str),
        "site": m["site"][sel].astype(str),
        "hist20": torch.from_numpy(hist20_rel),
        "truth30": torch.from_numpy(fut30_rel),
        "bio_seq": bio_seq,
        "q_bio": q_bio,
    }


@torch.inference_mode()
def eval_o2(net: O2BioEqJointForecaster, data: dict, ids: np.ndarray, dur: int, bio_scale: float = 1.0) -> np.ndarray:
    hist = data["hist20"][ids, -dur:]
    bs = data["bio_seq"][ids]
    qb = data["q_bio"][ids] * float(bio_scale)
    out = net(hist, bs, qb)
    paths = out["paths"].cpu().numpy()
    probs = out["logits"].softmax(dim=1).cpu().numpy()
    truth = data["truth30"][ids].numpy()
    dst = np.linalg.norm(paths - truth[:, None], axis=3)
    ade_k, fde_k = dst.mean(axis=2), dst[:, :, -1]
    top = probs.argmax(axis=1)
    ii = np.arange(len(ids))
    bayes = np.einsum("nk,nktw->ntw", probs, paths)
    bd = np.linalg.norm(bayes - truth, axis=2)
    return np.column_stack((ade_k[ii, top], fde_k[ii, top], ade_k.min(1), fde_k.min(1), bd.mean(1), bd[:, -1]))


def fit_motion_and_bio(seed: int, dur: int, data: dict, tr_ids: np.ndarray) -> tuple[O2BioEqJointForecaster, O2BioEqJointForecaster, float]:
    torch.manual_seed(seed)
    mot = O2BioEqJointForecaster(pose_mode="motion", action_mode="none")
    hist = data["hist20"][tr_ids, -dur:]
    truth = data["truth30"][tr_ids]
    bs = data["bio_seq"][tr_ids]
    qb = data["q_bio"][tr_ids]
    sw = make_site_weights(data["site"], tr_ids)
    opt = torch.optim.AdamW(mot.parameters(), lr=0.002, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=28, eta_min=1e-4)
    gen = torch.Generator().manual_seed(seed + dur * 17)
    mot.train()
    for _ in range(28):
        for b in torch.randperm(len(tr_ids), generator=gen).split(64):
            opt.zero_grad(set_to_none=True)
            loss = o2_joint_loss(mot(hist[b]), truth[b], sw[b])
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mot.parameters(), 2.0)
                opt.step()
        sched.step()
    mot.eval()

    def train_bio_on(sub_ids: np.ndarray) -> O2BioEqJointForecaster:
        torch.manual_seed(seed + 101 + dur)
        bio = O2BioEqJointForecaster(pose_mode="bio_pose", action_mode="none")
        st = bio.state_dict()
        for k, v in mot.state_dict().items():
            if not k.startswith("bio_"):
                st[k].copy_(v)
        bio.load_state_dict(st)
        for name, param in bio.named_parameters():
            param.requires_grad_(name in BIO_TRAINABLE)
        h_s, t_s = data["hist20"][sub_ids, -dur:], data["truth30"][sub_ids]
        bs_s, qb_s = data["bio_seq"][sub_ids], data["q_bio"][sub_ids]
        sw_s = make_site_weights(data["site"], sub_ids)
        opt_b = torch.optim.AdamW([p for p in bio.parameters() if p.requires_grad], lr=0.0015, weight_decay=3e-4)
        gen_b = torch.Generator().manual_seed(seed + 33 + dur)
        bio.train()
        for _ in range(18):
            for b in torch.randperm(len(sub_ids), generator=gen_b).split(64):
                opt_b.zero_grad(set_to_none=True)
                loss_b = o2_joint_loss(bio(h_s[b], bs_s[b], qb_s[b]), t_s[b], sw_s[b], reg_only=True)
                if loss_b is not None:
                    loss_b.backward()
                    torch.nn.utils.clip_grad_norm_(bio.parameters(), 1.0)
                    opt_b.step()
        return bio.eval()

    tr_sites = sorted(set(data["site"][tr_ids].tolist()))
    scales = (0.0, 0.25, 0.5, 1.0)
    oof = {sc: [] for sc in scales}
    for vs in tr_sites:
        in_tr = tr_ids[data["site"][tr_ids] != vs]
        in_va = tr_ids[data["site"][tr_ids] == vs]
        b_in = train_bio_on(in_tr)
        for sc in scales:
            oof[sc].append(float(eval_o2(b_in, data, in_va, dur, sc)[:, 0].mean()))
    best_sc = min(scales, key=lambda sc: np.mean(oof[sc]) + (1e-7 if sc > 0 else 0.0))
    bio_full = train_bio_on(tr_ids)
    return mot, bio_full, float(best_sc)


def paired_site_clip_ci(diff: np.ndarray, site: np.ndarray, clip: np.ndarray, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    site_clips = {s: sorted(set(clip[site == s].tolist())) for s in SITES}
    clip_rows = {c: np.flatnonzero(clip == c) for c in set(clip.tolist())}
    pt = np.mean([diff[site == s].mean(axis=0) for s in SITES], axis=0)
    draws = np.empty((20000, 2), dtype=np.float64)
    for r in range(20000):
        vals = []
        for s in SITES:
            cs = site_clips[s]
            picked = [cs[j] for j in rng.integers(0, len(cs), size=len(cs))]
            rows = np.concatenate([clip_rows[c] for c in picked])
            vals.append(diff[rows].mean(axis=0))
        draws[r] = np.mean(vals, axis=0)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return {
        "mean": {"ade_m": float(pt[0]), "fde_m": float(pt[1])},
        "ci95": {"ade_m": [float(lo[0]), float(hi[0])], "fde_m": [float(lo[1]), float(hi[1])]},
    }


def site_equal_summary(arr: np.ndarray, site: np.ndarray) -> dict:
    names = ("ade_m", "fde_m", "minade6_m", "minfde6_m", "bayes_ade_m", "bayes_fde_m")
    by_s = {s: dict(zip(names, arr[site == s].mean(axis=0).tolist())) for s in SITES}
    eq = dict(zip(names, np.mean([arr[site == s].mean(axis=0) for s in SITES], axis=0).tolist()))
    return {"site_equal": eq, "by_site": by_s}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("matched_npz", type=Path)
    ap.add_argument("pose_npz", type=Path)
    ap.add_argument("output_zip", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    torch.set_num_threads(4)
    data = load_matched_623(args.matched_npz, args.pose_npz)
    n = len(data["site"])
    truth_np = data["truth30"].numpy()
    steps = np.arange(1, 31, dtype=np.float64)[None, :, None]
    last2 = (data["hist20"][:, -1:] - data["hist20"][:, -2:-1]).numpy() * steps
    d_cv = np.linalg.norm(last2 - truth_np, axis=2)
    cv_metrics = np.column_stack((d_cv.mean(1), d_cv[:, -1], d_cv.mean(1), d_cv[:, -1], d_cv.mean(1), d_cv[:, -1]))

    results = {"windows": n, "seeds": list(args.seeds), "last_two_cv": site_equal_summary(cv_metrics, data["site"]), "durations": {}}
    ckpts: dict[str, bytes] = {}
    mot_mean_by_t: dict[int, np.ndarray] = {}
    bio_mean_by_t: dict[int, np.ndarray] = {}

    for dur in DURATIONS:
        t_grid = np.arange(dur, dtype=np.float64)
        t_c = t_grid - t_grid.mean()
        v_ols = (data["hist20"][:, -dur:].numpy() * t_c[None, :, None]).sum(axis=1, keepdims=True) / (t_c @ t_c)
        ols_pred = v_ols * steps
        d_ols = np.linalg.norm(ols_pred - truth_np, axis=2)
        ols_m = np.column_stack((d_ols.mean(1), d_ols[:, -1], d_ols.mean(1), d_ols[:, -1], d_ols.mean(1), d_ols[:, -1]))

        mot_seeds, bio_seeds, scales_rec = [], [], {}
        for seed in args.seeds:
            m_arr = np.zeros((n, 6), dtype=np.float64)
            b_arr = np.zeros((n, 6), dtype=np.float64)
            scales_rec[str(seed)] = {}
            for s in SITES:
                tr_ids = np.flatnonzero(data["site"] != s)
                te_ids = np.flatnonzero(data["site"] == s)
                mot, bio, sc = fit_motion_and_bio(seed, dur, data, tr_ids)
                m_arr[te_ids] = eval_o2(mot, data, te_ids, dur, 0.0)
                b_arr[te_ids] = eval_o2(bio, data, te_ids, dur, sc)
                scales_rec[str(seed)][s] = sc
                if seed == 42:
                    import io
                    for tag, model in (("motion", mot), ("bio_pose", bio)):
                        buf = io.BytesIO()
                        torch.save({"state_dict": model.state_dict(), "bio_scale": sc, "duration": dur, "site": s}, buf)
                        ckpts[f"T{dur}/{s}/{tag}.pt"] = buf.getvalue()
            mot_seeds.append(m_arr)
            bio_seeds.append(b_arr)
            print(f"Done T={dur} seed={seed}", flush=True)

        mot_avg = np.mean(mot_seeds, axis=0)
        bio_avg = np.mean(bio_seeds, axis=0)
        mot_mean_by_t[dur], bio_mean_by_t[dur] = mot_avg, bio_avg
        results["durations"][str(dur)] = {
            "ols_cv": site_equal_summary(ols_m, data["site"]),
            "o2_motion_5seed": site_equal_summary(mot_avg, data["site"]),
            "o2_bioeq_5seed": site_equal_summary(bio_avg, data["site"]),
            "selected_bio_scales": scales_rec,
            "paired_bio_minus_motion": paired_site_clip_ci(bio_avg[:, :2] - mot_avg[:, :2], data["site"], data["clip"]),
            "paired_bio_minus_last2_cv": paired_site_clip_ci(bio_avg[:, :2] - cv_metrics[:, :2], data["site"], data["clip"]),
            "paired_motion_minus_last2_cv": paired_site_clip_ci(mot_avg[:, :2] - cv_metrics[:, :2], data["site"], data["clip"]),
        }

    results["duration_scaling_T20_minus_T5"] = {
        "o2_motion": paired_site_clip_ci(mot_mean_by_t[20][:, :2] - mot_mean_by_t[5][:, :2], data["site"], data["clip"]),
        "o2_bioeq": paired_site_clip_ci(bio_mean_by_t[20][:, :2] - bio_mean_by_t[5][:, :2], data["site"], data["clip"]),
    }
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    members = {"report.json": (json.dumps(results, indent=2, allow_nan=False) + "\n").encode(), **ckpts}
    members["sha256.json"] = (json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in members.items()}, indent=2, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in members.items():
            zf.writestr(k, v)
    print("Saved", args.output_zip, flush=True)


if __name__ == "__main__":
    main()
