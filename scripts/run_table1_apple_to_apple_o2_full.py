"""Apple-to-apple 5-seed Table 1 & 2 benchmark (5,686/1,100/497 PINNS split, T=5,10,20) for Ours_O2_Full."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from scripts.cloud_experiment_runner import FIXED_SEEDS, stage_pinns_data
from src.data.constants import DT, T_FUT
from src.data.dataset_builder import materialize_window_split
from src.data.pinns_bev import SOURCE_LABEL
from src.metrics.metrics_and_stats import (
    compute_action_metrics,
    compute_sample_trajectory_metrics,
    holm_bonferroni_adjust,
    paired_hypothesis_test,
)
from src.models.bioeq_neural import biomech_sequence_torch
from src.models.o2_unified_full import O2UnifiedFullForecaster


def build_pinns_split_tensors(samples: list[dict], split: str, t_obs: int) -> dict:
    sub = [s for s in samples if s["split"] == split]
    mat = materialize_window_split(samples, split, t_obs=t_obs, dt=DT)
    n = len(sub)
    hist = np.zeros((n, t_obs, 2), dtype=np.float64)
    obs_dt = np.full((n, t_obs - 1), DT, dtype=np.float64)
    kpts5 = np.zeros((n, 5, 17, 3), dtype=np.float32)
    bbox5 = np.zeros((n, 5, 4), dtype=np.float32)
    sites = []
    for i, s in enumerate(sub):
        sites.append("_".join(str(s["seq_id"]).split("_")[:2]))
        hp = s["hist_pos_20"][-t_obs:].astype(np.float64)
        hm = s["hist_mask_20"][-t_obs:] > 0.5
        cur = int(np.flatnonzero(hm)[0])
        idx_m = np.zeros(t_obs, dtype=int)
        for j in range(t_obs):
            if hm[j]:
                cur = j
            idx_m[j] = cur
            hist[i, j] = hp[cur] - hp[-1]
        obs_dt[i] = np.maximum(np.diff(idx_m), 1) * DT
        hm5 = s["hist_mask_20"][-5:] > 0.5
        c5 = int(np.flatnonzero(hm5)[0])
        im5 = []
        for j in range(5):
            if hm5[j]:
                c5 = j
            im5.append(c5)
        kpts5[i] = s["hist_kpts_20"][-5:][im5]
        bbox5[i] = s["hist_bbox_20"][-5:][im5]
    bio_seq, q_bio = biomech_sequence_torch(torch.from_numpy(kpts5), torch.from_numpy(bbox5))
    return {
        "site": np.array(sites, dtype="<U24"),
        "history": torch.from_numpy(hist),
        "obs_dt": torch.from_numpy(obs_dt),
        "bio_seq": bio_seq,
        "q_bio": q_bio,
        "edges": mat["nbr_edges"].float(),
        "is_car": (mat["nbr_type_oh"][:, :, 6] > 0.5).float(),
        "nbr_mask": mat["nbr_mask"].float(),
        "fut_xy": mat["fut_xy"].double(),
        "fut_mask": mat["fut_mask"].double(),
        "y_act": mat["y_act"].float(),
        "m_act": mat["m_act"].float(),
        "bucket": mat["bucket"].numpy(),
    }


def masked_unified_loss(
    out: dict[str, torch.Tensor],
    fut_xy: torch.Tensor,
    fut_mask: torch.Tensor,
    y_act: torch.Tensor | None = None,
    m_act: torch.Tensor | None = None,
    pos_w: torch.Tensor | None = None,
    act_w: float = 0.25,
    reg_only: bool = False,
) -> torch.Tensor | None:
    valid = out["valid"] & (fut_mask.sum(1) > 0.5)
    if not torch.any(valid):
        return None
    fm = fut_mask[valid, None]
    cnt = fut_mask[valid].sum(1, keepdim=True).clamp_min(1.0)
    dist = torch.linalg.vector_norm(out["paths"][valid] - fut_xy[valid, None], dim=-1) * fm
    ade = dist.detach().sum(2) / cnt
    winner = ade[:, 1:] == ade[:, 1:].amin(1, keepdim=True)
    mw = winner.double() / winner.sum(1, keepdim=True)
    huber = ((torch.sqrt(dist[:, 1:].square() + 0.25) - 0.5) * fm).sum(2) / cnt
    reg_i = (huber * mw).sum(1)
    if reg_only:
        return reg_i.mean()
    soft = (-ade / 0.15).softmax(1).to(out["logits"].dtype)
    cls_i = -(soft * out["logits"][valid].log_softmax(1)).sum(1).double()
    total = (reg_i + 0.25 * cls_i).mean()
    if y_act is not None and m_act is not None and act_w > 0.0:
        va = valid & (m_act > 0.5)
        if torch.any(va):
            bce = F.binary_cross_entropy_with_logits(out["act_logits"][va], y_act[va], pos_weight=pos_w, reduction="mean").double()
            total = total + act_w * bce
    return total


def fit_table1_seed(seed: int, tr: dict, va: dict, dev: torch.device) -> tuple[O2UnifiedFullForecaster, float, float]:
    torch.manual_seed(seed)
    net = O2UnifiedFullForecaster().to(dev)
    h, dt, b, q = tr["history"].to(dev), tr["obs_dt"].to(dev), tr["bio_seq"].to(dev), tr["q_bio"].to(dev)
    e, c, m = tr["edges"].to(dev), tr["is_car"].to(dev), tr["nbr_mask"].to(dev)
    fy, fm, ya, ma = tr["fut_xy"].to(dev), tr["fut_mask"].to(dev), tr["y_act"].to(dev), tr["m_act"].to(dev)
    va_m = ma > 0.5
    pos = ya[va_m].sum(0).clamp_min(1.0)
    neg = (1.0 - ya[va_m]).sum(0).clamp_min(1.0)
    pw = (neg / pos).clamp(1.0, 12.0).to(dev)
    gen = torch.Generator().manual_seed(seed)

    s1 = ("motion.", "ctx_proj.", "act_even.", "act_turn.", "bridge_even.", "bridge_odd.", "central_", "pair_")
    for n_p, p in net.named_parameters():
        p.requires_grad = n_p.startswith(s1)
    opt1 = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=2.5e-3, weight_decay=1e-4)
    for _ in range(36):
        for ch_c in torch.randperm(len(h), generator=gen).split(256):
            ch = ch_c.to(dev)
            loss = masked_unified_loss(net(h[ch], bio_scale=0.0, graph_scale=0.0, obs_dt=dt[ch]), fy[ch], fm[ch], ya[ch], ma[ch], pw, 0.25)
            if loss is not None:
                opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()

    for n_p, p in net.named_parameters():
        p.requires_grad = n_p.startswith("bio_") or n_p == "gate_logit"
    opt2 = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1.5e-3, weight_decay=2e-4)
    for _ in range(22):
        for ch_c in torch.randperm(len(h), generator=gen).split(256):
            ch = ch_c.to(dev)
            loss = masked_unified_loss(net(h[ch], b[ch], q[ch], bio_scale=1.0, graph_scale=0.0, obs_dt=dt[ch]), fy[ch], fm[ch], ya[ch], ma[ch], pw, 0.15)
            if loss is not None:
                opt2.zero_grad(set_to_none=True); loss.backward(); opt2.step()

    net.eval()
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    hv, dtv, bv, qv = va["history"].to(dev), va["obs_dt"].to(dev), va["bio_seq"].to(dev), va["q_bio"].to(dev)
    ev, cv, mv = va["edges"].to(dev), va["is_car"].to(dev), va["nbr_mask"].to(dev)
    with torch.inference_mode():
        bsc_s = [float(compute_sample_trajectory_metrics(net(hv, bv, qv, bio_scale=g, graph_scale=0.0, obs_dt=dtv)["paths"].cpu().numpy(),
                 net(hv, bv, qv, bio_scale=g, graph_scale=0.0, obs_dt=dtv)["logits"].softmax(1).cpu().numpy(),
                 va["fut_xy"].numpy(), va["fut_mask"].numpy())["ADE_1"].mean()) for g in grid]
    best_b = float(grid[int(np.argmin(bsc_s))])

    for n_p, p in net.named_parameters():
        p.requires_grad = n_p.startswith(("edge_", "graph_proj")) or n_p == "null_logit"
    opt3 = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1.5e-3, weight_decay=3e-4)
    net.train()
    for _ in range(24):
        for ch_c in torch.randperm(len(h), generator=gen).split(256):
            ch = ch_c.to(dev)
            loss = masked_unified_loss(net(h[ch], b[ch], q[ch], e[ch], c[ch], m[ch], bio_scale=best_b, graph_scale=1.0, obs_dt=dt[ch]), fy[ch], fm[ch], act_w=0.0)
            if loss is not None:
                opt3.zero_grad(set_to_none=True); loss.backward(); opt3.step()

    net.eval()
    with torch.inference_mode():
        gsc_s = []
        for g in grid:
            ov = net(hv, bv, qv, ev, cv, mv, bio_scale=best_b, graph_scale=g, obs_dt=dtv)
            gsc_s.append(float(compute_sample_trajectory_metrics(ov["paths"].cpu().numpy(), ov["logits"].softmax(1).cpu().numpy(), va["fut_xy"].numpy(), va["fut_mask"].numpy())["ADE_1"].mean()))
    return net, best_b, float(grid[int(np.argmin(gsc_s))])


@torch.inference_mode()
def eval_table1_seed(net: O2UnifiedFullForecaster, te: dict, bsc: float, gsc: float, dev: torch.device) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    out = net(te["history"].to(dev), te["bio_seq"].to(dev), te["q_bio"].to(dev),
              te["edges"].to(dev), te["is_car"].to(dev), te["nbr_mask"].to(dev),
              bio_scale=bsc, graph_scale=gsc, obs_dt=te["obs_dt"].to(dev))
    paths = out["paths"].cpu().numpy()
    probs = out["logits"].softmax(1).cpu().numpy()
    m = compute_sample_trajectory_metrics(paths, probs, te["fut_xy"].numpy(), te["fut_mask"].numpy())
    bay = np.einsum("nk,nktw->ntw", probs, paths)
    fm = (te["fut_mask"].numpy() > 0.5).astype(np.float64)
    gt = np.nan_to_num(te["fut_xy"].numpy(), nan=0.0)
    db = np.linalg.norm(bay - gt, axis=-1) * fm
    m["Bayes_ADE"] = db.sum(1) / np.maximum(fm.sum(1), 1.0)
    li = np.argmax(np.arange(T_FUT)[None, :] * (fm > 0.5), axis=1)
    m["Bayes_FDE"] = np.linalg.norm(bay[np.arange(len(gt)), li] - gt[np.arange(len(gt)), li], axis=-1)
    pa = torch.sigmoid(out["act_logits"]).cpu().numpy()
    act_m = compute_action_metrics(te["y_act"].numpy(), pa, te["m_act"].numpy())
    return m, act_m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paired_errors_npz", type=Path)
    ap.add_argument("output_zip", type=Path)
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/w11_table1_pinns"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(FIXED_SEEDS))
    args = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    samples, audit = stage_pinns_data(args.scratch, bev_convention=SOURCE_LABEL)
    with np.load(args.paired_errors_npz, allow_pickle=False) as p_ar:
        paired_ref = {k: p_ar[k] for k in p_ar.files}

    durations_out = {}
    for t_obs in (5, 10, 20):
        tr = build_pinns_split_tensors(samples, "train", t_obs)
        va = build_pinns_split_tensors(samples, "val", t_obs)
        te = build_pinns_split_tensors(samples, "test", t_obs)
        seed_m, seed_a = [], []
        for seed in args.seeds:
            net, bsc, gsc = fit_table1_seed(seed, tr, va, dev)
            sm, sa = eval_table1_seed(net, te, bsc, gsc, dev)
            seed_m.append(sm); seed_a.append(sa)
            print(f"Done Table 1 Ours_O2_Full T={t_obs} seed={seed}", flush=True)

        keys_m = ("ADE_1", "FDE_1", "minADE_6", "minFDE_6", "NLL", "Brier", "Bayes_ADE", "Bayes_FDE")
        avg_m = {k: np.mean([d[k] for d in seed_m], axis=0) for k in keys_m}
        bk = te["bucket"]
        trans = (te["m_act"].numpy() > 0.5) & (te["y_act"].numpy()[:, :4].sum(-1) > 0.5)
        slices = {}
        for s_name, msk in (("Overall", np.ones(len(bk), dtype=bool)), ("Clear", bk == 0), ("Partial", bk == 1), ("Heavy", bk == 2), ("Transition", trans)):
            slices[s_name] = {"count": int(msk.sum()), **{k: float(avg_m[k][msk].mean()) for k in keys_m}}

        tests_top1, tests_bayes = [], []
        refs = ["M0", "M1", "M1p", "M2", "M3", "Ours_Full", "Ext1_MCENET", "Ext2_HST", "Ext3_SocialTransmotion"]
        for r_name in refs:
            rk = f"PINNS_T{t_obs}_{r_name}__ade1"
            if rk in paired_ref:
                t1 = paired_hypothesis_test(avg_m["ADE_1"], paired_ref[rk]); t1["comparison"] = f"T={t_obs} Overall: Ours_O2_Full(Top1) vs {r_name}"
                tb = paired_hypothesis_test(avg_m["Bayes_ADE"], paired_ref[rk]); tb["comparison"] = f"T={t_obs} Overall: Ours_O2_Full(Bayes) vs {r_name}"
                tests_top1.append(t1); tests_bayes.append(tb)
        if t_obs == 5:
            for b_lbl, b_id in (("Clear", 0), ("Partial", 1), ("Heavy", 2)):
                bm = bk == b_id
                for r_name in ("M0", "M1", "Ours_Full"):
                    rk = f"PINNS_T5_{r_name}__ade1"
                    if rk in paired_ref:
                        tb = paired_hypothesis_test(avg_m["Bayes_ADE"][bm], paired_ref[rk][bm])
                        tb["comparison"] = f"T=5 {b_lbl}: Ours_O2_Full(Bayes) vs {r_name}"
                        tests_bayes.append(tb)
        holm_bonferroni_adjust(tests_top1)
        holm_bonferroni_adjust(tests_bayes)

        durations_out[f"T{t_obs}"] = {
            "slices": slices,
            "action_macro_f1_mean": float(np.mean([a["macro_f1"] for a in seed_a])),
            "action_mAP_mean": float(np.mean([a["mAP"] for a in seed_a])),
            "paired_tests_top1": tests_top1,
            "paired_tests_bayes": tests_bayes,
        }

    report = {"pinns_audit": audit, "seeds": list(args.seeds), "durations": durations_out}
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    members = {"report.json": (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()}
    members["sha256.json"] = (json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in members.items()}, indent=2, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in members.items():
            zf.writestr(k, v)
    print("Saved", args.output_zip, flush=True)


if __name__ == "__main__":
    main()
