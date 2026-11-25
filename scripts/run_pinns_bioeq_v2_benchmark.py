"""Five-seed 4-site LOSO benchmark of M1_BioEq_v2 (site-balanced + decoupled two-stage)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.run_heldout_seed_sensitivity import split_indices
from scripts.run_pinns_bioeq_benchmark import FIELDS, SEEDS, prepare_cohort, summarize_bioeq
from src.models.bioeq_neural import BioEqGroundForecaster


def weighted_bioeq_loss(
    paths: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    sample_w: torch.Tensor,
    reg_only: bool = False,
) -> torch.Tensor | None:
    if not torch.any(valid):
        return None
    w_v = sample_w[valid]
    w_v = w_v / w_v.mean().clamp_min(1e-8)
    dist = torch.linalg.vector_norm(paths[valid] - target[valid, None], dim=-1)
    ade = dist.detach().mean(dim=2)
    winner = ade[:, 1:] == ade[:, 1:].amin(dim=1, keepdim=True)
    m_w = winner.double() / winner.sum(dim=1, keepdim=True)
    reg_i = ((torch.sqrt(dist[:, 1:].square() + 0.25) - 0.5).mean(dim=2) * m_w).sum(dim=1)
    if reg_only:
        return (reg_i * w_v.double()).mean()
    soft = (-ade / 0.15).softmax(dim=1).to(logits.dtype)
    cls_i = -(soft * logits[valid].log_softmax(dim=1)).sum(dim=1)
    return ((reg_i + 0.25 * cls_i.double()) * w_v.double()).mean()


def make_site_weights(site_arr: np.ndarray, ids: np.ndarray) -> torch.Tensor:
    sub = site_arr[ids]
    uniq, counts = np.unique(sub, return_counts=True)
    inv = {s: 1.0 / float(c) for s, c in zip(uniq.tolist(), counts.tolist())}
    w = np.array([inv[s] for s in sub.tolist()], dtype=np.float32)
    w /= w.mean()
    return torch.from_numpy(w)


@torch.inference_mode()
def eval_metrics(
    net: BioEqGroundForecaster,
    tensors: dict,
    ids: np.ndarray,
    device: torch.device,
    bio_scale: float = 1.0,
) -> np.ndarray:
    te = {k: v[ids].to(device) for k, v in tensors.items()}
    preds_p, preds_q = [], []
    q_in = te["q_bio"] * float(bio_scale)
    for b in torch.arange(len(ids), device=device).split(64):
        p, lg, _ = net(te["history"][b], te["pose"][b], te["quality"][b], te["bio_seq"][b], q_in[b])
        preds_p.append(p.cpu().numpy())
        preds_q.append(lg.softmax(dim=1).cpu().numpy())
    paths_np, probs_np = np.concatenate(preds_p), np.concatenate(preds_q)
    truth_np = tensors["truth"][ids].numpy()
    dst = np.linalg.norm(paths_np - truth_np[:, None], axis=3)
    ade_k, fde_k = dst.mean(axis=2), dst[:, :, -1]
    top = probs_np.argmax(axis=1)
    ii = np.arange(len(paths_np))
    bayes = np.einsum("nk,nktw->ntw", probs_np, paths_np)
    bd = np.linalg.norm(bayes - truth_np, axis=2)
    return np.column_stack((
        ade_k[ii, top], fde_k[ii, top], ade_k.min(1), fde_k.min(1),
        (top == 0).astype(np.float64), bd.mean(1), bd[:, -1],
    ))


def fit_backbone(seed: int, tensors: dict, ids: np.ndarray, site_arr: np.ndarray, device: torch.device, arm: str = "motion") -> BioEqGroundForecaster:
    torch.manual_seed(seed)
    base_init = BioEqGroundForecaster("motion").state_dict()
    torch.manual_seed(seed)
    net = BioEqGroundForecaster(arm).to(device)
    own = net.state_dict()
    for k in ("motion.weight_ih_l0", "motion.weight_hh_l0", "motion.bias_ih_l0", "motion.bias_hh_l0",
              "context.0.weight", "context.0.bias", "context.1.weight", "context.1.bias",
              "residual.weight", "residual.bias", "scores.weight", "scores.bias"):
        own[k].copy_(base_init[k])
    net.load_state_dict(own)
    tr = {k: v[ids].to(device) for k, v in tensors.items()}
    sw = make_site_weights(site_arr, ids).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=28, eta_min=1e-4)
    gen = torch.Generator().manual_seed(seed + 7)
    net.train()
    for _ in range(28):
        perm = torch.randperm(len(ids), generator=gen).to(device)
        for b in perm.split(64):
            opt.zero_grad(set_to_none=True)
            p, lg, ok = net(tr["history"][b], tr["pose"][b], tr["quality"][b], tr["bio_seq"][b], tr["q_bio"][b])
            loss = weighted_bioeq_loss(p, lg, tr["truth"][b], ok, sw[b])
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
        sched.step()
    return net.eval()


def fit_stage2_bio(mot_net: BioEqGroundForecaster, seed: int, tensors: dict, ids: np.ndarray, site_arr: np.ndarray, device: torch.device) -> BioEqGroundForecaster:
    torch.manual_seed(seed + 101)
    bio_net = BioEqGroundForecaster("bio_pose").to(device)
    st = bio_net.state_dict()
    mot_st = mot_net.state_dict()
    for k in ("motion.weight_ih_l0", "motion.weight_hh_l0", "motion.bias_ih_l0", "motion.bias_hh_l0",
              "context.0.weight", "context.0.bias", "context.1.weight", "context.1.bias",
              "residual.weight", "residual.bias", "scores.weight", "scores.bias"):
        st[k].copy_(mot_st[k])
    st["bio_score.weight"].zero_()
    st["bio_score.bias"].zero_()
    bio_net.load_state_dict(st)
    for name, p in bio_net.named_parameters():
        p.requires_grad_(name in ("bio_in.weight", "bio_in.bias", "bio_gru.weight_ih_l0", "bio_gru.weight_hh_l0",
                                  "bio_gru.bias_ih_l0", "bio_gru.bias_hh_l0", "bio_par.weight", "bio_par.bias", "gate_logit"))
    tr = {k: v[ids].to(device) for k, v in tensors.items()}
    sw = make_site_weights(site_arr, ids).to(device)
    trainable = [p for p in bio_net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=0.0015, weight_decay=3e-4)
    gen = torch.Generator().manual_seed(seed + 33)
    bio_net.train()
    for _ in range(14):
        perm = torch.randperm(len(ids), generator=gen).to(device)
        for b in perm.split(64):
            opt.zero_grad(set_to_none=True)
            p, lg, ok = bio_net(tr["history"][b], tr["pose"][b], tr["quality"][b], tr["bio_seq"][b], tr["q_bio"][b])
            loss = weighted_bioeq_loss(p, lg, tr["truth"][b], ok, sw[b], reg_only=True)
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.5)
                opt.step()
    return bio_net.eval()


def select_inner_bio_scale(seed: int, tensors: dict, tr_ids: np.ndarray, site_arr: np.ndarray, device: torch.device) -> float:
    src_sites = sorted(set(site_arr[tr_ids].tolist()))
    scales = (0.0, 0.25, 0.5, 1.0)
    site_ades = {sc: [] for sc in scales}
    for vs in src_sites:
        in_tr = tr_ids[site_arr[tr_ids] != vs]
        in_va = tr_ids[site_arr[tr_ids] == vs]
        m_net = fit_backbone(seed, tensors, in_tr, site_arr, device, arm="motion")
        b_net = fit_stage2_bio(m_net, seed, tensors, in_tr, site_arr, device)
        for sc in scales:
            site_ades[sc].append(float(eval_metrics(b_net, tensors, in_va, device, bio_scale=sc)[:, 0].mean()))
    best_sc, best_val = 0.0, float(np.mean(site_ades[0.0]))
    for sc in scales[1:]:
        val = float(np.mean(site_ades[sc]))
        if val < best_val - 1e-6:
            best_val, best_sc = val, sc
    return best_sc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--pose", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors, idx, keys, subhalf = prepare_cohort(args.data, args.pose)
    site_arr = np.empty(len(keys), dtype="<U24")
    for s, rows in idx.items():
        site_arr[rows] = s

    hist_np, truth_np = tensors["history"].numpy(), tensors["truth"].numpy()
    times = np.arange(1, 31, dtype=np.float64)[None, :, None]
    cv_path = hist_np[:, -1:] + (hist_np[:, -1:] - hist_np[:, -2:-1]) * times
    cv_dst = np.linalg.norm(cv_path - truth_np, axis=2)
    cv_metrics = np.column_stack((cv_dst.mean(1), cv_dst[:, -1], cv_dst.mean(1), cv_dst[:, -1], np.ones(len(keys)), cv_dst.mean(1), cv_dst[:, -1]))

    seed_metrics = {a: {} for a in ("motion", "raw_pose", "bio_pose")}
    selected_scales = {}
    states = {}
    for seed in SEEDS:
        for a in seed_metrics:
            seed_metrics[a][seed] = np.zeros((len(keys), len(FIELDS)), dtype=np.float64)
        for site, _ in idx.items():
            _, tr_list, te_chk = split_indices(idx, keys, site)
            tr_ids, te_ids = np.array(tr_list), np.array(te_chk)
            m_net = fit_backbone(seed, tensors, tr_ids, site_arr, device, arm="motion")
            r_net = fit_backbone(seed, tensors, tr_ids, site_arr, device, arm="raw_pose")
            b_net = fit_stage2_bio(m_net, seed, tensors, tr_ids, site_arr, device)
            sc = select_inner_bio_scale(seed, tensors, tr_ids, site_arr, device) if seed == 42 else selected_scales[(42, site)]
            selected_scales[(seed, site)] = sc
            seed_metrics["motion"][seed][te_ids] = eval_metrics(m_net, tensors, te_ids, device)
            seed_metrics["raw_pose"][seed][te_ids] = eval_metrics(r_net, tensors, te_ids, device)
            seed_metrics["bio_pose"][seed][te_ids] = eval_metrics(b_net, tensors, te_ids, device, bio_scale=sc)
            if seed == 42:
                states[f"{site}/motion.pt"] = {k: v.cpu() for k, v in m_net.state_dict().items()}
                states[f"{site}/bio_pose.pt"] = {k: v.cpu() for k, v in b_net.state_dict().items()}

    report = summarize_bioeq(seed_metrics, cv_metrics, idx, keys, subhalf)
    report["selected_bio_scales_seed42"] = {s: selected_scales[(42, s)] for s in sorted(idx)}
    report["windows"] = len(keys)
    report["subhalf_windows"] = int(subhalf.sum())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("bioeq_v2_5seed_report.json", json.dumps(report, indent=2) + "\n")
        for name, st in states.items():
            tmp_pt = args.output.parent / "_tmp_v2.pt"
            torch.save(st, tmp_pt)
            z.write(tmp_pt, arcname=name)
            tmp_pt.unlink()
    print(json.dumps(report["all_1482"]["site_equal"], indent=2))
    print(json.dumps(report["paired_5seed"], indent=2))


if __name__ == "__main__":
    main()
