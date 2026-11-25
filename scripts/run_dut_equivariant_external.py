"""Zero-shot external evaluation of PINNS-trained SO(2) BioEq forecaster on 3,066 DUT windows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.benchmark_heldout_sites import load_site_windows
from scripts.run_heldout_seed_sensitivity import split_indices
from scripts.train_pinns_heldout_sites import tensorize
from src.models.bioeq_neural import BioEqGroundForecaster, bioeq_loss, local_frenet_torch

ANGLES = (0, 90, 180, 270)


def fit_pinns_folds(pinns_dir: Path, pose_path: Path, seed: int = 42) -> dict[str, BioEqGroundForecaster]:
    groups, _, _ = load_site_windows(pinns_dir)
    t, idx, keys, _, _, _ = tensorize(groups, pose_path)
    xy = np.stack([x.xy for s in sorted(groups) for x in groups[s]])
    hist_all = torch.from_numpy(xy[:, :5].copy())
    truth_all = torch.from_numpy(xy[:, 5:].copy())
    models = {}
    for site in sorted(idx.keys()):
        _, tr_list, _ = split_indices(idx, keys, site)
        tr_ids = torch.tensor(tr_list, dtype=torch.long)
        torch.manual_seed(seed)
        net = BioEqGroundForecaster("motion")
        opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=28, eta_min=1e-4)
        gen = torch.Generator().manual_seed(seed + 7)
        net.train()
        for _ in range(28):
            perm = torch.randperm(len(tr_ids), generator=gen)
            for b in perm.split(64):
                opt.zero_grad(set_to_none=True)
                ids = tr_ids[b]
                p, lg, ok = net(hist_all[ids])
                loss = bioeq_loss(p, lg, truth_all[ids], ok)
                if loss is not None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                    opt.step()
            sched.step()
        models[site] = net.eval()
    return models


@torch.inference_mode()
def forward_dut(
    net: BioEqGroundForecaster,
    hist: torch.Tensor,
    ticks: torch.Tensor,
    clock_aware: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not clock_aware:
        p, lg, _ = net(hist)
        return p.numpy(), lg.softmax(dim=1).numpy()

    origin, basis, valid = local_frenet_torch(hist)
    body = (hist - origin) @ basis
    dt_hist = ((ticks[:, 1:5] - ticks[:, :4]).double() / 23976.0).clamp_min(1e-6)
    vel_steps = (body[:, 1:] - body[:, :-1]) / dt_hist[:, :, None]
    step = torch.cat((torch.zeros_like(body[:, :1]), vel_steps), dim=1)
    feat = torch.cat((body, step, torch.linalg.vector_norm(step, dim=2, keepdim=True)), dim=2)
    _, hm = net.motion(feat.float())
    ctx = net.context(hm[0])
    logits = torch.where(valid[:, None], net.scores(ctx), 0.0)
    knots = net.residual(ctx).view(len(hist), 5, 3, 2).double()
    knots = torch.cat((torch.zeros_like(knots[:, :, :1]), knots), dim=2)

    tau = ((ticks[:, 5:] - ticks[:, 4:5]).double() / 23976.0).clamp(0.0, 3.0)
    scaled = (tau / 1.0).clamp(0.0, 2.999999)
    left = scaled.long()
    phase = (scaled - left.double())[:, None, :, None]
    rows = torch.arange(len(hist))[:, None, None]
    modes_idx = torch.arange(5)[None, :, None]
    k_l = knots[rows, modes_idx, left[:, None, :]]
    k_r = knots[rows, modes_idx, left[:, None, :] + 1]
    delta = k_l * (1.0 - phase) + k_r * phase
    world = torch.einsum("bkts,bws->bktw", delta, basis)
    v_last = (hist[:, -1:] - hist[:, -2:-1]) / dt_hist[:, -1:, None]
    cv = origin + v_last * tau[:, :, None]
    paths = torch.cat((cv[:, None], cv[:, None] + world), dim=1)
    return paths.numpy(), logits.softmax(dim=1).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinns-data", type=Path, required=True)
    ap.add_argument("--pose", type=Path, required=True)
    ap.add_argument("--dut-code", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import sys as _sys
    _sys.path.insert(0, str(args.dut_code))
    import motion_cloud as mc

    keys, _, ticks_np, xy_np = mc.build(args.dut_code / "dut_raw")
    models = fit_pinns_folds(args.pinns_data, args.pose, seed=42)
    hist_base = xy_np[:, :5] - xy_np[:, 4:5]
    truth = xy_np[:, 5:] - xy_np[:, 4:5]
    diff = hist_base[:, -1] - hist_base[:, -2]
    elapsed = ticks_np[:, 5:] - ticks_np[:, 4, None]
    cv_meas = diff[:, None] * elapsed[:, :, None] / (ticks_np[:, 4] - ticks_np[:, 3])[:, None, None]
    cv_err = np.column_stack((np.linalg.norm(cv_meas - truth, axis=-1).mean(1), np.linalg.norm(cv_meas - truth, axis=-1)[:, -1]))

    ticks_t = torch.from_numpy(ticks_np)
    report = {"windows": len(keys), "angles": {}}
    for deg, (c, s) in zip(ANGLES, ((1, 0), (0, 1), (-1, 0), (0, -1))):
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        h_rot = torch.from_numpy(hist_base @ rot.T)
        res_deg = {"cv_measured": mc.grouped(cv_err, np.ones(len(keys), bool), keys)}
        for tag, ca in (("so2_nominal_10hz", False), ("so2_clock_aware", True)):
            fold_top, fold_bayes = [], []
            for site in sorted(models):
                p_rot, pr = forward_dut(models[site], h_rot, ticks_t, clock_aware=ca)
                p_w = p_rot @ rot
                top_p = p_w[np.arange(len(keys)), pr.argmax(1)]
                bay_p = np.einsum("nk,nktw->ntw", pr, p_w)
                fold_top.append(np.column_stack((np.linalg.norm(top_p - truth, axis=-1).mean(1), np.linalg.norm(top_p - truth, axis=-1)[:, -1])))
                fold_bayes.append(np.column_stack((np.linalg.norm(bay_p - truth, axis=-1).mean(1), np.linalg.norm(bay_p - truth, axis=-1)[:, -1])))
            res_deg[f"{tag}_top1"] = mc.grouped(np.mean(fold_top, axis=0), np.ones(len(keys), bool), keys)
            res_deg[f"{tag}_bayes"] = mc.grouped(np.mean(fold_bayes, axis=0), np.ones(len(keys), bool), keys)
        report["angles"][str(deg)] = res_deg

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("dut_equivariant_external.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["angles"]["0"], indent=2))


if __name__ == "__main__":
    main()
