"""Five-seed four-site LOSO benchmark of BioEqGroundForecaster (Kaggle GPU/CPU runner)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.benchmark_heldout_sites import load_site_windows
from scripts.run_heldout_seed_sensitivity import SITE_COUNTS, json_digest, split_indices
from scripts.train_pinns_heldout_sites import atomic_write, save_json, tensorize
from scripts.verify_heldout_bundle import IDS_SHA, LABELS_SHA, parameter_digest
from src.models.bioeq_neural import BioEqGroundForecaster, bioeq_loss, biomech_sequence_torch, local_frenet_torch

POSE_SHA = "b493a47536664ddb4fd66603c1bbf6b2c2e33f553520382027b3ae96b53447d0"
SEEDS = (42, 1024, 2025, 3407, 4096)
ARMS = ("motion", "raw_pose", "bio_pose")
FIELDS = ("ade_m", "fde_m", "minade6_m", "minfde6_m", "mode0_selected", "bayes_ade_m", "bayes_fde_m")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_cohort(data_dir: Path, pose_path: Path) -> tuple[dict, dict, list, np.ndarray]:
    if sha_bytes(pose_path.read_bytes()) != POSE_SHA:
        raise ValueError("Pose cache SHA-256 mismatch")
    groups, _, label_sha = load_site_windows(data_dir)
    if label_sha != LABELS_SHA:
        raise ValueError("PINNS label SHA-256 mismatch")
    t, idx, keys, _, _, _ = tensorize(groups, pose_path)
    if json_digest(keys) != IDS_SHA or {k: len(v) for k, v in idx.items()} != SITE_COUNTS:
        raise ValueError("Cohort key or count mismatch")
    xy = np.stack([x.xy for s in sorted(groups) for x in groups[s]])
    with np.load(pose_path, allow_pickle=False) as pz:
        ck = list(zip(pz["clip"].tolist(), pz["person_id"].tolist(), pz["t0"].tolist()))
        order = {k: i for i, k in enumerate(ck)}
        ids = [order[k] for k in keys]
        kpts = torch.from_numpy(pz["hist_kpts"][ids].copy())
        bbox = torch.from_numpy(pz["hist_bbox"][ids].copy())
    bio_seq, q_bio = biomech_sequence_torch(kpts, bbox)
    path_5f = np.linalg.norm(np.diff(xy[:, :5], axis=1), axis=-1).sum(axis=1)
    first_seen = {}
    for i, (clip, pid, t0) in enumerate(keys):
        first_seen[(clip, pid)] = min(first_seen.get((clip, pid), t0), t0)
    subhalf = np.array([(keys[i][2] == first_seen[(keys[i][0], keys[i][1])]) and (path_5f[i] < 0.5) for i in range(len(keys))])
    tensors = {
        "history": torch.from_numpy(xy[:, :5].copy()),
        "truth": torch.from_numpy(xy[:, 5:].copy()),
        "pose": t["x_p"],
        "quality": t["q_vec"],
        "bio_seq": bio_seq,
        "q_bio": q_bio,
    }
    return tensors, idx, keys, subhalf


def fit_and_eval_arm(
    arm: str,
    seed: int,
    tensors: dict,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    torch.manual_seed(seed)
    base_init = BioEqGroundForecaster("motion").state_dict()
    torch.manual_seed(seed)
    model = BioEqGroundForecaster(arm).to(device)
    own = model.state_dict()
    for k in ("motion.weight_ih_l0", "motion.weight_hh_l0", "motion.bias_ih_l0", "motion.bias_hh_l0",
              "context.0.weight", "context.0.bias", "context.1.weight", "context.1.bias",
              "residual.weight", "residual.bias", "scores.weight", "scores.bias"):
        own[k].copy_(base_init[k])
    model.load_state_dict(own)

    tr = {k: v[train_ids].to(device) for k, v in tensors.items()}
    te = {k: v[test_ids].to(device) for k, v in tensors.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=28, eta_min=1e-4)
    gen = torch.Generator().manual_seed(seed + 7)
    model.train()
    n = len(train_ids)
    for _ in range(28):
        perm = torch.randperm(n, generator=gen).to(device)
        for batch in perm.split(64):
            opt.zero_grad(set_to_none=True)
            paths, logits, valid = model(
                tr["history"][batch], tr["pose"][batch], tr["quality"][batch],
                tr["bio_seq"][batch], tr["q_bio"][batch],
            )
            loss = bioeq_loss(paths, logits, tr["truth"][batch], valid)
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
            opt.step()
        sched.step()

    model.eval()
    preds_p, preds_q = [], []
    with torch.inference_mode():
        for batch in torch.arange(len(test_ids), device=device).split(64):
            p, lg, _ = model(
                te["history"][batch], te["pose"][batch], te["quality"][batch],
                te["bio_seq"][batch], te["q_bio"][batch],
            )
            preds_p.append(p.cpu().numpy())
            preds_q.append(lg.softmax(dim=1).cpu().numpy())
    paths_np = np.concatenate(preds_p, axis=0)
    probs_np = np.concatenate(preds_q, axis=0)
    truth_np = tensors["truth"][test_ids].numpy()
    dst = np.linalg.norm(paths_np - truth_np[:, None], axis=3)
    ade_k, fde_k = dst.mean(axis=2), dst[:, :, -1]
    top = probs_np.argmax(axis=1)
    ii = np.arange(len(paths_np))
    bayes_path = np.einsum("nk,nktw->ntw", probs_np, paths_np)
    b_dst = np.linalg.norm(bayes_path - truth_np, axis=2)
    metrics = np.column_stack((
        ade_k[ii, top], fde_k[ii, top], ade_k.min(axis=1), fde_k.min(axis=1),
        (top == 0).astype(np.float64), b_dst.mean(axis=1), b_dst[:, -1],
    ))
    return metrics, {k: v.cpu() for k, v in model.state_dict().items()}


def summarize_bioeq(
    seed_metrics: dict[str, dict[int, np.ndarray]],
    cv_metrics: np.ndarray,
    idx: dict[str, np.ndarray],
    keys: list,
    subhalf: np.ndarray,
) -> dict:
    sites = sorted(idx.keys())
    mean_by_arm = {arm: np.mean(list(seed_metrics[arm].values()), axis=0) for arm in ARMS}
    mean_by_arm["last_two_cv"] = cv_metrics

    def group_table(mask: np.ndarray) -> dict:
        by_site, valid_sites = {}, []
        for s in sites:
            rows = idx[s][mask[idx[s]]]
            if len(rows):
                valid_sites.append(s)
                by_site[s] = {
                    "windows": int(len(rows)),
                    "metrics": {a: dict(zip(FIELDS, mean_by_arm[a][rows].mean(axis=0).tolist())) for a in mean_by_arm},
                }
        macro = {
            a: dict(zip(FIELDS, np.mean([mean_by_arm[a][idx[s][mask[idx[s]]]].mean(axis=0) for s in valid_sites], axis=0).tolist()))
            for a in mean_by_arm
        }
        return {"by_site": by_site, "site_equal": macro}

    rng = np.random.default_rng(2026)
    boots = []
    for s in sites:
        rows = idx[s]
        clips, inv = np.unique([keys[i][0] for i in rows], return_inverse=True)
        w = rng.multinomial(len(clips), np.full(len(clips), 1.0 / len(clips)), size=20000)
        boots.append((s, rows, inv, w, w @ np.bincount(inv)))

    pairs = [("bio_pose", "motion"), ("bio_pose", "raw_pose"), ("bio_pose", "last_two_cv"), ("motion", "last_two_cv")]
    paired = {}
    for a, b in pairs:
        diff = mean_by_arm[a][:, :2] - mean_by_arm[b][:, :2]
        rep = np.zeros((20000, 2))
        by_s = {}
        for s, rows, inv, w, denom in boots:
            tot = np.column_stack([np.bincount(inv, weights=diff[rows, j]) for j in range(2)])
            rep += (w @ tot) / denom[:, None] / 4.0
            by_s[s] = dict(zip(("ade_m", "fde_m"), diff[rows].mean(axis=0).tolist()))
        paired[f"{a}_minus_{b}"] = {
            "mean": dict(zip(("ade_m", "fde_m"), np.mean([ list(by_s[s].values()) for s in sites ], axis=0).tolist())),
            "paired_clip_95": {k: np.percentile(rep[:, j], [2.5, 97.5]).tolist() for j, k in enumerate(("ade_m", "fde_m"))},
            "by_site": by_s,
        }
    return {"all_1482": group_table(np.ones(len(keys), dtype=bool)), "subhalf_323": group_table(subhalf), "paired_5seed": paired}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--pose", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors, idx, keys, subhalf = prepare_cohort(args.data, args.pose)
    hist_np, truth_np = tensors["history"].numpy(), tensors["truth"].numpy()
    times = np.arange(1, 31, dtype=np.float64)[None, :, None]
    cv_path = hist_np[:, -1:] + (hist_np[:, -1:] - hist_np[:, -2:-1]) * times
    cv_dst = np.linalg.norm(cv_path - truth_np, axis=2)
    cv_metrics = np.column_stack((cv_dst.mean(1), cv_dst[:, -1], cv_dst.mean(1), cv_dst[:, -1], np.ones(len(keys)), cv_dst.mean(1), cv_dst[:, -1]))

    seed_metrics = {a: {} for a in ARMS}
    states = {}
    for seed in SEEDS:
        for a in ARMS:
            seed_metrics[a][seed] = np.zeros((len(keys), len(FIELDS)), dtype=np.float64)
        for site, te_list in idx.items():
            _, tr_list, te_chk = split_indices(idx, keys, site)
            tr_ids, te_ids = np.array(tr_list), np.array(te_chk)
            for a in ARMS:
                m, st = fit_and_eval_arm(a, seed, tensors, tr_ids, te_ids, device)
                seed_metrics[a][seed][te_ids] = m
                if seed == 42:
                    states[f"{site}/{a}.pt"] = st

    report = summarize_bioeq(seed_metrics, cv_metrics, idx, keys, subhalf)
    report.update({"seeds": list(SEEDS), "device": str(device), "windows": len(keys), "subhalf_windows": int(subhalf.sum())})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("bioeq_5seed_report.json", json.dumps(report, indent=2) + "\n")
        for name, st in states.items():
            buf = Path(args.output.parent) / "_tmp.pt"
            torch.save(st, buf)
            z.write(buf, arcname=name)
            buf.unlink()
    print(json.dumps(report["all_1482"]["site_equal"], indent=2))
    print(json.dumps(report["paired_5seed"], indent=2))


if __name__ == "__main__":
    main()
