"""Five-seed JTA 2.5Hz (T=2 0.4s & T=8 2.8s) O(2) BioEq & Null-Sink Graph + Zero-Shot PINNS->JTA Transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch

from scripts.cloud_experiment_runner import ST_RELEASE_URL, _download_verified_zip, _unpack_verified_zip
from scripts.run_o2_hetero_graph_pinns_and_dut import O2NullSinkGraphForecaster
from src.data.dataset_builder import materialize_window_split
from src.data.jta_builder import process_jta_root
from src.metrics.metrics_and_stats import compute_sample_trajectory_metrics, paired_hypothesis_test
from src.models.bioeq_neural import biomech_sequence_torch
from src.models.bioeq_o2_joint import O2BioEqJointForecaster

SEEDS = (42, 1024, 2025, 3407, 4096)


def build_jta_tensors(samples: list[dict], split: str, t_obs: int) -> dict:
    sub = [s for s in samples if s["split"] == split]
    mat = materialize_window_split(samples, split, t_obs=t_obs, dt=0.4)
    n = len(sub)
    hist = np.zeros((n, t_obs, 2), dtype=np.float64)
    obs_dt = np.full((n, t_obs - 1), 0.4, dtype=np.float64)
    kpts = np.zeros((n, t_obs, 17, 3), dtype=np.float32)
    bbox = np.zeros((n, t_obs, 4), dtype=np.float32)
    for i, s in enumerate(sub):
        hp = s["hist_pos_20"][-t_obs:].astype(np.float64)
        hm = s["hist_mask_20"][-t_obs:] > 0.5
        # Forward-fill any missing early frame so causal step dt accounts for exact gap
        valid_idx = np.flatnonzero(hm)
        cur = valid_idx[0]
        idx_map = np.zeros(t_obs, dtype=int)
        for j in range(t_obs):
            if hm[j]:
                cur = j
            idx_map[j] = cur
            hist[i, j] = hp[cur] - hp[-1]
        gaps = np.maximum(np.diff(idx_map), 1)
        obs_dt[i] = gaps * 0.4
        kpts[i] = s["hist_kpts_20"][-t_obs:][idx_map]
        bbox[i] = s["hist_bbox_20"][-t_obs:][idx_map]
    bio_seq, q_bio = biomech_sequence_torch(torch.from_numpy(kpts), torch.from_numpy(bbox))
    fut_s = np.broadcast_to(np.arange(1, 13, dtype=np.float64) * 0.4, (n, 12)).copy()
    is_car = (mat["nbr_type_oh"][:, :, 6] > 0.5).float()
    return {
        "history": torch.from_numpy(hist),
        "obs_dt": torch.from_numpy(obs_dt),
        "fut_s": torch.from_numpy(fut_s),
        "bio_seq": bio_seq,
        "q_bio": q_bio,
        "edges": mat["nbr_edges"].float(),
        "is_car": is_car,
        "nbr_mask": mat["nbr_mask"].float(),
        "fut_xy": mat["fut_xy"].double(),
        "fut_mask": mat["fut_mask"].double(),
        "bucket": mat["bucket"].numpy(),
    }


def masked_jta_loss(out: dict[str, torch.Tensor], fut_xy: torch.Tensor, fut_mask: torch.Tensor, reg_only: bool = False) -> torch.Tensor | None:
    valid = out["valid"] & (fut_mask.sum(dim=1) > 0.5)
    if not torch.any(valid):
        return None
    fm = fut_mask[valid, None]
    cnt = fut_mask[valid].sum(dim=1, keepdim=True).clamp_min(1.0)
    dist = torch.linalg.vector_norm(out["paths"][valid] - fut_xy[valid, None], dim=-1) * fm
    ade = dist.detach().sum(dim=2) / cnt
    winner = ade[:, 1:] == ade[:, 1:].amin(dim=1, keepdim=True)
    m_w = winner.double() / winner.sum(dim=1, keepdim=True)
    huber = ((torch.sqrt(dist[:, 1:].square() + 0.25) - 0.5) * fm).sum(dim=2) / cnt
    reg_i = (huber * m_w).sum(dim=1)
    if reg_only:
        return reg_i.mean()
    soft = (-ade / 0.25).softmax(dim=1).to(out["logits"].dtype)
    cls_i = -(soft * out["logits"][valid].log_softmax(dim=1)).sum(dim=1).double()
    return (reg_i + 0.25 * cls_i).mean()


def fit_jta_models(seed: int, tr: dict, va: dict) -> tuple[O2NullSinkGraphForecaster, float, float]:
    torch.manual_seed(seed)
    base = O2BioEqJointForecaster(pose_mode="bio_pose", action_mode="none")
    for n, p in base.named_parameters():
        p.requires_grad = not (n.startswith("bio_") or n == "gate_logit")
    gen = torch.Generator().manual_seed(seed)
    opt1 = torch.optim.AdamW([p for p in base.parameters() if p.requires_grad], lr=2.5e-3, weight_decay=1e-4)
    for _ in range(30):
        for ch in torch.randperm(len(tr["history"]), generator=gen).split(128):
            out = base(tr["history"][ch], obs_dt=tr["obs_dt"][ch], fut_times_s=tr["fut_s"][ch])
            loss = masked_jta_loss(out, tr["fut_xy"][ch], tr["fut_mask"][ch])
            if loss is not None:
                opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()

    for n, p in base.named_parameters():
        p.requires_grad = n.startswith("bio_") or n == "gate_logit"
    opt2 = torch.optim.AdamW([p for p in base.parameters() if p.requires_grad], lr=1.5e-3, weight_decay=2e-4)
    for _ in range(20):
        for ch in torch.randperm(len(tr["history"]), generator=gen).split(128):
            out = base(tr["history"][ch], tr["bio_seq"][ch], tr["q_bio"][ch], obs_dt=tr["obs_dt"][ch], fut_times_s=tr["fut_s"][ch])
            loss = masked_jta_loss(out, tr["fut_xy"][ch], tr["fut_mask"][ch], reg_only=True)
            if loss is not None:
                opt2.zero_grad(set_to_none=True); loss.backward(); opt2.step()

    base.eval()
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    with torch.inference_mode():
        b_scores = []
        for bsc in grid:
            ov = base(va["history"], va["bio_seq"], va["q_bio"] * bsc, obs_dt=va["obs_dt"], fut_times_s=va["fut_s"])
            m = compute_sample_trajectory_metrics(ov["paths"].numpy(), ov["logits"].softmax(1).numpy(), va["fut_xy"].numpy(), va["fut_mask"].numpy())
            b_scores.append(float(m["ADE_1"].mean()))
    best_bsc = float(grid[int(np.argmin(b_scores))])

    for p in base.parameters():
        p.requires_grad = False
    gnet = O2NullSinkGraphForecaster(base)
    optg = torch.optim.AdamW([p for n, p in gnet.named_parameters() if not n.startswith("base.")], lr=1.5e-3, weight_decay=2e-4)
    for _ in range(24):
        for ch in torch.randperm(len(tr["history"]), generator=gen).split(128):
            out = gnet(tr["history"][ch], tr["edges"][ch], tr["is_car"][ch], tr["nbr_mask"][ch],
                       tr["bio_seq"][ch], tr["q_bio"][ch] * best_bsc, graph_scale=1.0, obs_dt=tr["obs_dt"][ch], fut_times_s=tr["fut_s"][ch])
            loss = masked_jta_loss(out, tr["fut_xy"][ch], tr["fut_mask"][ch])
            if loss is not None:
                optg.zero_grad(set_to_none=True); loss.backward(); optg.step()
    gnet.eval()
    with torch.inference_mode():
        g_scores = []
        for gsc in grid:
            ov = gnet(va["history"], va["edges"], va["is_car"], va["nbr_mask"], va["bio_seq"], va["q_bio"] * best_bsc,
                      graph_scale=gsc, obs_dt=va["obs_dt"], fut_times_s=va["fut_s"])
            m = compute_sample_trajectory_metrics(ov["paths"].numpy(), ov["logits"].softmax(1).numpy(), va["fut_xy"].numpy(), va["fut_mask"].numpy())
            g_scores.append(float(m["ADE_1"].mean()))
    best_gsc = float(grid[int(np.argmin(g_scores))])
    return gnet, best_bsc, best_gsc


@torch.inference_mode()
def eval_jta_dict(gnet: O2NullSinkGraphForecaster, te: dict, bsc: float, gsc: float) -> dict[str, np.ndarray]:
    out = gnet(te["history"], te["edges"], te["is_car"], te["nbr_mask"], te["bio_seq"], te["q_bio"] * bsc,
               graph_scale=gsc, obs_dt=te["obs_dt"], fut_times_s=te["fut_s"])
    paths = out["paths"].numpy()
    probs = out["logits"].softmax(1).numpy()
    m = compute_sample_trajectory_metrics(paths, probs, te["fut_xy"].numpy(), te["fut_mask"].numpy())
    bay = np.einsum("nk,nktw->ntw", probs, paths)
    fm = (te["fut_mask"].numpy() > 0.5).astype(np.float64)
    gt = te["fut_xy"].numpy()
    d_bay = np.linalg.norm(bay - gt, axis=-1) * fm
    m["Bayes_ADE"] = d_bay.sum(axis=1) / np.maximum(fm.sum(axis=1), 1.0)
    last_i = np.argmax(np.arange(12)[None, :] * (fm > 0.5), axis=1)
    m["Bayes_FDE"] = np.linalg.norm(bay[np.arange(len(gt)), last_i] - gt[np.arange(len(gt)), last_i], axis=-1)
    return m


def summarize_jta(seed_list: list[dict[str, np.ndarray]], bk: np.ndarray) -> dict:
    keys = ("ADE_1", "FDE_1", "minADE_6", "minFDE_6", "Bayes_ADE", "Bayes_FDE")
    avg_samp = {k: np.mean([d[k] for d in seed_list], axis=0) for k in keys}
    slices = {}
    for s_name, mask in (("Overall", np.ones(len(bk), dtype=bool)), ("Clear", bk == 0), ("Partial", bk == 1), ("Heavy", bk == 2)):
        slices[s_name] = {"count": int(mask.sum()), **{k: float(avg_samp[k][mask].mean()) for k in keys}}
    return {"slices": slices, "_ade1": avg_samp["ADE_1"], "_bayes_ade": avg_samp["Bayes_ADE"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_zip", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix="jta_o2_") as td:
        sc = Path(td)
        st_zip = sc / "st_releases.zip"
        st_dir = sc / "st_unpacked"
        _download_verified_zip(ST_RELEASE_URL, st_zip)
        _unpack_verified_zip(st_zip, st_dir)
        jta_root = [p for p in st_dir.rglob("data") if p.is_dir() and "jta" in str(p).lower() and "__macosx" not in str(p).lower()][0]
        jta_samples, jta_audit = process_jta_root(jta_root, max_files_per_split=24, seed=42)

    report = {"jta_audit": jta_audit, "seeds": list(args.seeds), "durations": {}}
    for t_obs in (2, 8):
        tr = build_jta_tensors(jta_samples, "train", t_obs)
        va = build_jta_tensors(jta_samples, "val", t_obs)
        te = build_jta_tensors(jta_samples, "test", t_obs)
        cv_p = ((te["history"][:, -1:] - te["history"][:, -2:-1]) / te["obs_dt"][:, -1:, None]).numpy() * te["fut_s"].numpy()[:, :, None]
        cv_m = compute_sample_trajectory_metrics(np.repeat(cv_p[:, None], 6, axis=1), np.full((len(cv_p), 6), 1.0 / 6.0), te["fut_xy"].numpy(), te["fut_mask"].numpy())
        cv_m["Bayes_ADE"], cv_m["Bayes_FDE"] = cv_m["ADE_1"], cv_m["FDE_1"]

        mot_s, bio_s, grp_s, eq_swings = [], [], [], []
        for seed in args.seeds:
            gn, bsc, gsc = fit_jta_models(seed, tr, va)
            mot_s.append(eval_jta_dict(gn, te, 0.0, 0.0))
            bio_s.append(eval_jta_dict(gn, te, bsc, 0.0))
            grp_s.append(eval_jta_dict(gn, te, bsc, gsc))
            if seed == 42:
                ades = []
                for mat in ([[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]], [[1, 0], [0, -1]]):
                    r = np.array(mat, dtype=np.float64)
                    te_r = {**te, "history": te["history"] @ torch.from_numpy(r.T)}
                    er = te["edges"].clone()
                    er[:, :, 0:2] = (er[:, :, 0:2].double() @ torch.from_numpy(r.T)).float()
                    er[:, :, 2:4] = (er[:, :, 2:4].double() @ torch.from_numpy(r.T)).float()
                    te_r["edges"] = er
                    with torch.inference_mode():
                        oo = gn(te_r["history"], te_r["edges"], te_r["is_car"], te_r["nbr_mask"], te_r["bio_seq"], te_r["q_bio"] * bsc, graph_scale=gsc, obs_dt=te_r["obs_dt"], fut_times_s=te_r["fut_s"])
                        pw = oo["paths"].numpy() @ r
                        mm = compute_sample_trajectory_metrics(pw, oo["logits"].softmax(1).numpy(), te["fut_xy"].numpy(), te["fut_mask"].numpy())
                        ades.append(float(mm["ADE_1"].mean()))
                eq_swings.append(max(ades) - min(ades))
            print(f"Done JTA T={t_obs} seed={seed}", flush=True)

        bk = te["bucket"]
        sm_cv = summarize_jta([cv_m], bk)
        sm_mot = summarize_jta(mot_s, bk)
        sm_bio = summarize_jta(bio_s, bk)
        sm_grp = summarize_jta(grp_s, bk)
        t_entry = {
            "cv_last2": {k: v for k, v in sm_cv.items() if not k.startswith("_")},
            "o2_motion": {k: v for k, v in sm_mot.items() if not k.startswith("_")},
            "o2_bioeq_v2": {k: v for k, v in sm_bio.items() if not k.startswith("_")},
            "o2_nullsink_graph": {k: v for k, v in sm_grp.items() if not k.startswith("_")},
            "seed42_four_axis_ade_swing_m": float(eq_swings[0]),
            "paired_tests": {
                "graph_vs_cv_overall": paired_hypothesis_test(sm_grp["_ade1"], sm_cv["_ade1"]),
                "bayes_graph_vs_cv_overall": paired_hypothesis_test(sm_grp["_bayes_ade"], sm_cv["_ade1"]),
            },
        }
        report["durations"][f"T{t_obs}"] = t_entry

    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    members = {"report.json": (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()}
    members["sha256.json"] = (json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in members.items()}, indent=2, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in members.items():
            zf.writestr(k, v)
    print("Saved", args.output_zip, flush=True)


if __name__ == "__main__":
    main()
