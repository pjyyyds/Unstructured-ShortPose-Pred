"""Zero-shot 5-seed Ours_O2_Full + Null-Sink Graph on 1,677 outdoor JRDB 2.5 Hz windows (3 scenes)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

from scripts.run_dair_o2_bioeq_external import SEEDS, person_bootstrap_ci
from scripts.run_imptc_and_dair_o2_full_external import TRANSFORMS
from scripts.run_pinns_matched_o2_multiduration import SITES
from scripts.run_unified_o2_full_gpu_benchmark import fit_unified_fold, load_unified_cohort, select_unified_scales


def load_jrdb_1677_with_neighbors(repo_root: Path) -> dict:
    jrdb_dir = repo_root / "reviews/jrdb_development"
    if str(jrdb_dir) not in sys.path:
        sys.path.insert(0, str(jrdb_dir))
    import check_training_coordinates as ctc  # type: ignore
    import evaluate_outdoor_motion as eom  # type: ignore

    tr_arc = ctc.TrainingArchive(*eom.ARCHIVES["tracks"])
    od_arc = ctc.TrainingArchive(*eom.ARCHIVES["odom"])
    hist_l, truth_l, scene_l, actor_l = [], [], [], []
    edges_l, masks_l = [], []

    for i, sc in enumerate(eom.SCENES):
        b_tr = tr_arc.member(eom.MEMBERS["tracks"].format(sc))
        b_od = od_arc.member(eom.MEMBERS["odom"].format(sc))
        if hashlib.sha256(b_tr).hexdigest() != eom.DIGESTS["tracks"][i] or hashlib.sha256(b_od).hexdigest() != eom.DIGESTS["odom"][i]:
            raise ValueError(f"JRDB SHA-256 mismatch for {sc}")
        tracks, _ = eom.read_tracks(b_tr)
        ticks = {f for rows in tracks.values() for f in rows}
        _ = eom.odometry(b_od, ticks)
        by_frame: dict[int, dict[int, np.ndarray]] = {}
        for p, rows in tracks.items():
            for f, xy in rows.items():
                by_frame.setdefault(int(f), {})[int(p)] = np.asarray(xy, dtype=np.float64)

        for person, rows in sorted(tracks.items()):
            times = np.asarray(sorted(rows))
            for run in np.split(times, np.flatnonzero(np.diff(times) != eom.STEP) + 1):
                for off in range(0, len(run) - eom.FUTURE - 1, 2 + eom.FUTURE):
                    ch = run[off : off + 2 + eom.FUTURE]
                    xy = np.stack([np.asarray(rows[int(t)], dtype=np.float64) for t in ch])
                    hist_l.append(xy[:2] - xy[1:2])
                    truth_l.append(xy[2:] - xy[1:2])
                    scene_l.append(sc)
                    actor_l.append(f"{sc}:{person}")
                    e_cur, m_cur = _jrdb_edges(by_frame, int(ch[0]), int(ch[1]), int(person))
                    edges_l.append(e_cur)
                    masks_l.append(m_cur)
    if len(hist_l) != 1677:
        raise ValueError(f"Expected 1677 JRDB windows, found {len(hist_l)}")
    return {
        "history": np.stack(hist_l),
        "truth": np.stack(truth_l),
        "scene": np.asarray(scene_l, dtype=str),
        "actors": np.asarray(actor_l, dtype=str),
        "edges": np.stack(edges_l),
        "is_car": np.zeros((len(hist_l), 12), dtype=np.float32),
        "nbr_mask": np.stack(masks_l),
        "scenes_order": list(eom.SCENES),
    }


def _jrdb_edges(by_frame: dict, f0: int, f1: int, ego_p: int) -> tuple[np.ndarray, np.ndarray]:
    s0, s1 = by_frame.get(f0, {}), by_frame.get(f1, {})
    ego_xy = s1[ego_p]
    ego_v = (ego_xy - s0[ego_p]) / 0.4
    pool = []
    for pid, p_xy in s1.items():
        if pid == ego_p:
            continue
        dist = float(np.hypot(*(p_xy - ego_xy)))
        if 1e-5 < dist <= 10.0:
            pool.append((dist, pid))
    pool.sort()
    edge = np.zeros((12, 9), dtype=np.float32)
    mask = np.zeros(12, dtype=np.float32)
    for j, (dist, pid) in enumerate(pool[:12]):
        dp = s1[pid] - ego_xy
        ok = pid in s0
        dv = ((s1[pid] - s0[pid]) / 0.4 - ego_v) if ok else np.zeros(2)
        rad = -float(dp @ dv) / (dist + 1e-6) if ok else 0.0
        edge[j] = [dp[0], dp[1], dv[0], dv[1], dist, rad, float(ok), 1.0 if ok else 0.5, 1.0]
        mask[j] = 1.0
    return edge, mask


@torch.inference_mode()
def score_jrdb_ensemble(jrdb: dict, ens_t2: list) -> dict:
    n = len(jrdb["truth"])
    obs_dt = torch.full((n, 1), 0.4, dtype=torch.float64)
    fut_s = torch.arange(1, 8, dtype=torch.float64)[None, :].expand(n, -1) * 0.4
    truth = jrdb["truth"]
    scenes = jrdb["scenes_order"]
    by_tf: dict[str, dict] = {}
    base_e: dict[str, np.ndarray] = {}
    for tf_name, op, _ in TRANSFORMS:
        hg = torch.from_numpy(jrdb["history"] @ op.T)
        eg = jrdb["edges"].copy()
        eg[:, :, 0:2] = (eg[:, :, 0:2].astype(np.float64) @ op.T).astype(np.float32)
        eg[:, :, 2:4] = (eg[:, :, 2:4].astype(np.float64) @ op.T).astype(np.float32)
        eg_t, car_t, msk_t = torch.from_numpy(eg), torch.from_numpy(jrdb["is_car"]), torch.from_numpy(jrdb["nbr_mask"])

        cv_p = (jrdb["history"][:, 1:] - jrdb["history"][:, :1]) * np.arange(1, 8, dtype=np.float64)[None, :, None]
        d_cv = np.linalg.norm(cv_p - truth, axis=-1)
        e_cv = np.stack((d_cv.mean(1), d_cv[:, -1], d_cv[:, -1]), axis=-1)

        arms = {"no_graph_bayes": [], "graph_bayes": [], "graph_top1": []}
        for net, bsc, gsc in ens_t2:
            for tag, e_in, g_in, mode in (("no_graph_bayes", None, 0.0, "bayes"), ("graph_bayes", eg_t, max(gsc, 0.25), "bayes"), ("graph_top1", eg_t, max(gsc, 0.25), "top1")):
                oo = net(hg, None, None, e_in, car_t if e_in is not None else None, msk_t if e_in is not None else None, bio_scale=0.0, graph_scale=g_in, use_action_bridge=True, obs_dt=obs_dt, fut_times_s=fut_s)
                pw = oo["paths"].numpy() @ op
                pr = oo["logits"].softmax(1).numpy()
                pred = pw[np.arange(n), pr.argmax(1)] if mode == "top1" else np.einsum("nk,nktw->ntw", pr, pw)
                d = np.linalg.norm(pred - truth, axis=-1)
                dm = np.linalg.norm(pw - truth[:, None], axis=-1)
                arms[tag].append(np.stack((d.mean(1), d[:, -1], dm[:, :, -1].min(1)), axis=-1))
        cur = {"last_two_cv": e_cv, **{k: np.mean(v, axis=0) for k, v in arms.items()}}
        if tf_name == "0deg":
            base_e = cur
        by_tf[tf_name] = {
            m: {
                "scene_equal_ade_m": float(np.mean([e[jrdb["scene"] == sc, 0].mean() for sc in scenes])),
                "scene_equal_fde_m": float(np.mean([e[jrdb["scene"] == sc, 1].mean() for sc in scenes])),
                "scene_equal_min_fde6_m": float(np.mean([e[jrdb["scene"] == sc, 2].mean() for sc in scenes])),
                "pooled_ade_m": float(e[:, 0].mean()),
                "pooled_fde_m": float(e[:, 1].mean()),
            }
            for m, e in cur.items()
        }
    sw = float(max(by_tf[t]["graph_bayes"]["pooled_ade_m"] for t, _, _ in TRANSFORMS) - min(by_tf[t]["graph_bayes"]["pooled_ade_m"] for t, _, _ in TRANSFORMS))
    return {
        "n_windows": n,
        "windows_with_10m_neighbors": int((jrdb["nbr_mask"].sum(1) > 0).sum()),
        "five_axis_max_ade_swing_m": sw,
        "by_transform": by_tf,
        "by_scene_0deg": {sc: {m: {"ade_m": float(e[jrdb["scene"] == sc, 0].mean()), "fde_m": float(e[jrdb["scene"] == sc, 1].mean())} for m, e in base_e.items()} for sc in scenes},
        "paired_person_bootstrap_ci_0deg": {
            "graph_bayes_minus_cv": person_bootstrap_ci(base_e["graph_bayes"][:, :2] - base_e["last_two_cv"][:, :2], np.ones(n, dtype=bool), jrdb["actors"]),
            "graph_minus_nograph": person_bootstrap_ci(base_e["graph_bayes"][:, :2] - base_e["no_graph_bayes"][:, :2], np.ones(n, dtype=bool), jrdb["actors"]),
        },
    }


def run_jrdb_suite(repo_root: Path, meta_dir: Path, pose_npz: Path, joint_zip: Path, out_dir: Path, seeds: tuple[int, ...] = SEEDS) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    jrdb = load_jrdb_1677_with_neighbors(repo_root)
    data5 = load_unified_cohort(meta_dir, pose_npz, joint_zip)
    data2 = {**data5, "history": data5["history"][:, -2:]}
    ens_t2 = []
    for seed in seeds:
        for s in SITES:
            tr_ids = np.flatnonzero(data2["site"] != s)
            bs2, gs2 = select_unified_scales(seed + 101, data2, tr_ids, device)
            n2 = fit_unified_fold(seed + 101, data2, tr_ids, device).cpu().eval()
            ens_t2.append((n2, bs2, gs2))
    rep = {"study": "JRDB 1,677 outdoor 2.5 Hz zero-shot evaluation of 5-seed Ours_O2_Full + Null-Sink Graph", "seeds": list(seeds), "jrdb_outdoor_1677": score_jrdb_ensemble(jrdb, ens_t2)}
    rp, sp = out_dir / "report.json", out_dir / "sha256.json"
    rp.write_text(json.dumps(rep, indent=2) + "\n")
    sp.write_text(json.dumps({"report.json": hashlib.sha256(rp.read_bytes()).hexdigest()}, indent=2) + "\n")
    with zipfile.ZipFile(out_dir / "jrdb_outdoor_o2_full_5seed.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(rp, "report.json"); zf.write(sp, "sha256.json")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    run_jrdb_suite(args.repo_root, args.meta_dir, args.pose_npz, args.joint_zip, args.out_dir)
