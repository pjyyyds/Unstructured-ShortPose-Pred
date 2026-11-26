"""Zero-shot 5-seed Ours_O2_Full neighbor-graph evaluation on 54 Beijing DAIR-V2X roadside windows."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

import numpy as np
import torch

from scripts.run_dair_attached_o2_bioeq import load_attached_dair_54
from scripts.run_dair_o2_bioeq_external import COCO_SWAP, RESILIENT_SPAN, SEEDS, person_bootstrap_ci
from scripts.run_imptc_and_dair_o2_full_external import TRANSFORMS
from scripts.run_pinns_matched_o2_multiduration import SITES
from scripts.run_unified_o2_full_gpu_benchmark import fit_unified_fold, load_unified_cohort, select_unified_scales
from src.models.bioeq_neural import biomech_sequence_torch

SPLIT_URL = "https://raw.githubusercontent.com/AIR-THU/DAIR-V2X/c885c54af0c34bc515fa9ca8b5e8fda76a15462c/data/split_datas/cooperative-split-data-spd.json"
SPLIT_SHA = "0a59332221f3618639f34fb70678d03c5221ab80fd69f2cf41acffca0d8859fd"
VEHICLE_TYPES = {"Car", "Van", "Truck", "Bus", "Motorcyclist"}
VRU_TYPES = {"Pedestrian", "Cyclist", "Tricyclist", "Barrowlist"}


def build_dair_world_neighbor_edges(repo_root: Path, dair_zip: Path, work_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import importlib.util
    import shutil
    import tempfile

    src_dir = repo_root / "reviews/dair_roadside_external"
    exp_dir = (work_dir or Path(tempfile.mkdtemp(prefix="dair_nb_"))) / "dair_range"
    exp_dir.mkdir(parents=True, exist_ok=True)
    fr_txt = (src_dir / "full_range.py").read_text().replace("max_workers=10", "max_workers=2").replace("max_workers=5", "max_workers=2")
    fr_txt = fr_txt.split("if __name__=='__main__':")[0] + "\n" + RESILIENT_SPAN + "\n"
    fr_path = exp_dir / "full_range.py"
    fr_path.write_text(fr_txt)
    spec = importlib.util.spec_from_file_location("dair_full_range_resilient", fr_path)
    assert spec is not None and spec.loader is not None
    full_range = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(full_range)

    with zipfile.ZipFile(dair_zip) as zf:
        import io
        with np.load(io.BytesIO(zf.read("private.npz")), allow_pickle=False) as ar:
            keys = json.loads(str(ar["keys_json"]))
    raw_split = urllib.request.urlopen(SPLIT_URL, timeout=45).read()
    if hashlib.sha256(raw_split).hexdigest() != SPLIT_SHA:
        raise ValueError("DAIR official split SHA mismatch")

    D = "V2X-Seq-SPD/infrastructure-side/"
    m_dir = full_range.directory("metadata")
    info = json.loads(full_range.ranges("metadata", m_dir, [D + "data_info.json"])[D + "data_info.json"].read_text())
    need_seqs = {str(k[0]) for k in keys}
    fr_map = {(r["sequence_id"], int(r["frame_id"])): r for r in info if r["sequence_id"] in need_seqs}

    need_files = []
    for s, _, t0 in keys:
        for dt_f in range(-4, 1):
            r = fr_map[(str(s), int(t0) + dt_f)]
            need_files.extend([D + r["label_camera_std_path"], D + r["calib_virtuallidar_to_world_path"]])
    fetched = full_range.ranges("metadata", m_dir, need_files)
    return _assemble_dair_edges(keys, fr_map, fetched, D)


def _assemble_dair_edges(keys: list, fr_map: dict, fetched: dict, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(keys)
    edges = np.zeros((n, 12, 9), dtype=np.float32)
    cars = np.zeros((n, 12), dtype=np.float32)
    masks = np.zeros((n, 12), dtype=np.float32)
    unambig = np.ones(n, dtype=bool)
    allowed = VEHICLE_TYPES | VRU_TYPES
    for idx, (s, pid, t0) in enumerate(keys):
        snaps, ts_s = [], []
        dup_pairs_t0: set[tuple[str, str]] = set()
        for dt_f in range(-4, 1):
            r = fr_map[(str(s), int(t0) + dt_f)]
            cal = json.loads(fetched[prefix + r["calib_virtuallidar_to_world_path"]].read_text())
            rot = np.asarray(cal["rotation"], dtype=np.float64)
            trn = np.asarray(cal["translation"], dtype=np.float64).reshape(3)
            obs = json.loads(fetched[prefix + r["label_camera_std_path"]].read_text())
            cnts = collections.Counter((o["type"], str(o["track_id"])) for o in obs if o["type"] in allowed)
            dups = {pair for pair, c in cnts.items() if c > 1}
            if dt_f == 0 and dups:
                unambig[idx] = False
                dup_pairs_t0 = dups
            cur = {}
            for o in obs:
                tp, tid = o["type"], str(o["track_id"])
                if tp not in allowed or (tp, tid) in dups:
                    continue
                loc = o["3d_location"]
                pt_l = np.array([float(loc["x"]), float(loc["y"]), float(loc["z"])], dtype=np.float64)
                if np.isfinite(pt_l).all():
                    cur[(tp, tid)] = (rot @ pt_l + trn)[:2]
            snaps.append(cur)
            ts_s.append(int(r["image_timestamp"]) / 1e6)
        ego = ("Pedestrian", str(int(pid)))
        now = snaps[-1]
        if ego not in now:
            raise ValueError(f"Missing target pedestrian at {s}/{pid}/{t0}")

        def _vel(k: tuple[str, str]) -> tuple[np.ndarray, int, float]:
            cnt = sum(k in sn for sn in snaps)
            gap = next((g for g in range(1, 5) if k in snaps[-1 - g]), 0)
            v = (now[k] - snaps[-1 - gap][k]) / max(ts_s[-1] - ts_s[-1 - gap], 1e-4) if gap else np.zeros(2)
            return v, gap, cnt / 5.0

        ev, eg, rho_e = _vel(ego)
        pool = []
        for k_nb, p_nb in now.items():
            if k_nb == ego or k_nb in dup_pairs_t0:
                continue
            dist = float(np.hypot(*(p_nb - now[ego])))
            if 1e-5 < dist <= 10.0:
                pool.append((dist, k_nb))
        pool.sort()
        for j, (dist, k_nb) in enumerate(pool[:12]):
            vn, gn, rho_j = _vel(k_nb)
            dp = now[k_nb] - now[ego]
            ok = bool(eg and gn)
            dv = vn - ev if ok else np.zeros(2)
            rad = -float(dp @ dv) / (dist + 1e-6) if ok else 0.0
            edges[idx, j] = [dp[0], dp[1], dv[0], dv[1], dist, rad, float(ok), rho_j, rho_e]
            cars[idx, j] = 1.0 if k_nb[0] in VEHICLE_TYPES else 0.0
            masks[idx, j] = 1.0
    if int(unambig.sum()) != 51:
        raise ValueError(f"Expected 51 unambiguous DAIR windows per frozen audit, got {int(unambig.sum())}")
    return edges, cars, masks, unambig


@torch.inference_mode()
def score_dair_graph_vs_nograph(dair: dict, edges: np.ndarray, cars: np.ndarray, masks: np.ndarray, unambig: np.ndarray, ens_t2: list) -> dict:
    ts = dair["ts_us"]
    obs_dt = torch.from_numpy(((ts[:, 4:5] - ts[:, 3:4]) / 1e6).astype(np.float64))
    fut_s = torch.from_numpy(((ts[:, 5:] - ts[:, 4:5]) / 1e6).astype(np.float64))
    truth = dair["truth"]
    n = len(truth)
    strict_m = np.max(np.abs(np.diff(ts[:, :5], axis=1) - 100000), axis=1) <= 1000
    by_tf: dict[str, dict] = {}
    base_e: dict[str, np.ndarray] = {}
    for tf_name, op, refl in TRANSFORMS:
        hg = torch.from_numpy(dair["hist2"] @ op.T)
        kp = dair["kpts"].copy()
        if refl:
            kp[..., 0] = 1920.0 - kp[..., 0]
            kp = kp[:, :, COCO_SWAP, :]
        bs_t, qb_t = biomech_sequence_torch(torch.from_numpy(kp), torch.from_numpy(dair["bbox"]))
        eg = edges.copy()
        eg[:, :, 0:2] = (eg[:, :, 0:2].astype(np.float64) @ op.T).astype(np.float32)
        eg[:, :, 2:4] = (eg[:, :, 2:4].astype(np.float64) @ op.T).astype(np.float32)
        eg_t, car_t, msk_t = torch.from_numpy(eg), torch.from_numpy(cars), torch.from_numpy(masks)

        arms = {"no_graph_bayes": [], "real_graph_bayes": [], "real_graph_top1": []}
        for net, bsc, gsc in ens_t2:
            for tag, e_in, g_in, mode in (
                ("no_graph_bayes", None, 0.0, "bayes"),
                ("real_graph_bayes", eg_t, max(gsc, 0.25), "bayes"),
                ("real_graph_top1", eg_t, max(gsc, 0.25), "top1"),
            ):
                oo = net(hg, bs_t, qb_t, e_in, car_t if e_in is not None else None, msk_t if e_in is not None else None, bio_scale=bsc, graph_scale=g_in, use_action_bridge=True, obs_dt=obs_dt, fut_times_s=fut_s)
                pw = oo["paths"].numpy() @ op
                pr = oo["logits"].softmax(1).numpy()
                pred = pw[np.arange(n), pr.argmax(1)] if mode == "top1" else np.einsum("nk,nktw->ntw", pr, pw)
                d = np.linalg.norm(pred - truth, axis=-1)
                dm = np.linalg.norm(pw - truth[:, None], axis=-1)
                arms[tag].append(np.stack((d.mean(1), d[:, -1], dm[:, :, -1].min(1)), axis=-1))
        cur = {k: np.mean(v, axis=0) for k, v in arms.items()}
        if tf_name == "0deg":
            base_e = cur
        by_tf[tf_name] = {
            sub_k: {m: {"ade_m": float(e[sub_m, 0].mean()), "fde_m": float(e[sub_m, 1].mean()), "min_fde6_m": float(e[sub_m, 2].mean())} for m, e in cur.items()}
            for sub_k, sub_m in (("all_54", np.ones(n, dtype=bool)), ("unambig_51", unambig), ("strict_31", strict_m))
        }
    rot4_sw = float(max(by_tf[t]["unambig_51"]["real_graph_bayes"]["ade_m"] for t, _, _ in TRANSFORMS[:4]) - min(by_tf[t]["unambig_51"]["real_graph_bayes"]["ade_m"] for t, _, _ in TRANSFORMS[:4]))
    return {
        "n_all": n,
        "n_unambiguous": int(unambig.sum()),
        "windows_with_10m_neighbors": int((masks.sum(1) > 0).sum()),
        "windows_with_10m_vehicles_or_moto": int((cars.sum(1) > 0).sum()),
        "rot4_ade_swing_m": rot4_sw,
        "by_transform": by_tf,
        "paired_person_bootstrap_ci_0deg": {
            "unambig51_graph_minus_nograph": person_bootstrap_ci(base_e["real_graph_bayes"][:, :2] - base_e["no_graph_bayes"][:, :2], unambig, dair["actors"]),
            "all54_graph_minus_nograph": person_bootstrap_ci(base_e["real_graph_bayes"][:, :2] - base_e["no_graph_bayes"][:, :2], np.ones(n, dtype=bool), dair["actors"]),
            "strict31_graph_minus_nograph": person_bootstrap_ci(base_e["real_graph_bayes"][:, :2] - base_e["no_graph_bayes"][:, :2], strict_m, dair["actors"]),
        },
    }


def run_dair_graph_suite(repo_root: Path, meta_dir: Path, pose_npz: Path, joint_zip: Path, dair_zip: Path, out_dir: Path, seeds: tuple[int, ...] = SEEDS) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    dair, _ = load_attached_dair_54(dair_zip)
    edges, cars, masks, unambig = build_dair_world_neighbor_edges(repo_root, dair_zip)
    data5 = load_unified_cohort(meta_dir, pose_npz, joint_zip)
    data2 = {**data5, "history": data5["history"][:, -2:]}
    ens_t2 = []
    for seed in seeds:
        for s in SITES:
            tr_ids = np.flatnonzero(data2["site"] != s)
            bs2, gs2 = select_unified_scales(seed + 101, data2, tr_ids, device)
            n2 = fit_unified_fold(seed + 101, data2, tr_ids, device).cpu().eval()
            ens_t2.append((n2, bs2, gs2))
    rep = {"study": "DAIR-V2X 54-window real 10m VRU/Vehicle neighbor-graph evaluation for Ours_O2_Full", "seeds": list(seeds), "dair_neighbor_graph": score_dair_graph_vs_nograph(dair, edges, cars, masks, unambig, ens_t2)}
    rp, sp = out_dir / "report.json", out_dir / "sha256.json"
    rp.write_text(json.dumps(rep, indent=2) + "\n")
    sp.write_text(json.dumps({"report.json": hashlib.sha256(rp.read_bytes()).hexdigest()}, indent=2) + "\n")
    with zipfile.ZipFile(out_dir / "dair_roadside_neighbor_graph_o2_5seed.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(rp, "report.json"); zf.write(sp, "sha256.json")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--dair-zip", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    run_dair_graph_suite(args.repo_root, args.meta_dir, args.pose_npz, args.joint_zip, args.dair_zip, args.out_dir)
