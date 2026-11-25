"""Zero-shot 5-seed unified Ours_O2_Full evaluation on German IMPTC (564) & Beijing DAIR-V2X (54)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tarfile
import time
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import torch

from reviews.imptc_external.external_motion import SAMPLE_MD5, SEQS, coordinates_for, prepare
from scripts.run_dair_attached_o2_bioeq import load_attached_dair_54
from scripts.run_dair_o2_bioeq_external import COCO_SWAP, SEEDS, person_bootstrap_ci, reduce_dair
from scripts.run_pinns_matched_o2_multiduration import SITES
from scripts.run_unified_o2_full_gpu_benchmark import fit_unified_fold, load_unified_cohort, select_unified_scales
from src.models.bioeq_neural import biomech_sequence_torch
from src.models.o2_unified_full import O2UnifiedFullForecaster

ZENODO_IMPTC = "https://zenodo.org/api/records/14811016"


def ensure_imptc_sample(dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size == 347692691 and hashlib.file_digest(dest.open("rb"), "md5").hexdigest() == SAMPLE_MD5:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(ZENODO_IMPTC, timeout=30) as r:
        rec = json.load(r)
    spec = next(i for i in rec["files"] if i["key"] == "imptc_samples.tar.gz")
    total, url = int(spec["size"]), spec["links"]["self"]
    if total != 347692691 or spec["checksum"] != f"md5:{SAMPLE_MD5}":
        raise ValueError("Unexpected Zenodo IMPTC metadata")

    def _chunk(args_t: tuple[int, int]) -> bytes:
        s, e = args_t
        for retry in range(4):
            try:
                req = Request(f"{url}?bytes={s}-{e - 1}", headers={"Range": f"bytes={s}-{e - 1}", "User-Agent": "ShortPose-Audit/1.0"})
                with urlopen(req, timeout=50) as resp:
                    blob = resp.read(e - s + 1)
                if len(blob) == e - s:
                    return blob
            except Exception:
                if retry == 3:
                    raise
                time.sleep(2**retry)
        raise RuntimeError("Chunk download failed")

    part = dest.with_suffix(".part")
    step = 8 * 1024 * 1024
    with ThreadPoolExecutor(4) as pool, part.open("wb") as wf:
        for b in pool.map(_chunk, [(s, min(total, s + step)) for s in range(0, total, step)]):
            wf.write(b)
    if part.stat().st_size != total or hashlib.file_digest(part.open("rb"), "md5").hexdigest() != SAMPLE_MD5:
        raise ValueError("Downloaded IMPTC sample failed MD5 check")
    part.replace(dest)
    return dest


def load_imptc_with_neighbors(sample_path: Path) -> dict:
    windows, cohort = prepare(sample_path)
    if len(windows) != 564:
        raise ValueError(f"Expected 564 IMPTC windows, got {len(windows)}")
    hist_abs = coordinates_for(sample_path, windows, slice(0, 5))
    fut_abs = coordinates_for(sample_path, windows, slice(5, 35))
    hist_rel = hist_abs - hist_abs[:, -1:]
    truth_rel = fut_abs - hist_abs[:, -1:]
    ts_us = np.asarray([w.ticks for w in windows], dtype=np.int64)

    need_ticks = {s: set() for s in SEQS}
    for w in windows:
        need_ticks[w.seq].update(w.ticks[:5])
    snaps = {s: {t: {} for t in need_ticks[s]} for s in SEQS}
    with tarfile.open(sample_path, mode="r|gz") as arc:
        for m in arc:
            if not m.isfile():
                continue
            pts = Path(m.name).parts
            if len(pts) == 4 and pts[0] in SEQS and pts[1] in ("vrus", "vehicles") and pts[3] == "track.json":
                tr = json.load(arc.extractfile(m))
                kind = "Car" if pts[1] == "vehicles" else "Person"
                key = (kind, str(pts[2]))
                wanted = need_ticks[pts[0]]
                for rec in tr.get("track_data", {}).values():
                    t = int(rec["ts"])
                    if t in wanted and rec.get("status") == 2:
                        snaps[pts[0]][t][key] = np.asarray(rec["coordinates"][:2], dtype=np.float64)

    edges = np.zeros((len(windows), 12, 9), dtype=np.float32)
    cars = np.zeros((len(windows), 12), dtype=np.float32)
    masks = np.zeros((len(windows), 12), dtype=np.float32)
    for idx, w in enumerate(windows):
        t5 = w.ticks[:5]
        h_snaps = [snaps[w.seq][t] for t in t5]
        ego = ("Person", str(w.track))
        now = h_snaps[-1]
        if ego not in now:
            continue

        def _vel(k: tuple[str, str]) -> tuple[np.ndarray, int, float]:
            cnt = sum(k in s for s in h_snaps)
            gap = next((d for d in range(1, 5) if k in h_snaps[-1 - d]), 0)
            dt_s = (t5[-1] - t5[-1 - gap]) / 1e6 if gap else 0.1
            v = (now[k] - h_snaps[-1 - gap][k]) / max(dt_s, 1e-4) if gap else np.zeros(2)
            return v, gap, cnt / 5.0

        ego_v, ego_gap, rho_e = _vel(ego)
        pool = []
        for k_nb, pos_nb in now.items():
            if k_nb == ego:
                continue
            dist = float(np.hypot(*(pos_nb - now[ego])))
            if 1e-5 < dist <= 10.0:
                pool.append((dist, k_nb))
        pool.sort()
        for j, (dist, k_nb) in enumerate(pool[:12]):
            v_nb, gap_nb, rho_j = _vel(k_nb)
            dp = now[k_nb] - now[ego]
            ok = bool(ego_gap and gap_nb)
            dv = v_nb - ego_v if ok else np.zeros(2)
            rad = -float(dp @ dv) / (dist + 1e-6) if ok else 0.0
            edges[idx, j] = [dp[0], dp[1], dv[0], dv[1], dist, rad, float(ok), rho_j, rho_e]
            cars[idx, j] = 1.0 if k_nb[0] == "Car" else 0.0
            masks[idx, j] = 1.0
    return {
        "cohort": cohort,
        "history": hist_rel,
        "truth": truth_rel,
        "ts_us": ts_us,
        "seq": np.asarray([w.seq for w in windows], dtype=str),
        "actors": np.asarray([f"{w.seq}:{w.track}" for w in windows], dtype=str),
        "edges": edges,
        "is_car": cars,
        "nbr_mask": masks,
    }


TRANSFORMS = (
    ("0deg", np.array([[1.0, 0.0], [0.0, 1.0]]), False),
    ("90deg", np.array([[0.0, -1.0], [1.0, 0.0]]), False),
    ("180deg", np.array([[-1.0, 0.0], [0.0, -1.0]]), False),
    ("270deg", np.array([[0.0, 1.0], [-1.0, 0.0]]), False),
    ("mirror_y", np.array([[1.0, 0.0], [0.0, -1.0]]), True),
)


@torch.inference_mode()
def score_imptc_ensemble(imptc: dict, ensemble: list[tuple[O2UnifiedFullForecaster, float, float]]) -> dict:
    ts = imptc["ts_us"]
    obs_dt = torch.from_numpy(((ts[:, 1:5] - ts[:, 0:4]) / 1e6).astype(np.float64))
    fut_s = torch.from_numpy(((ts[:, 5:35] - ts[:, 4:5]) / 1e6).astype(np.float64))
    truth = imptc["truth"]
    n = len(truth)
    zero_bio = torch.zeros((n, 5, 11), dtype=torch.float32)
    zero_q = torch.zeros((n,), dtype=torch.float32)
    zero_e = torch.zeros((n, 12, 9), dtype=torch.float32)
    zero_c = torch.zeros((n, 12), dtype=torch.float32)
    zero_m = torch.zeros((n, 12), dtype=torch.float32)

    # Verify exact blackout-pose + empty-graph identity against explicit zero-scale call
    h0 = torch.from_numpy(imptc["history"])
    net0, bs0, gs0 = ensemble[0]
    p_a = net0(h0, zero_bio, zero_q, zero_e, zero_c, zero_m, bio_scale=bs0, graph_scale=gs0, use_action_bridge=True, obs_dt=obs_dt, fut_times_s=fut_s)["paths"]
    p_b = net0(h0, bio_scale=0.0, graph_scale=0.0, use_action_bridge=True, obs_dt=obs_dt, fut_times_s=fut_s)["paths"]
    max_blackout_empty_diff = float((p_a - p_b).abs().max().item())

    by_tf: dict[str, dict] = {}
    base_errs: dict[str, np.ndarray] = {}
    for tf_name, op, _ in TRANSFORMS:
        hg = torch.from_numpy(imptc["history"] @ op.T)
        eg = imptc["edges"].copy()
        eg[:, :, 0:2] = (eg[:, :, 0:2].astype(np.float64) @ op.T).astype(np.float32)
        eg[:, :, 2:4] = (eg[:, :, 2:4].astype(np.float64) @ op.T).astype(np.float32)
        eg_t = torch.from_numpy(eg)
        car_t, msk_t = torch.from_numpy(imptc["is_car"]), torch.from_numpy(imptc["nbr_mask"])

        v_meas = (imptc["history"][:, -1:] - imptc["history"][:, -2:-1]) / ((ts[:, 4:5] - ts[:, 3:4]) / 1e6)[:, :, None]
        cv_m = v_meas * ((ts[:, 5:35] - ts[:, 4:5]) / 1e6)[:, :, None]
        d_cv = np.linalg.norm(cv_m - truth, axis=-1)
        e_cv = np.stack((d_cv.mean(1), d_cv[:, -1], d_cv[:, -1]), axis=-1)

        arms = {k: [] for k in ("u0_clk_bayes", "full_nom_top1", "full_clk_top1", "full_clk_bayes", "full_graph_clk_bayes")}
        for net, bs, gs in ensemble:
            for tag, e_in, g_in, dt_in, ft_in, mode in (
                ("u0_clk_bayes", None, 0.0, obs_dt, fut_s, "bayes"),
                ("full_nom_top1", eg_t, gs, None, None, "top1"),
                ("full_clk_top1", eg_t, gs, obs_dt, fut_s, "top1"),
                ("full_clk_bayes", None, 0.0, obs_dt, fut_s, "full_ng"),
                ("full_graph_clk_bayes", eg_t, gs, obs_dt, fut_s, "bayes"),
            ):
                use_br = tag != "u0_clk_bayes"
                oo = net(hg, zero_bio, zero_q, e_in, car_t if e_in is not None else None, msk_t if e_in is not None else None, bio_scale=bs, graph_scale=g_in, use_action_bridge=use_br, obs_dt=dt_in, fut_times_s=ft_in)
                pw = oo["paths"].numpy() @ op
                pr = oo["logits"].softmax(1).numpy()
                pred = pw[np.arange(n), pr.argmax(1)] if mode == "top1" else np.einsum("nk,nktw->ntw", pr, pw)
                d = np.linalg.norm(pred - truth, axis=-1)
                dm = np.linalg.norm(pw - truth[:, None], axis=-1)
                arms[tag].append(np.stack((d.mean(1), d[:, -1], dm[:, :, -1].min(1)), axis=-1))

        cur = {"actual_time_cv": e_cv, **{k: np.mean(v, axis=0) for k, v in arms.items()}}
        if tf_name == "0deg":
            base_errs = cur
        by_tf[tf_name] = {
            m: {
                "seq_equal_ade_m": float(np.mean([e[imptc["seq"] == s, 0].mean() for s in SEQS])),
                "seq_equal_fde_m": float(np.mean([e[imptc["seq"] == s, 1].mean() for s in SEQS])),
                "pooled_ade_m": float(e[:, 0].mean()),
                "pooled_fde_m": float(e[:, 1].mean()),
                "pooled_min_fde6_m": float(e[:, 2].mean()),
            }
            for m, e in cur.items()
        }
    swings = {
        m: float(max(by_tf[t][m]["pooled_ade_m"] for t, _, _ in TRANSFORMS) - min(by_tf[t][m]["pooled_ade_m"] for t, _, _ in TRANSFORMS))
        for m in ("full_clk_bayes", "full_graph_clk_bayes")
    }
    return {
        "n_windows": n,
        "windows_with_10m_neighbors": int((imptc["nbr_mask"].sum(1) > 0).sum()),
        "windows_with_10m_vehicles": int((imptc["is_car"].sum(1) > 0).sum()),
        "max_blackout_empty_algebraic_diff_m": max_blackout_empty_diff,
        "five_axis_max_ade_swing_m": swings,
        "archived_m0_reference_seq_equal": {"ade_m": 0.3944, "fde_m": 0.8657},
        "by_transform": by_tf,
        "by_sequence_0deg": {
            s: {m: {"ade_m": float(e[imptc["seq"] == s, 0].mean()), "fde_m": float(e[imptc["seq"] == s, 1].mean())} for m, e in base_errs.items()}
            for s in SEQS
        },
        "paired_person_bootstrap_ci_0deg": {
            "full_graph_clk_minus_cv": person_bootstrap_ci(base_errs["full_graph_clk_bayes"][:, :2] - base_errs["actual_time_cv"][:, :2], np.ones(n, dtype=bool), imptc["actors"]),
            "full_graph_minus_no_graph": person_bootstrap_ci(base_errs["full_graph_clk_bayes"][:, :2] - base_errs["full_clk_bayes"][:, :2], np.ones(n, dtype=bool), imptc["actors"]),
        },
    }


@torch.inference_mode()
def score_dair_unified(dair: dict, ens_t5: list, ens_t2: list) -> dict:
    ts = dair["ts_us"]
    obs_dt = torch.from_numpy(((ts[:, 4:5] - ts[:, 3:4]) / 1e6).astype(np.float64))
    fut_s = torch.from_numpy(((ts[:, 5:] - ts[:, 4:5]) / 1e6).astype(np.float64))
    truth = dair["truth"]
    n = len(truth)
    strict_m = np.max(np.abs(np.diff(ts[:, :5], axis=1) - 100000), axis=1) <= 1000
    by_tf: dict[str, dict] = {}
    base_e: dict[str, np.ndarray] = {}
    for r_i, (tf_name, op, refl) in enumerate(TRANSFORMS):
        hg = dair["hist2"] @ op.T
        hg_t = torch.from_numpy(hg)
        kp = dair["kpts"].copy()
        if refl:
            kp[..., 0] = 1920.0 - kp[..., 0]
            kp = kp[:, :, COCO_SWAP, :]
        bs_t, qb_t = biomech_sequence_torch(torch.from_numpy(kp), torch.from_numpy(dair["bbox"]))
        vm = (hg[:, -1:] - hg[:, -2:-1]) / ((ts[:, 4:5] - ts[:, 3:4]) / 1e6)[:, :, None]
        cvm = (vm * ((ts[:, 5:] - ts[:, 4:5]) / 1e6)[:, :, None]) @ op
        dc = np.linalg.norm(cvm - truth, axis=-1)
        ecv = np.stack((dc.mean(1), dc[:, -1], dc[:, -1]), axis=-1)

        arms = {k: [] for k in ("ours_o2_full_t5_clk_bayes", "ours_o2_full_t2_clk_top1", "ours_o2_full_t2_clk_bayes")}
        for tag, ens, mode in (
            ("ours_o2_full_t5_clk_bayes", ens_t5, "bayes"),
            ("ours_o2_full_t2_clk_top1", ens_t2, "top1"),
            ("ours_o2_full_t2_clk_bayes", ens_t2, "bayes"),
        ):
            for net, bsc, _ in ens:
                oo = net(hg_t, bs_t, qb_t, bio_scale=bsc, graph_scale=0.0, use_action_bridge=True, obs_dt=obs_dt, fut_times_s=fut_s)
                pw = oo["paths"].numpy() @ op
                pr = oo["logits"].softmax(1).numpy()
                pred = pw[np.arange(n), pr.argmax(1)] if mode == "top1" else np.einsum("nk,nktw->ntw", pr, pw)
                d = np.linalg.norm(pred - truth, axis=-1)
                dm = np.linalg.norm(pw - truth[:, None], axis=-1)
                arms[tag].append(np.stack((d.mean(1), d[:, -1], dm[:, :, -1].min(1)), axis=-1))

        cur = {"cv_measured": ecv, **{k: np.mean(v, axis=0) for k, v in arms.items()}}
        if r_i < 4:
            cur["archived_M0"] = np.concatenate([dair["m0_by_rot"][:, r_i, :], dair["m0_by_rot"][:, r_i, 1:2]], axis=-1)
            cur["archived_M1"] = np.concatenate([dair["m1_by_rot"][:, r_i, :], dair["m1_by_rot"][:, r_i, 1:2]], axis=-1)
        if tf_name == "0deg":
            base_e = cur
        by_tf[tf_name] = {
            "all_54": {m: {"ade_m": float(e[:, 0].mean()), "fde_m": float(e[:, 1].mean()), "min_fde6_m": float(e[:, 2].mean())} for m, e in cur.items()},
            "strict_31": {m: {"ade_m": float(e[strict_m, 0].mean()), "fde_m": float(e[strict_m, 1].mean()), "min_fde6_m": float(e[strict_m, 2].mean())} for m, e in cur.items()},
        }
    rot4_swing = float(max(by_tf[t]["all_54"]["ours_o2_full_t2_clk_bayes"]["ade_m"] for t, _, _ in TRANSFORMS[:4]) - min(by_tf[t]["all_54"]["ours_o2_full_t2_clk_bayes"]["ade_m"] for t, _, _ in TRANSFORMS[:4]))
    five_swing = float(max(by_tf[t]["all_54"]["ours_o2_full_t2_clk_bayes"]["ade_m"] for t, _, _ in TRANSFORMS) - min(by_tf[t]["all_54"]["ours_o2_full_t2_clk_bayes"]["ade_m"] for t, _, _ in TRANSFORMS))
    return {
        "n_all": n,
        "n_strict_1ms": int(strict_m.sum()),
        "rot4_ade_swing_m": rot4_swing,
        "five_axis_ade_swing_m": five_swing,
        "by_transform": by_tf,
        "paired_person_bootstrap_ci_0deg": {
            "all54_t2_bayes_minus_cv": person_bootstrap_ci(base_e["ours_o2_full_t2_clk_bayes"][:, :2] - base_e["cv_measured"][:, :2], np.ones(n, dtype=bool), dair["actors"]),
            "all54_t2_bayes_minus_m1": person_bootstrap_ci(base_e["ours_o2_full_t2_clk_bayes"][:, :2] - base_e["archived_M1"][:, :2], np.ones(n, dtype=bool), dair["actors"]),
            "strict31_t2_bayes_minus_cv": person_bootstrap_ci(base_e["ours_o2_full_t2_clk_bayes"][:, :2] - base_e["cv_measured"][:, :2], strict_m, dair["actors"]),
            "strict31_t2_bayes_minus_m1": person_bootstrap_ci(base_e["ours_o2_full_t2_clk_bayes"][:, :2] - base_e["archived_M1"][:, :2], strict_m, dair["actors"]),
        },
    }


def run_external_suite(meta_dir: Path, pose_npz: Path, joint_zip: Path, dair_zip: Path, out_dir: Path, imptc_tar: Path | None = None, seeds: tuple[int, ...] = SEEDS) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_p = ensure_imptc_sample(imptc_tar or (out_dir / "scratch_imptc/imptc_samples.tar.gz"))
    imptc = load_imptc_with_neighbors(tar_p)
    dair, _ = load_attached_dair_54(dair_zip)
    data5 = load_unified_cohort(meta_dir, pose_npz, joint_zip)
    data2 = {**data5, "history": data5["history"][:, -2:]}

    ens_t5, ens_t2 = [], []
    for seed in seeds:
        for s in SITES:
            tr_ids = np.flatnonzero(data5["site"] != s)
            bs5, gs5 = select_unified_scales(seed, data5, tr_ids, device)
            n5 = fit_unified_fold(seed, data5, tr_ids, device).cpu().eval()
            ens_t5.append((n5, bs5, gs5))
            bs2, gs2 = select_unified_scales(seed + 101, data2, tr_ids, device)
            n2 = fit_unified_fold(seed + 101, data2, tr_ids, device).cpu().eval()
            ens_t2.append((n2, bs2, gs2))

    imptc_res = score_imptc_ensemble(imptc, ens_t5)
    dair_res = score_dair_unified(dair, ens_t5, ens_t2)
    agg = {
        "study": "Zero-shot 5-seed unified Ours_O2_Full on German IMPTC (564) & Beijing DAIR-V2X (54)",
        "seeds": list(seeds),
        "imptc_564": imptc_res,
        "dair_v2x_54": dair_res,
    }
    agg_p, md_p, man_p = out_dir / "report.json", out_dir / "RESULTS.md", out_dir / "sha256.json"
    agg_p.write_text(json.dumps(agg, indent=2) + "\n")
    im0 = imptc_res["by_transform"]["0deg"]
    da0 = dair_res["by_transform"]["0deg"]
    md_p.write_text(
        f"# Zero-Shot Unified `Ours_O2_Full` on IMPTC (564) & DAIR-V2X (54)\n\n"
        f"- **IMPTC (564 windows, seq-equal)**: `Ours_O2_Full (clock+graph)` `{im0['full_graph_clk_bayes']['seq_equal_ade_m']:.4f} / {im0['full_graph_clk_bayes']['seq_equal_fde_m']:.4f} m` "
        f"vs archived `M0` `0.3944 / 0.8657 m` and `CV` `{im0['actual_time_cv']['seq_equal_ade_m']:.4f} / {im0['actual_time_cv']['seq_equal_fde_m']:.4f} m`.\n"
        f"- **DAIR-V2X (54 windows / 31 strict)**: `Ours_O2_Full` `{da0['all_54']['ours_o2_full_t2_clk_bayes']['ade_m']:.4f}` (all 54) / `{da0['strict_31']['ours_o2_full_t2_clk_bayes']['ade_m']:.4f} m` (strict 31).\n"
    )
    mani = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (agg_p, md_p)}
    man_p.write_text(json.dumps(mani, indent=2) + "\n")
    with zipfile.ZipFile(out_dir / "imptc_and_dair_o2_full_5seed.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in (agg_p, md_p, man_p):
            zf.write(p, arcname=p.name)
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--dair-zip", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    run_external_suite(args.meta_dir, args.pose_npz, args.joint_zip, args.dair_zip, args.out_dir)
