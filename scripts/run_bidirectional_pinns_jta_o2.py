"""Five-seed bidirectional zero-shot cross-dataset transfer: PINNS (10Hz Real) <-> JTA (2.5Hz Sim)."""
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
from scripts.run_jta_o2_bioeq_5seed import SEEDS, build_jta_tensors, eval_jta_dict, fit_jta_models, summarize_jta
from scripts.run_o2_hetero_graph_pinns_and_dut import O2NullSinkGraphForecaster
from scripts.run_pinns_bioeq_benchmark import prepare_cohort
from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_pinns_matched_o2_multiduration import SITES
from src.data.jta_builder import process_jta_root
from src.models.bioeq_o2_joint import O2BioEqJointForecaster, o2_joint_loss


def fit_pinns_all_sites(seed: int, tensors: dict, site_arr: np.ndarray) -> tuple[O2NullSinkGraphForecaster, float]:
    all_ids = np.arange(len(site_arr))
    torch.manual_seed(seed)
    base = O2BioEqJointForecaster(pose_mode="bio_pose", action_mode="none")
    for n, p in base.named_parameters():
        p.requires_grad = not (n.startswith("bio_") or n == "gate_logit")
    w_all = make_site_weights(site_arr, all_ids)
    hist, truth = tensors["history"], tensors["truth"]
    bseq, qbio = tensors["bio_seq"], tensors["q_bio"]
    gen = torch.Generator().manual_seed(seed)
    opt1 = torch.optim.AdamW([p for p in base.parameters() if p.requires_grad], lr=2e-3, weight_decay=1e-4)
    for _ in range(45):
        for ch in torch.randperm(len(all_ids), generator=gen).split(64):
            loss = o2_joint_loss(base(hist[ch]), truth[ch], w_all[ch], act_weight=0.0)
            if loss is not None:
                opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()
    for n, p in base.named_parameters():
        p.requires_grad = n.startswith("bio_") or n == "gate_logit"
    opt2 = torch.optim.AdamW([p for p in base.parameters() if p.requires_grad], lr=1.5e-3, weight_decay=2e-4)
    for _ in range(25):
        for ch in torch.randperm(len(all_ids), generator=gen).split(64):
            loss = o2_joint_loss(base(hist[ch], bseq[ch], qbio[ch]), truth[ch], w_all[ch], act_weight=0.0, reg_only=True)
            if loss is not None:
                opt2.zero_grad(set_to_none=True); loss.backward(); opt2.step()
    for p in base.parameters():
        p.requires_grad = False
    return O2NullSinkGraphForecaster(base).eval(), 0.5


@torch.inference_mode()
def eval_pinns_zero_shot(gn: O2NullSinkGraphForecaster, bsc: float, tensors: dict) -> np.ndarray:
    n = len(tensors["history"])
    edg = torch.zeros((n, 12, 9), dtype=torch.float32)
    car = torch.zeros((n, 12), dtype=torch.float32)
    msk = torch.zeros((n, 12), dtype=torch.float32)
    out = gn(tensors["history"], edg, car, msk, tensors["bio_seq"], tensors["q_bio"] * bsc, graph_scale=0.0)
    p = out["paths"]
    q = out["logits"].softmax(dim=1)
    top = p[torch.arange(n), q.argmax(1)]
    bay = torch.einsum("bk,bktw->btw", q.double(), p)
    dt = torch.linalg.vector_norm(top - tensors["truth"], dim=-1)
    db = torch.linalg.vector_norm(bay - tensors["truth"], dim=-1)
    return torch.stack((dt.mean(1), dt[:, -1], db.mean(1), db[:, -1]), dim=1).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pinns_metadata", type=Path)
    ap.add_argument("pose_npz", type=Path)
    ap.add_argument("output_zip", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    torch.set_num_threads(4)
    tensors_p, idx_p, keys_p, _ = prepare_cohort(args.pinns_metadata, args.pose_npz)
    site_arr = np.empty(len(keys_p), dtype="<U24")
    for s, rows in idx_p.items():
        site_arr[rows] = s

    with tempfile.TemporaryDirectory(prefix="bidir_jta_") as td:
        sc = Path(td)
        st_zip, st_dir = sc / "st_releases.zip", sc / "st_unpacked"
        _download_verified_zip(ST_RELEASE_URL, st_zip)
        _unpack_verified_zip(st_zip, st_dir)
        jta_root = [p for p in st_dir.rglob("data") if p.is_dir() and "jta" in str(p).lower() and "__macosx" not in str(p).lower()][0]
        jta_samples, _ = process_jta_root(jta_root, max_files_per_split=24, seed=42)

    jta_tr5 = build_jta_tensors(jta_samples, "train", 5)
    jta_va5 = build_jta_tensors(jta_samples, "val", 5)
    jta_te2 = build_jta_tensors(jta_samples, "test", 2)
    jta_te5 = build_jta_tensors(jta_samples, "test", 5)

    p2j_t2, p2j_t5, j2p_t5 = [], [], []
    for seed in args.seeds:
        gn_p, bsc_p = fit_pinns_all_sites(seed, tensors_p, site_arr)
        p2j_t2.append(eval_jta_dict(gn_p, jta_te2, bsc_p, 0.0))
        p2j_t5.append(eval_jta_dict(gn_p, jta_te5, bsc_p, 0.0))
        gn_j, bsc_j, _ = fit_jta_models(seed, jta_tr5, jta_va5)
        j2p_t5.append(eval_pinns_zero_shot(gn_j, bsc_j, tensors_p))
        print(f"Done bidirectional PINNS <-> JTA seed={seed}", flush=True)

    j2p_avg = np.mean(j2p_t5, axis=0)
    by_site = {}
    for s in SITES:
        v = j2p_avg[site_arr == s].mean(0)
        by_site[s] = {"ade_m": float(v[0]), "fde_m": float(v[1]), "bayes_ade_m": float(v[2]), "bayes_fde_m": float(v[3])}
    v_eq = np.mean([[by_site[s][k] for k in ("ade_m", "fde_m", "bayes_ade_m", "bayes_fde_m")] for s in SITES], axis=0)

    def clean_sm(sm: dict) -> dict:
        return {k: v for k, v in sm.items() if not k.startswith("_")}

    report = {
        "seeds": list(args.seeds),
        "real2sim_pinns_to_jta_T2_04s": clean_sm(summarize_jta(p2j_t2, jta_te2["bucket"])),
        "real2sim_pinns_to_jta_T5_16s": clean_sm(summarize_jta(p2j_t5, jta_te5["bucket"])),
        "sim2real_jta_to_pinns_T5_04s": {
            "site_equal": {"ade_m": float(v_eq[0]), "fde_m": float(v_eq[1]), "bayes_ade_m": float(v_eq[2]), "bayes_fde_m": float(v_eq[3])},
            "by_site": by_site,
            "pinns_last2_cv_reference": {"ade_m": 0.4838738, "fde_m": 1.1045585},
        },
    }
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    members = {"report.json": (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()}
    members["sha256.json"] = (json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in members.items()}, indent=2, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in members.items():
            zf.writestr(k, v)
    print("Saved", args.output_zip, flush=True)


if __name__ == "__main__":
    main()
