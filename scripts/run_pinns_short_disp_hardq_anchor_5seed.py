"""Wave 15 GPU2: Definitive <0.5m short-displacement test — inner-LOSO CV-anchored O(2) motion vs hard-quality-gated ground pose (5 seeds)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from scripts.benchmark_heldout_sites import load_site_windows
from scripts.ppt_mainline_o2_models import build_ground_pose_o2_inputs, local_frenet_with_pose
from scripts.run_emergence_and_action_sensitivity import build_emergence_masks
from scripts.run_pinns_bioeq_benchmark import prepare_cohort
from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_pinns_matched_o2_multiduration import SITES, paired_site_clip_ci
from scripts.run_pinns_o2_joint_actions_5seed import SEEDS
from src.data.features import assign_quality_bucket
from src.data.pinns_extractor import _check_right_handed_homography
from src.models.bioeq_o2_joint import o2_joint_loss


class LowDispAnchoredO2Net(nn.Module):
    """O(2) forecaster with low-displacement CV anchor (d_ref) and zero-init Clear/Partial pose residual."""

    def __init__(self) -> None:
        super().__init__()
        self.motion = nn.GRU(5, 48, batch_first=True)
        self.ctx_proj = nn.Sequential(nn.LayerNorm(48), nn.Linear(48, 48), nn.SiLU())
        self.central_dx = nn.Linear(48, 3 * 3)
        self.central_dy = nn.Linear(48, 3 * 3, bias=False)
        self.pair_knots = nn.Linear(48, 3 * 2)
        self.central_scores = nn.Linear(48, 4)
        self.pair_score = nn.Linear(48, 1)
        for m in (self.central_dx, self.pair_knots):
            nn.init.normal_(m.weight, std=0.01); nn.init.zeros_(m.bias)
        nn.init.normal_(self.central_dy.weight, std=0.01)

        self.pose_in = nn.Linear(16, 24)
        self.pose_gru = nn.GRU(24, 24, batch_first=True)
        self.q_gate = nn.Sequential(nn.Linear(48 + 24 + 4, 24), nn.SiLU(), nn.Linear(24, 1))
        self.pose_dx = nn.Linear(24, 3 * 3, bias=False)
        self.pose_dy = nn.Linear(24, 3 * 3, bias=False)
        self.pose_pair = nn.Linear(24, 3 * 2, bias=False)
        for m in (self.pose_dx, self.pose_dy, self.pose_pair):
            nn.init.zeros_(m.weight)

    def forward(
        self,
        history: torch.Tensor,
        bio_even: torch.Tensor | None = None,
        pose_vec_world: torch.Tensor | None = None,
        signed_knee: torch.Tensor | None = None,
        q_vec: torch.Tensor | None = None,
        hard_q_mask: torch.Tensor | None = None,
        d_ref: float = 0.0,
        pose_scale: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        bsz = len(history)
        use_p = (pose_scale > 0.0) and (hard_q_mask is not None) and (pose_vec_world is not None)
        pvw = (pose_vec_world * hard_q_mask[:, None, None, None].to(pose_vec_world.dtype)) if use_p else None
        origin, basis, valid = local_frenet_with_pose(history, pvw, use_pose_backup=use_p)
        body = (history - origin) @ basis
        vel = torch.cat((torch.zeros_like(body[:, :1]), (body[:, 1:] - body[:, :-1]) / 0.1), dim=1)
        fp = torch.cat((body, vel, torch.linalg.vector_norm(vel, dim=2, keepdim=True)), dim=2).float()
        fn = fp * fp.new_tensor([1.0, -1.0, 1.0, -1.0, 1.0])
        _, hm = self.motion(torch.cat((fp, fn), dim=0))
        cb = self.ctx_proj(hm[0])
        cp, cn = cb[:bsz], cb[bsz:]
        ce, co = 0.5 * (cp + cn), 0.5 * (cp - cn)

        disp5 = torch.linalg.vector_norm(history[:, 1:] - history[:, :-1], dim=-1).sum(dim=-1).float()
        m_scale = (disp5 / max(d_ref, 1e-6)).clamp(0.0, 1.0)[:, None, None] if d_ref > 0.0 else disp5.new_ones((bsz, 1, 1))
        c_dx = m_scale * self.central_dx(ce).view(bsz, 3, 3)
        c_dy = m_scale * self.central_dy(co).view(bsz, 3, 3)
        p_pos = m_scale * self.pair_knots(cp).view(bsz, 3, 2)
        p_neg = m_scale * self.pair_knots(cn).view(bsz, 3, 2)

        gate = disp5.new_zeros(bsz)
        if use_p and bio_even is not None and signed_knee is not None and q_vec is not None:
            pvl = torch.einsum("btvk,bks->btvs", pvw.double(), basis).float()
            fc, fl = pvl[:, :, 0], pvl[:, :, 1]
            flp = fc.new_tensor([1.0, -1.0])
            pp = torch.cat((bio_even, fc, fl, signed_knee), dim=-1)
            pn = torch.cat((bio_even, fc * flp, fl * flp, -signed_knee), dim=-1)
            _, hpb = self.pose_gru(F.silu(self.pose_in(torch.cat((pp, pn), dim=0))))
            hpp, hpn = hpb[0][:bsz], hpb[0][bsz:]
            hpe, hpo = 0.5 * (hpp + hpn), 0.5 * (hpp - hpn)
            gate = float(pose_scale) * hard_q_mask.float() * torch.sigmoid(self.q_gate(torch.cat((ce, hpe, q_vec), dim=-1)).squeeze(-1)) * torch.exp(-((disp5 / 0.60) ** 2))
            c_dx = c_dx + gate[:, None, None] * self.pose_dx(hpe).view(bsz, 3, 3)
            c_dy = c_dy + gate[:, None, None] * self.pose_dy(hpo).view(bsz, 3, 3)
            p_pos = p_pos + gate[:, None, None] * self.pose_pair(hpp).view(bsz, 3, 2)
            p_neg = p_neg + gate[:, None, None] * self.pose_pair(hpn).view(bsz, 3, 2)

        ck = torch.stack((c_dx, c_dy), dim=-1)
        m4 = torch.stack((p_neg[..., 0], -p_neg[..., 1]), dim=-1)[:, None]
        knots5 = torch.cat((ck[:, :2], p_pos[:, None], m4, ck[:, 2:]), dim=1).double()
        knots5 = torch.cat((torch.zeros_like(knots5[:, :, :1]), knots5), dim=2)
        csc = self.central_scores(ce)
        cv_boost = (3.0 * (1.0 - (disp5 / max(d_ref, 1e-6)).clamp(0.0, 1.0)) * (1.0 - gate))[:, None] if d_ref > 0.0 else 0.0
        lg = torch.cat((csc[:, :1] + cv_boost, csc[:, 1:3], self.pair_score(cp), self.pair_score(cn), csc[:, 3:]), dim=1)
        logits = torch.where(valid[:, None], lg, 0.0)

        t_sec = torch.arange(1, 31, dtype=history.dtype, device=history.device)[None, :] * 0.1
        idx = (t_sec - 1e-9).floor().long().clamp(0, 2)
        ph = (t_sec - idx.to(history.dtype))[:, None, :, None]
        bi, mi = torch.arange(bsz, device=history.device)[:, None, None], torch.arange(5, device=history.device)[None, :, None]
        delta = knots5[bi, mi, idx[:, None, :]] * (1.0 - ph) + knots5[bi, mi, (idx + 1)[:, None, :]] * ph
        world = torch.einsum("bkts,bws->bktw", delta, basis)
        cv = origin + ((history[:, -1:] - history[:, -2:-1]) / 0.1) * t_sec[:, :, None]
        return {"paths": torch.cat((cv[:, None], cv[:, None] + world), dim=1), "logits": logits, "valid": valid, "gate": gate}
def _fit_two_stage(seed: int, data: dict, tr_ids: np.ndarray, q_mask_key: str, device: torch.device) -> LowDispAnchoredO2Net:
    torch.manual_seed(seed)
    net = LowDispAnchoredO2Net().to(device)
    w_tr = make_site_weights(data["site"], tr_ids).to(device)
    tr_t = torch.from_numpy(tr_ids)
    truth_tr = data["truth"][tr_ids].to(device)

    m0_pref = ("motion.", "ctx_proj.", "central_", "pair_")
    for nm, p in net.named_parameters():
        p.requires_grad = nm.startswith(m0_pref)
    opt1 = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=2e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(45):
        for ch in torch.randperm(len(tr_ids), generator=gen).split(64):
            id_b = tr_t[ch]
            loss = o2_joint_loss(net(data["history"][id_b].to(device), d_ref=0.35, pose_scale=0.0), truth_tr[ch], w_tr[ch], act_weight=0.0)
            if loss is not None:
                opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()

    # Stage 2: STRICTLY freeze M0 backbone & scores; train zero-init pose knot residual ONLY on eligible quality windows
    p_pref = ("pose_", "q_gate.")
    for nm, p in net.named_parameters():
        p.requires_grad = nm.startswith(p_pref)
    opt2 = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1.2e-3, weight_decay=3e-4)
    for _ in range(25):
        for ch in torch.randperm(len(tr_ids), generator=gen).split(64):
            id_b = tr_t[ch]
            out = net(
                data["history"][id_b].to(device), data["bio_even"][id_b].to(device), data["pose_vec_world"][id_b].to(device),
                data["signed_knee"][id_b].to(device), data["quality"][id_b].to(device), data[q_mask_key][id_b].to(device),
                d_ref=0.35, pose_scale=1.0,
            )
            loss = o2_joint_loss(out, truth_tr[ch], w_tr[ch], act_weight=0.0, reg_only=True)
            if loss is not None:
                opt2.zero_grad(set_to_none=True); loss.backward(); opt2.step()
    return net.eval()


@torch.inference_mode()
def _eval_rows(net: LowDispAnchoredO2Net, data: dict, ids: np.ndarray, q_mask_key: str, d_ref: float, p_sc: float, device: torch.device) -> np.ndarray:
    tid = torch.from_numpy(ids)
    out = net(
        data["history"][tid].to(device), data["bio_even"][tid].to(device), data["pose_vec_world"][tid].to(device),
        data["signed_knee"][tid].to(device), data["quality"][tid].to(device), data[q_mask_key][tid].to(device),
        d_ref=d_ref, pose_scale=p_sc,
    )
    p = out["paths"].cpu().numpy()
    q = out["logits"].softmax(dim=1).cpu().numpy()
    tr = data["truth"][ids].numpy()
    dst = np.linalg.norm(p - tr[:, None], axis=-1)
    ade_k, fde_k = dst.mean(2), dst[:, :, -1]
    top = q.argmax(1); ii = np.arange(len(ids))
    bay = np.einsum("nk,nktw->ntw", q, p)
    bd = np.linalg.norm(bay - tr, axis=-1)
    return np.column_stack((ade_k[ii, top], fde_k[ii, top], ade_k.min(1), fde_k.min(1), bd.mean(1), bd[:, -1], out["gate"].cpu().numpy()))


def _select_inner_scales(seed: int, data: dict, tr_ids: np.ndarray, device: torch.device) -> tuple[float, float, float]:
    src = sorted(set(data["site"][tr_ids].tolist()))
    d_grid = (0.0, 0.25, 0.40, 0.55, 0.70)
    p_grid = (0.0, 0.25, 0.50, 0.75, 1.0)
    d_err = np.zeros(len(d_grid))
    inner_c = []
    inner_cp = []
    for vs in src:
        fi, vi = tr_ids[data["site"][tr_ids] != vs], tr_ids[data["site"][tr_ids] == vs]
        nc = _fit_two_stage(seed, data, fi, "qmask_clear", device)
        ncp = _fit_two_stage(seed, data, fi, "qmask_clear_partial", device)
        inner_c.append((nc, vi)); inner_cp.append((ncp, vi))
        for i, dr in enumerate(d_grid):
            d_err[i] += float(_eval_rows(nc, data, vi, "qmask_clear", dr, 0.0, device)[:, 0].mean())
    best_d = float(d_grid[int(d_err.argmin())])
    pc_err = np.zeros(len(p_grid)); pcp_err = np.zeros(len(p_grid))
    for (nc, vi), (ncp, _) in zip(inner_c, inner_cp):
        for i, ps in enumerate(p_grid):
            pc_err[i] += float(_eval_rows(nc, data, vi, "qmask_clear", best_d, ps, device)[:, 0].mean())
            pcp_err[i] += float(_eval_rows(ncp, data, vi, "qmask_clear_partial", best_d, ps, device)[:, 0].mean())
    return best_d, float(p_grid[int(pc_err.argmin())]), float(p_grid[int(pcp_err.argmin())])


def load_short_disp_cohort(meta_dir: Path, pose_npz: Path) -> dict:
    tensors, idx, keys, _ = prepare_cohort(meta_dir, pose_npz)
    groups, _, _ = load_site_windows(meta_dir)
    rows = [r for s in sorted(groups) for r in groups[s]]
    clips = np.array([r.clip for r in rows], dtype=str)
    site_arr = np.empty(len(keys), dtype="<U24")
    for s, ids in idx.items():
        site_arr[ids] = s
    with np.load(pose_npz, allow_pickle=False) as pz:
        ck = list(zip(pz["clip"].tolist(), pz["person_id"].tolist(), pz["t0"].tolist()))
        om = {k: i for i, k in enumerate(ck)}
        ord_i = [om[k] for k in keys]
        kpts = torch.from_numpy(pz["hist_kpts"][ord_i].copy())
        bbox = torch.from_numpy(pz["hist_bbox"][ord_i].copy())

    n = len(rows)
    hm_arr, ys_arr = np.zeros((n, 3, 3), dtype=np.float64), np.zeros(n, dtype=np.float64)
    for c in sorted(set(clips.tolist())):
        cal = json.loads((meta_dir / "videos" / c.rsplit("_", 1)[0] / "calib.json").read_text())
        hm = _check_right_handed_homography(cal); ys = float(cal["y_min"] + cal["y_max"])
        for i in np.flatnonzero(clips == c):
            hm_arr[i] = hm; ys_arr[i] = ys
    gp = build_ground_pose_o2_inputs(tensors["history"], kpts, bbox, torch.from_numpy(hm_arr), torch.from_numpy(ys_arr))
    q_np = tensors["quality"].numpy()
    qb = np.array([assign_quality_bucket(q_np[i]) for i in range(n)], dtype=int)
    emg = build_emergence_masks(meta_dir, {"site": site_arr, "history": tensors["history"]})
    return {
        **tensors, **gp, "site": site_arr, "clip": clips, "q_buckets": qb,
        "qmask_clear": torch.from_numpy((qb == 0).astype(np.float32)),
        "qmask_clear_partial": torch.from_numpy((qb <= 1).astype(np.float32)),
        "strata": {
            **emg, "clear_pose": qb == 0, "partial_pose": qb == 1, "heavy_pose": qb == 2,
            "first_short_clear": emg["first_short_323"] & (qb == 0),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("meta_dir", type=Path)
    ap.add_argument("pose_npz", type=Path)
    ap.add_argument("output_zip", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_short_disp_cohort(args.meta_dir, args.pose_npz)
    n = len(data["site"]); site_arr, clip_arr = data["site"], data["clip"]
    h_np, t_np = data["history"].numpy(), data["truth"].numpy()
    cv_p = h_np[:, -1:] + (h_np[:, -1:] - h_np[:, -2:-1]) * np.arange(1, 31, dtype=np.float64)[None, :, None]
    d_cv = np.linalg.norm(cv_p - t_np, axis=-1)
    cv_m = np.column_stack((d_cv.mean(1), d_cv[:, -1], d_cv.mean(1), d_cv[:, -1], d_cv.mean(1), d_cv[:, -1], np.zeros(n)))

    arms = ("M0_unanchored", "M0_anchor", "M1_hardq_clear", "M1_hardq_clear_partial")
    acc = {a: np.zeros((len(args.seeds), n, 7), dtype=np.float64) for a in arms}
    chosen_scales = []
    for si, seed in enumerate(args.seeds):
        for s in SITES:
            tr_ids, te_ids = np.flatnonzero(site_arr != s), np.flatnonzero(site_arr == s)
            d_ref, pc, pcp = _select_inner_scales(seed, data, tr_ids, dev)
            chosen_scales.append({"seed": seed, "heldout_site": s, "d_ref": d_ref, "pose_scale_clear": pc, "pose_scale_cp": pcp})
            nc = _fit_two_stage(seed, data, tr_ids, "qmask_clear", dev)
            ncp = _fit_two_stage(seed, data, tr_ids, "qmask_clear_partial", dev)
            acc["M0_unanchored"][si, te_ids] = _eval_rows(nc, data, te_ids, "qmask_clear", 0.0, 0.0, dev)
            acc["M0_anchor"][si, te_ids] = _eval_rows(nc, data, te_ids, "qmask_clear", d_ref, 0.0, dev)
            acc["M1_hardq_clear"][si, te_ids] = _eval_rows(nc, data, te_ids, "qmask_clear", d_ref, pc, dev)
            acc["M1_hardq_clear_partial"][si, te_ids] = _eval_rows(ncp, data, te_ids, "qmask_clear_partial", d_ref, pcp, dev)
        print(f"Finished short-disp hardq anchor seed={seed}", flush=True)

    mm = {"CV": cv_m, **{a: acc[a].mean(0) for a in arms}}
    cols = ("ade1_m", "fde1_m", "min_ade6_m", "min_fde6_m", "bayes_ade_m", "bayes_fde_m", "mean_gate")
    contrasts = (("M0_anchor_minus_CV", "M0_anchor", "CV"), ("M1_clear_minus_CV", "M1_hardq_clear", "CV"), ("M1_clear_minus_M0_anchor", "M1_hardq_clear", "M0_anchor"), ("M1_cp_minus_M0_anchor", "M1_hardq_clear_partial", "M0_anchor"), ("M0_anchor_minus_M0_unanchored", "M0_anchor", "M0_unanchored"))
    st_out = {}
    for sn, sm in data["strata"].items():
        if not sm.any():
            continue
        ss, sc = site_arr[sm], clip_arr[sm]
        md = {}
        for m, arr in mm.items():
            sub = arr[sm]
            bys = {s: {c: float(sub[ss == s, j].mean()) for j, c in enumerate(cols)} for s in SITES if (ss == s).any()}
            md[m] = {"site_equal": {c: float(np.mean([v[c] for v in bys.values()])) for c in cols}, "pooled": {c: float(sub[:, j].mean()) for j, c in enumerate(cols)}}
        st_out[sn] = {"n_windows": int(sm.sum()), "models": md, "paired_top1_clip_ci": {cn: paired_site_clip_ci(mm[ma][sm, :2] - mm[mb][sm, :2], ss, sc) for cn, ma, mb in contrasts}}

    report = {"experiment": "pinns_short_disp_hardq_anchor_5seed", "n_windows": n, "seeds": list(args.seeds), "inner_loso_selected": chosen_scales, "strata": st_out}
    raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = json.dumps({"pinns_short_disp_hardq_anchor_5seed.json": hashlib.sha256(raw).hexdigest()}, indent=2).encode("utf-8")
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("pinns_short_disp_hardq_anchor_5seed.json", raw)
        zf.writestr("sha256.json", sh)
    print("Saved:", args.output_zip, flush=True)


if __name__ == "__main__":
    main()
