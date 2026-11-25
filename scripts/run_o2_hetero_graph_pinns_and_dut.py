"""Five-seed O(2) parity-equivariant null-sink ped-vehicle graph on PINNS (1,482) & zero-shot DUT (3,066)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_pinns_matched_o2_multiduration import SITES, paired_site_clip_ci
from src.models.bioeq_neural import biomech_sequence_torch, local_frenet_torch
from src.models.bioeq_o2_joint import O2BioEqJointForecaster, o2_joint_loss

SEEDS = (42, 1024, 2025, 3407, 4096)


class O2NullSinkGraphForecaster(nn.Module):
    """Exact O(2)-equivariant forecaster with null-sink heterogeneous neighbor attention."""

    def __init__(self, base: O2BioEqJointForecaster) -> None:
        super().__init__()
        self.base = base
        self.edge_mlp = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 48))
        self.edge_att = nn.Linear(48 + 48, 1)
        self.null_logit = nn.Parameter(torch.tensor(1.5))
        self.out_proj = nn.Linear(48, 48, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def _graph_delta(self, ctx: torch.Tensor, e_feat: torch.Tensor, nbr_mask: torch.Tensor) -> torch.Tensor:
        emb = self.edge_mlp(e_feat)
        q_exp = ctx[:, None, :].expand_as(emb)
        att_n = self.edge_att(torch.cat((q_exp, emb), dim=-1)).squeeze(-1)
        att_n = att_n.masked_fill(nbr_mask < 0.5, -1e9)
        null_col = self.null_logit.expand(len(ctx), 1)
        weights = torch.softmax(torch.cat((att_n, null_col), dim=1), dim=1)[:, :12]
        pooled = (weights[:, :, None] * emb).sum(dim=1)
        return self.out_proj(pooled)

    def forward(
        self,
        history: torch.Tensor,
        edges: torch.Tensor,
        is_car: torch.Tensor,
        nbr_mask: torch.Tensor,
        bio_seq: torch.Tensor | None = None,
        q_bio: torch.Tensor | None = None,
        graph_scale: float = 1.0,
        obs_dt: torch.Tensor | None = None,
        fut_times_s: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, t_obs, _ = history.shape
        origin, basis, valid = local_frenet_torch(history)
        body = (history - origin) @ basis
        dt_step = history.new_full((bsz, t_obs - 1), 0.1) if obs_dt is None else obs_dt.to(history.dtype).clamp_min(1e-4)
        vel = torch.cat((torch.zeros_like(body[:, :1]), (body[:, 1:] - body[:, :-1]) / dt_step[:, :, None]), dim=1)
        spd = torch.linalg.vector_norm(vel, dim=2, keepdim=True)
        fp = torch.cat((body, vel, spd), dim=2).float()
        fn = fp * fp.new_tensor([1.0, -1.0, 1.0, -1.0, 1.0])
        _, hb = self.base.motion(torch.cat((fp, fn), dim=0))
        cb = self.base.ctx_proj(hb[0])
        ctx_pos, ctx_neg = cb[:bsz], cb[bsz:]

        if graph_scale > 0.0:
            dp_loc = torch.einsum("bmk,bks->bms", edges[:, :, 0:2].double(), basis).float()
            dv_loc = torch.einsum("bmk,bks->bms", edges[:, :, 2:4].double(), basis).float()
            inv = torch.stack((edges[:, :, 4] / 10.0, edges[:, :, 5], edges[:, :, 6], is_car), dim=-1).float()
            ep = torch.cat((dp_loc, dv_loc, inv), dim=-1)
            en = torch.cat((dp_loc * ep.new_tensor([1.0, -1.0]), dv_loc * ep.new_tensor([1.0, -1.0]), inv), dim=-1)
            ctx_pos = ctx_pos + float(graph_scale) * self._graph_delta(ctx_pos, ep, nbr_mask)
            ctx_neg = ctx_neg + float(graph_scale) * self._graph_delta(ctx_neg, en, nbr_mask)

        ctx_even = 0.5 * (ctx_pos + ctx_neg)
        ctx_odd = 0.5 * (ctx_pos - ctx_neg)
        last_v = (torch.linalg.vector_norm(history[:, -1] - history[:, -2], dim=-1) / dt_step[:, -1]).float()
        if self.base.pose_mode == "bio_pose" and bio_seq is not None and q_bio is not None:
            regime = q_bio * torch.exp(-((last_v / self.base.v_ref) ** 2)) * torch.sigmoid(self.base.gate_logit)
            _, hbio = self.base.bio_gru(F.silu(self.base.bio_in(bio_seq)))
            h_bio = hbio[0]
        else:
            regime, h_bio = last_v.new_zeros(bsz), last_v.new_zeros((bsz, 24))

        c_dx = self.base.central_dx(ctx_even).view(bsz, 3, 3)
        c_dy = self.base.central_dy(ctx_odd).view(bsz, 3, 3)
        p_pos = self.base.pair_knots(ctx_pos).view(bsz, 3, 2)
        p_neg = self.base.pair_knots(ctx_neg).view(bsz, 3, 2)
        if self.base.pose_mode == "bio_pose" and bio_seq is not None and q_bio is not None:
            c_dx = c_dx + regime[:, None, None] * self.base.bio_par_central(h_bio).view(bsz, 3, 3)
            dp = regime[:, None] * self.base.bio_par_pair(h_bio)
            p_pos = torch.stack((p_pos[..., 0] + dp, p_pos[..., 1]), dim=-1)
            p_neg = torch.stack((p_neg[..., 0] + dp, p_neg[..., 1]), dim=-1)

        c_knots = torch.stack((c_dx, c_dy), dim=-1)
        m3 = p_pos[:, None]
        m4 = torch.stack((p_neg[..., 0], -p_neg[..., 1]), dim=-1)[:, None]
        knots5 = torch.cat((c_knots[:, :2], m3, m4, c_knots[:, 2:]), dim=1).double()
        knots5 = torch.cat((torch.zeros_like(knots5[:, :, :1]), knots5), dim=2)
        c_sc = self.base.central_scores(ctx_even)
        s3, s4 = self.base.pair_score(ctx_pos), self.base.pair_score(ctx_neg)
        logits = torch.where(valid[:, None], torch.cat((c_sc[:, :3], s3, s4, c_sc[:, 3:]), dim=1), 0.0)

        t_sec = torch.arange(1, 31, dtype=history.dtype, device=history.device)[None, :] * 0.1 if fut_times_s is None else fut_times_s.to(history.dtype)
        idx = (t_sec - 1e-9).floor().long().clamp(0, 2)
        phase = (t_sec - idx.to(history.dtype))[:, None, :, None]
        b_i = torch.arange(bsz, device=history.device)[:, None, None]
        m_i = torch.arange(5, device=history.device)[None, :, None]
        delta = knots5[b_i, m_i, idx[:, None, :]] * (1.0 - phase) + knots5[b_i, m_i, (idx + 1)[:, None, :]] * phase
        world = torch.einsum("bkts,bws->bktw", delta, basis)
        cv = origin + ((history[:, -1:] - history[:, -2:-1]) / dt_step[:, -1:, None]) * t_sec[:, :, None]
        return {"paths": torch.cat((cv[:, None], cv[:, None] + world), dim=1), "logits": logits, "valid": valid}


def _pinns_window_edges(history_snaps: list[dict], ego_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ego = ("Person", int(ego_id))
    now = history_snaps[-1]
    def st(k: tuple[str, int]) -> tuple[np.ndarray, int, float]:
        cnt = sum(k in s for s in history_snaps)
        gap = next((d for d in range(1, 5) if k in history_snaps[-1 - d]), 0)
        v = (now[k] - history_snaps[-1 - gap][k]) / (0.1 * gap) if gap else np.zeros(2)
        return v, gap, cnt / 5.0
    ego_v, ego_gap, rho = st(ego)
    pool = []
    for kind, ident in now:
        if (kind, ident) == ego:
            continue
        dist = float(np.hypot(*(now[(kind, ident)] - now[ego])))
        if 1e-5 < dist <= 10.0:
            pool.append((dist, kind, ident))
    pool.sort()
    edge = np.zeros((12, 9), dtype=np.float32)
    car = np.zeros(12, dtype=np.float32)
    mask = np.zeros(12, dtype=np.float32)
    for j, (dist, kind, ident) in enumerate(pool[:12]):
        v, gap, rho_j = st((kind, ident))
        dp = now[(kind, ident)] - now[ego]
        ok = bool(ego_gap and gap)
        dv = v - ego_v if ok else np.zeros(2)
        rad = -float(dp @ dv) / (dist + 1e-6) if ok else 0.0
        edge[j] = [dp[0], dp[1], dv[0], dv[1], dist, rad, float(ok), rho_j, rho]
        car[j] = 1.0 if kind == "Car" else 0.0
        mask[j] = 1.0
    return edge, car, mask


def load_pinns_graph_cohort(meta_root: Path, pose_npz: Path) -> dict:
    from scripts.run_pinns_bioeq_benchmark import prepare_cohort
    from src.data.pinns_extractor import _check_right_handed_homography, _parse_pinns_label_file
    from scripts.benchmark_heldout_sites import load_site_windows

    tensors, idx, keys, _ = prepare_cohort(meta_root, pose_npz)
    groups, _, _ = load_site_windows(meta_root)
    rows = [r for s in sorted(groups) for r in groups[s]]
    clips = np.array([r.clip for r in rows], dtype=str)
    site_arr = np.empty(len(keys), dtype="<U24")
    for s, ids in idx.items():
        site_arr[ids] = s
    edges = np.zeros((len(keys), 12, 9), dtype=np.float32)
    cars = np.zeros((len(keys), 12), dtype=np.float32)
    masks = np.zeros((len(keys), 12), dtype=np.float32)
    for c in sorted(set(clips.tolist())):
        calib = json.loads((meta_root / "videos" / c.rsplit("_", 1)[0] / "calib.json").read_text())
        parsed = _parse_pinns_label_file(meta_root / "labels" / f"{c}.txt", _check_right_handed_homography(calib), calib["y_min"], calib["y_max"])
        snaps = {t: {(o["type"], int(o["id"])): np.array([o["wx"], o["wy"]], dtype=np.float64) for o in v if o["type"] in ("Person", "Car")} for t, v in parsed.items()}
        for i in np.flatnonzero(clips == c):
            w = rows[i]
            edges[i], cars[i], masks[i] = _pinns_window_edges([snaps[t] for t in range(w.t0 - 4, w.t0 + 1)], w.person_id)
    return {
        **tensors,
        "site": site_arr,
        "clip": clips,
        "edges": torch.from_numpy(edges),
        "is_car": torch.from_numpy(cars),
        "nbr_mask": torch.from_numpy(masks),
    }


def _fit_base_and_graph(seed: int, data: dict, tr_ids: np.ndarray) -> tuple[O2NullSinkGraphForecaster, float]:
    torch.manual_seed(seed)
    base = O2BioEqJointForecaster(pose_mode="bio_pose", action_mode="none")
    for n, p in base.named_parameters():
        p.requires_grad = not (n.startswith("bio_") or n == "gate_logit")
    w_tr = make_site_weights(data["site"][tr_ids], torch.device("cpu"))
    hist, truth = data["history"][tr_ids], data["truth"][tr_ids]
    bseq, qbio = data["bio_seq"][tr_ids], data["q_bio"][tr_ids]
    edg, car, msk = data["edges"][tr_ids], data["is_car"][tr_ids], data["nbr_mask"][tr_ids]

    opt1 = torch.optim.AdamW([p for p in base.parameters() if p.requires_grad], lr=2e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(45):
        for ch in torch.randperm(len(tr_ids), generator=gen).split(64):
            loss = o2_joint_loss(base(hist[ch]), truth[ch], w_tr[ch], act_weight=0.0)
            if loss is not None:
                opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()

    for n, p in base.named_parameters():
        p.requires_grad = n.startswith("bio_") or n == "gate_logit"
    opt2 = torch.optim.AdamW([p for p in base.parameters() if p.requires_grad], lr=1.5e-3, weight_decay=2e-4)
    for _ in range(25):
        for ch in torch.randperm(len(tr_ids), generator=gen).split(64):
            loss = o2_joint_loss(base(hist[ch], bseq[ch], qbio[ch]), truth[ch], w_tr[ch], act_weight=0.0, reg_only=True)
            if loss is not None:
                opt2.zero_grad(set_to_none=True); loss.backward(); opt2.step()

    for p in base.parameters():
        p.requires_grad = False
    gnet = O2NullSinkGraphForecaster(base)
    g_params = [p for n, p in gnet.named_parameters() if not n.startswith("base.")]
    optg = torch.optim.AdamW(g_params, lr=1.5e-3, weight_decay=3e-4)
    for _ in range(30):
        for ch in torch.randperm(len(tr_ids), generator=gen).split(64):
            out = gnet(hist[ch], edg[ch], car[ch], msk[ch], bseq[ch], qbio[ch] * 0.5, graph_scale=1.0)
            loss = o2_joint_loss(out, truth[ch], w_tr[ch], act_weight=0.0)
            if loss is not None:
                optg.zero_grad(set_to_none=True); loss.backward(); optg.step()
    return gnet.eval(), 0.5


def select_graph_scale(seed: int, data: dict, tr_ids: np.ndarray) -> float:
    src_sites = sorted(set(data["site"][tr_ids].tolist()))
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    scores = np.zeros(len(grid), dtype=np.float64)
    for vs in src_sites:
        fi = tr_ids[data["site"][tr_ids] != vs]
        vi = tr_ids[data["site"][tr_ids] == vs]
        gn, bsc = _fit_base_and_graph(seed, data, fi)
        with torch.inference_mode():
            for gi, gs in enumerate(grid):
                out = gn(data["history"][vi], data["edges"][vi], data["is_car"][vi], data["nbr_mask"][vi],
                         data["bio_seq"][vi], data["q_bio"][vi] * bsc, graph_scale=gs)
                top = out["logits"].argmax(1)
                pr = out["paths"][torch.arange(len(vi)), top]
                scores[gi] += float(torch.linalg.vector_norm(pr - data["truth"][vi], dim=-1).mean())
    return float(grid[int(scores.argmin())])


@torch.inference_mode()
def eval_pinns_rows(gn: O2NullSinkGraphForecaster, data: dict, ids: np.ndarray, bsc: float, gsc: float, mode: str) -> np.ndarray:
    msk = data["nbr_mask"][ids]
    car = data["is_car"][ids]
    if mode == "empty":
        msk = torch.zeros_like(msk)
    elif mode == "drop_car":
        msk = msk * (car < 0.5)
    elif mode == "drop_ped":
        msk = msk * (car > 0.5)
    out = gn(data["history"][ids], data["edges"][ids], car, msk, data["bio_seq"][ids], data["q_bio"][ids] * bsc, graph_scale=gsc)
    p = out["paths"]
    q = out["logits"].softmax(dim=1)
    top = p[torch.arange(len(ids)), q.argmax(1)]
    bay = torch.einsum("bk,bktw->btw", q.double(), p)
    dt = torch.linalg.vector_norm(top - data["truth"][ids], dim=-1)
    db = torch.linalg.vector_norm(bay - data["truth"][ids], dim=-1)
    return torch.stack((dt.mean(1), dt[:, -1], db.mean(1), db[:, -1]), dim=1).numpy()


@torch.inference_mode()
def eval_dut_transfer(dut_zip: Path, fold_models: list[tuple[O2NullSinkGraphForecaster, float]]) -> dict:
    with zipfile.ZipFile(dut_zip) as zf:
        pg = np.load(io.BytesIO(zf.read("private_graph.npz")), allow_pickle=False)
        et = np.load(io.BytesIO(zf.read("errors_and_targets.npz")), allow_pickle=False)
        keys = json.loads(str(pg["keys_json"]))
        clips = np.array([k[0] for k in keys], dtype=str)
        h_rel = pg["history"] - pg["history"][:, 4:5]
        frames = pg["frames"]
        edges_raw = pg["edges"].astype(np.float32)
        kind = pg["source"]
        truth = et["truth"]
        cv_err = et["cv_measured"]
        res_ch = pg["reserve_changed"].astype(bool)
        res_last = pg["reserve_last_edge"].astype(np.float32)

    offsets = np.array([(j * 23976 + 4999) // 10000 for j in range(35)], dtype=np.int64)
    obs_dt = torch.from_numpy(np.diff(frames, axis=1) * (1000.0 / 23976.0))
    fut_s = torch.from_numpy(np.broadcast_to((offsets[5:] - offsets[4]) * (1000.0 / 23976.0), (len(frames), 30)).copy())
    mats = {
        "0": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "90": np.array([[0.0, -1.0], [1.0, 0.0]]),
        "180": np.array([[-1.0, 0.0], [0.0, -1.0]]),
        "270": np.array([[0.0, 1.0], [-1.0, 0.0]]),
        "mirror_y": np.array([[1.0, 0.0], [0.0, -1.0]]),
    }
    def clip_eq(err: np.ndarray, m: np.ndarray | None = None) -> dict:
        sel = np.ones(len(err), dtype=bool) if m is None else m
        cs = sorted(set(clips[sel].tolist()))
        v = np.mean([err[(clips == c) & sel].mean(0) for c in cs], axis=0)
        return {"ade_m": float(v[0]), "fde_m": float(v[1])}

    axes_out = {}
    for ax_name, rot in mats.items():
        h_t = torch.from_numpy(h_rel @ rot.T)
        e_rot = edges_raw.copy()
        e_rot[:, :, 0:2] = (e_rot[:, :, 0:2].astype(np.float64) @ rot.T).astype(np.float32)
        e_rot[:, :, 2:4] = (e_rot[:, :, 2:4].astype(np.float64) @ rot.T).astype(np.float32)
        e_t = torch.from_numpy(e_rot)
        car_t = torch.from_numpy((kind == 1).astype(np.float32))
        msk_t = torch.from_numpy((kind >= 0).astype(np.float32))
        nograph_e, real_e, res_e = [], [], []
        for gn, gsc in fold_models:
            o0 = gn(h_t, e_t, car_t, msk_t, graph_scale=0.0, obs_dt=obs_dt, fut_times_s=fut_s)
            o1 = gn(h_t, e_t, car_t, msk_t, graph_scale=gsc, obs_dt=obs_dt, fut_times_s=fut_s)
            for acc, oo in ((nograph_e, o0), (real_e, o1)):
                pw = oo["paths"].numpy() @ rot
                q = oo["logits"].softmax(1).numpy()
                bp = np.einsum("nk,nktw->ntw", q, pw)
                d = np.linalg.norm(bp - truth, axis=-1)
                acc.append(np.column_stack((d.mean(1), d[:, -1])))
            if ax_name == "0":
                er = e_rot.copy(); cr = (kind == 1).astype(np.float32); mr = (kind >= 0).astype(np.float32)
                er[res_ch, -1] = res_last[res_ch]; cr[res_ch, -1] = 1.0; mr[res_ch, -1] = 1.0
                ors = gn(h_t, torch.from_numpy(er), torch.from_numpy(cr), torch.from_numpy(mr), graph_scale=gsc, obs_dt=obs_dt, fut_times_s=fut_s)
                pw = ors["paths"].numpy() @ rot; q = ors["logits"].softmax(1).numpy()
                bp = np.einsum("nk,nktw->ntw", q, pw); d = np.linalg.norm(bp - truth, axis=-1)
                res_e.append(np.column_stack((d.mean(1), d[:, -1])))
        ng_m, rl_m = np.mean(nograph_e, axis=0), np.mean(real_e, axis=0)
        entry = {"cv_measured": clip_eq(cv_err), "o2_no_graph_bayes": clip_eq(ng_m), "o2_nullsink_real_bayes": clip_eq(rl_m)}
        if ax_name == "0":
            rs_m = np.mean(res_e, axis=0)
            entry["o2_reserve_slot12_car_affected_810"] = clip_eq(rs_m - rl_m, res_ch)
        axes_out[ax_name] = entry
    swing = max(axes_out[k]["o2_nullsink_real_bayes"]["ade_m"] for k in mats) - min(axes_out[k]["o2_nullsink_real_bayes"]["ade_m"] for k in mats)
    return {"windows": len(frames), "max_five_axis_ade_swing_m": float(swing), "axes": axes_out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pinns_metadata", type=Path)
    ap.add_argument("pose_npz", type=Path)
    ap.add_argument("dut_graph_zip", type=Path)
    ap.add_argument("output_zip", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    torch.set_num_threads(4)
    data = load_pinns_graph_cohort(args.pinns_metadata, args.pose_npz)
    n = len(data["site"])
    site_arr, clip_arr = data["site"], data["clip"]
    arms = ("no_graph", "real", "empty", "drop_car", "drop_ped")
    acc = {a: [] for a in arms}
    scales_log = {}
    all_fold_models = []

    for seed in args.seeds:
        cur = {a: np.zeros((n, 4), dtype=np.float64) for a in arms}
        scales_log[str(seed)] = {}
        for s in SITES:
            tr_ids = np.flatnonzero(site_arr != s)
            te_ids = np.flatnonzero(site_arr == s)
            gn, bsc = _fit_base_and_graph(seed, data, tr_ids)
            gsc = select_graph_scale(seed, data, tr_ids)
            scales_log[str(seed)][s] = gsc
            all_fold_models.append((gn, gsc))
            cur["no_graph"][te_ids] = eval_pinns_rows(gn, data, te_ids, bsc, 0.0, "real")
            for a in ("real", "empty", "drop_car", "drop_ped"):
                cur[a][te_ids] = eval_pinns_rows(gn, data, te_ids, bsc, gsc, a)
        for a in arms:
            acc[a].append(cur[a])
        print(f"Done O(2) null-sink graph seed={seed}", flush=True)

    avg = {a: np.mean(acc[a], axis=0) for a in arms}
    def site_eq(arr: np.ndarray) -> dict:
        v = np.mean([arr[site_arr == s].mean(0) for s in SITES], axis=0)
        return {"ade_m": float(v[0]), "fde_m": float(v[1]), "bayes_ade_m": float(v[2]), "bayes_fde_m": float(v[3])}

    dut_res = eval_dut_transfer(args.dut_graph_zip, all_fold_models)
    report = {
        "pinns_windows": n,
        "seeds": list(args.seeds),
        "selected_graph_scales": scales_log,
        "pinns_5seed_site_equal": {a: site_eq(avg[a]) for a in arms},
        "pinns_paired_contrasts": {
            "real_minus_no_graph": paired_site_clip_ci(avg["real"][:, :2] - avg["no_graph"][:, :2], site_arr, clip_arr),
            "real_minus_empty": paired_site_clip_ci(avg["real"][:, :2] - avg["empty"][:, :2], site_arr, clip_arr),
            "empty_minus_no_graph": paired_site_clip_ci(avg["empty"][:, :2] - avg["no_graph"][:, :2], site_arr, clip_arr),
            "drop_car_minus_real": paired_site_clip_ci(avg["drop_car"][:, :2] - avg["real"][:, :2], site_arr, clip_arr),
            "drop_ped_minus_real": paired_site_clip_ci(avg["drop_ped"][:, :2] - avg["real"][:, :2], site_arr, clip_arr),
        },
        "dut_zero_shot_5seed_transfer": dut_res,
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
