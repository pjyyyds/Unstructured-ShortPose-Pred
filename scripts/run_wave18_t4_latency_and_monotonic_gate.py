"""Wave 18 GPU2-B (Slides 9, 14, 17): Tesla T4 hardware latency/VRAM/FLOPs table + structurally monotonic q-gate remedy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from scripts.ppt_mainline_o2_models import PPTMainlineO2Forecaster, local_frenet_with_pose
from scripts.run_dair_o2_bioeq_external import SEEDS
from scripts.run_jrdb_rc1_pose_quality_and_initiation_5seed import _loc_bootstrap_ci
from scripts.run_jrdb_rc2_rc3_graph_and_actions_5seed import fast_person_bootstrap_ci, load_jrdb_10scene_rc2_rc3
from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import _copy_matching, load_ppt_cohort_with_causal_maps
from scripts.run_wave17_ppt_ext_baselines_loso_5seed import _make_ext_model
from scripts.run_wave17_slide5_pose_noise_and_gate_response import (
    TEST_CONDS,
    build_jrdb_stress_packs,
    build_pinns_stress_packs,
    eval_model_on_pack,
    summarize_gate_response,
)
from scripts.run_wave18_conflict_gated_interaction_5seed import ConflictGatedM2bForecaster, safe_site_clip_ci
from src.models.bioeq_o2_joint import o2_joint_loss
from src.models.o2_unified_full import O2UnifiedFullForecaster


class MonotonicQualityM1Forecaster(PPTMainlineO2Forecaster):
    """M1 with structurally monotonic quality gate g = m_pose * (q1*q2)^gamma * sigmoid(b0 + softplus(w_q)^T q)."""

    def __init__(self) -> None:
        super().__init__(arm="M1")
        self.mono_log_gamma = nn.Parameter(torch.tensor(0.35))
        self.mono_w_raw = nn.Parameter(torch.tensor([0.8, 0.8, 0.3, 0.2]))
        self.mono_bias = nn.Parameter(torch.tensor(-0.8))

    def compute_mono_gate(self, q_vec: torch.Tensor, m_pose: torch.Tensor) -> torch.Tensor:
        q = q_vec.clamp(0.0, 1.0)
        gamma = F.softplus(self.mono_log_gamma).clamp(0.1, 4.0)
        prod_q = (q[:, 0] * q[:, 1]).clamp(0.0, 1.0).pow(gamma)
        w_pos = F.softplus(self.mono_w_raw)
        lin = self.mono_bias + (q * w_pos[None, :]).sum(dim=-1)
        return m_pose * prod_q * torch.sigmoid(lin)

    def forward(self, history: torch.Tensor, bio_even: torch.Tensor | None = None, pose_vec_world: torch.Tensor | None = None,
                signed_knee: torch.Tensor | None = None, q_vec: torch.Tensor | None = None, m_pose: torch.Tensor | None = None,
                p_env: torch.Tensor | None = None, m_map: torch.Tensor | None = None, edges: torch.Tensor | None = None,
                is_car: torch.Tensor | None = None, nbr_mask: torch.Tensor | None = None, obs_dt: torch.Tensor | None = None,
                fut_times_s: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        bsz, t_obs, _ = history.shape
        pvw_m = (pose_vec_world * m_pose[:, None, None, None].to(pose_vec_world.dtype)) if (pose_vec_world is not None and m_pose is not None) else None
        origin, basis, valid = local_frenet_with_pose(history, pvw_m, use_pose_backup=(m_pose is not None))
        body = (history - origin) @ basis
        dt_s = history.new_full((bsz, t_obs - 1), 0.1) if obs_dt is None else obs_dt.to(history.dtype).clamp_min(1e-4)
        vel = torch.cat((torch.zeros_like(body[:, :1]), (body[:, 1:] - body[:, :-1]) / dt_s[:, :, None]), dim=1)
        fp = torch.cat((body, vel, torch.linalg.vector_norm(vel, dim=2, keepdim=True)), dim=2).float()
        _, hm = self.motion(torch.cat((fp, fp * fp.new_tensor([1.0, -1.0, 1.0, -1.0, 1.0])), dim=0))
        cb = self.ctx_proj(hm[0])
        cp, cn = cb[:bsz], cb[bsz:]
        disp_all = torch.linalg.vector_norm(history[:, 1:] - history[:, :-1], dim=-1).sum(dim=-1).float()
        gate = disp_all.new_zeros(bsz)
        if bio_even is not None and pose_vec_world is not None and signed_knee is not None and m_pose is not None:
            pvl = torch.einsum("btvk,bks->btvs", pose_vec_world.double(), basis).float()
            fc, fl = pvl[:, :, 0], pvl[:, :, 1]
            flp = fc.new_tensor([1.0, -1.0])
            _, hpb = self.pose_gru(F.silu(self.pose_in(torch.cat((torch.cat((bio_even, fc, fl, signed_knee), -1), torch.cat((bio_even, fc * flp, fl * flp, -signed_knee), -1)), 0))))
            hpp, hpn = hpb[0][:bsz], hpb[0][bsz:]
            qv = q_vec if q_vec is not None else disp_all.new_zeros((bsz, 4))
            gate = self.compute_mono_gate(qv, m_pose)
            cp, cn = cp + gate[:, None] * self.pose_proj(hpp), cn + gate[:, None] * self.pose_proj(hpn)
        ce, co = 0.5 * (cp + cn), 0.5 * (cp - cn)
        ld = torch.tanh((disp_all / 0.25).clamp_min(0.05))[:, None, None]
        ck = torch.stack((ld * self.central_dx(ce).view(bsz, 3, 3), ld * self.central_dy(co).view(bsz, 3, 3)), dim=-1)
        pp_k, pn_k = ld * self.pair_knots(cp).view(bsz, 3, 2), ld * self.pair_knots(cn).view(bsz, 3, 2)
        knots5 = torch.cat((ck[:, :2], pp_k[:, None], torch.stack((pn_k[..., 0], -pn_k[..., 1]), -1)[:, None], ck[:, 2:]), dim=1).double()
        knots5 = torch.cat((torch.zeros_like(knots5[:, :, :1]), knots5), dim=2)
        csc = self.central_scores(ce)
        c0 = csc[:, :1] + self.cv_low_bonus * torch.exp(-((disp_all / 0.25) ** 2))[:, None]
        logits = torch.where(valid[:, None], torch.cat((c0, csc[:, 1:3], self.pair_score(cp), self.pair_score(cn), csc[:, 3:]), dim=1), 0.0)
        ts = torch.arange(1, 31, dtype=history.dtype, device=history.device)[None, :] * 0.1 if fut_times_s is None else fut_times_s.to(history.dtype)
        idx = (ts - 1e-9).floor().long().clamp(0, 2)
        ph = (ts - idx.to(history.dtype))[:, None, :, None]
        bi, mi = torch.arange(bsz, device=history.device)[:, None, None], torch.arange(5, device=history.device)[None, :, None]
        delta = knots5[bi, mi, idx[:, None, :]] * (1.0 - ph) + knots5[bi, mi, (idx + 1)[:, None, :]] * ph
        cv = origin + ((history[:, -1:] - history[:, -2:-1]) / dt_s[:, -1:, None]) * ts[:, :, None]
        return {"paths": torch.cat((cv[:, None], cv[:, None] + torch.einsum("bkts,bws->bktw", delta, basis)), dim=1), "logits": logits, "valid": valid, "gate": gate}


def _measure_fn(fn, device: torch.device, n_warmup: int, n_timed: int) -> dict[str, float]:
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as fcm:
        _ = fn()
    flops = int(fcm.get_total_flops())

    with torch.inference_mode():
        for _ in range(n_warmup):
            _ = fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            base_mem = torch.cuda.memory_allocated(device)
            times_ms = []
            for _ in range(n_timed):
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                _ = fn()
                ev1.record()
                torch.cuda.synchronize(device)
                times_ms.append(float(ev0.elapsed_time(ev1)))
            peak_mib = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            act_mib = float(max(0, torch.cuda.max_memory_allocated(device) - base_mem)) / (1024.0 * 1024.0)
        else:
            times_ms = []
            for _ in range(n_timed):
                t0 = time.perf_counter()
                _ = fn()
                times_ms.append((time.perf_counter() - t0) * 1000.0)
            peak_mib, act_mib = 0.0, 0.0

    arr = np.asarray(times_ms, dtype=np.float64)
    p50, p95, p99 = np.percentile(arr, [50.0, 95.0, 99.0])
    return {"flops": flops, "mean_ms": float(arr.mean()), "std_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "p50_ms": float(p50), "p95_ms": float(p95), "p99_ms": float(p99), "peak_vram_mib": peak_mib, "activation_vram_mib": act_mib}


def profile_all_models_hardware(device: torch.device, n_warmup: int = 50, n_timed: int = 250) -> dict:
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    out: dict[str, object] = {"device_name": dev_name, "n_warmup": n_warmup, "n_timed": n_timed, "batch_sizes": {}}
    for bsz in (1, 12):
        hist = torch.randn(bsz, 5, 2, dtype=torch.float64, device=device) * 0.4
        be = torch.randn(bsz, 5, 11, device=device)
        pvw = torch.randn(bsz, 5, 2, 2, dtype=torch.float64, device=device) * 0.2
        sk = torch.randn(bsz, 5, 1, device=device)
        qv = torch.full((bsz, 4), 0.85, device=device)
        mp = torch.ones(bsz, device=device)
        pe = torch.randn(bsz, 16, 10, 11, device=device)
        mm = torch.ones(bsz, device=device)
        edg = torch.randn(bsz, 12, 9, device=device).abs()
        car = torch.zeros(bsz, 12, device=device)
        nmk = torch.ones(bsz, 12, device=device)
        bseq = torch.randn(bsz, 5, 11, device=device)
        qbio = torch.full((bsz,), 0.85, device=device)
        ext_b = {"x_m": torch.randn(bsz, 5, 9, device=device), "x_p": torch.randn(bsz, 5, 17, 6, device=device),
                 "p_env": pe, "m_map": mm[:, None], "nbr_edges": edg, "nbr_mask": nmk, "nbr_type_oh": torch.zeros(bsz, 12, 8, device=device)}

        specs: list[tuple[str, nn.Module, object]] = []
        for arm in ("M0", "M1a", "M1b", "M1", "M2a", "M2b", "M2", "M3"):
            m = PPTMainlineO2Forecaster(arm).to(device).eval()
            specs.append((f"O2_{arm}", m, lambda m=m: m(hist, be, pvw, sk, qv, mp, pe, mm, edg, car, nmk)))
        mm_q = MonotonicQualityM1Forecaster().to(device).eval()
        specs.append(("O2_M1_mono_q", mm_q, lambda mm_q=mm_q: mm_q(hist, be, pvw, sk, qv, mp)))
        cg = ConflictGatedM2bForecaster(use_conflict_gate=True).to(device).eval()
        specs.append(("O2_M2b_conflict_gate", cg, lambda cg=cg: cg(hist, be, pvw, sk, qv, mp, pe, mm, edg, car, nmk)))
        uf = O2UnifiedFullForecaster().to(device).eval()
        specs.append(("Ours_O2_Full", uf, lambda uf=uf: uf(hist, bseq, qbio, edg, car, nmk)))
        for enm, ek in (("Ext1_MCENET", "mcenet"), ("Ext2_HST", "hst"), ("Ext3_SocialTransmotion", "st")):
            em = _make_ext_model(ek, fut_steps=30, dt=0.1).to(device).eval()
            specs.append((enm, em, lambda em=em: em(ext_b)))

        b_res = {}
        for nm, mod, fn in specs:
            st = _measure_fn(fn, device, n_warmup, n_timed)
            st["params"] = int(sum(p.numel() for p in mod.parameters()))
            st["agents_per_sec"] = float(bsz * 1000.0 / max(st["mean_ms"], 1e-6))
            b_res[nm] = st
        out["batch_sizes"][f"B{bsz}"] = b_res
    return out


def evaluate_monotonic_gate_remedy(
    tag: str, grp_arr: np.ndarray, unit_arr: np.ndarray, is_pinns: bool, hist_g: torch.Tensor, truth_g: torch.Tensor,
    truth_np: np.ndarray, packs: dict[str, dict[str, torch.Tensor]], q_buckets: np.ndarray, obs_dt_g: torch.Tensor,
    fut_s_g: torch.Tensor, bsz: int, ep_m0: int, ep_m1: int, device: torch.device,
) -> tuple[dict, dict[str, np.ndarray]]:
    n = len(truth_np)
    groups = sorted(set(grp_arr.tolist()))
    arms = ("M0", "M1_unconstrained", "M1_mono_q")
    top_s = {(tc, a): np.zeros((len(SEEDS), n, 2)) for tc in TEST_CONDS for a in arms}
    gate_s = {(tc, a): np.zeros((len(SEEDS), n)) for tc in TEST_CONDS for a in ("M1_unconstrained", "M1_mono_q")}

    for si, seed in enumerate(SEEDS):
        for g in groups:
            tr_ids = np.flatnonzero(grp_arr != g)
            te_ids = np.flatnonzero(grp_arr == g)
            w_tr = make_site_weights(grp_arr, tr_ids).to(device)
            tr_t = torch.from_numpy(tr_ids).to(device)
            nat = packs["Test_Natural"]

            def _fit(net: PPTMainlineO2Forecaster, prefs: tuple[str, ...], ep: int, lr: float, wd: float, soff: int) -> PPTMainlineO2Forecaster:
                for nm, p in net.named_parameters():
                    p.requires_grad = nm.startswith(prefs) or nm in prefs
                opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
                gen = torch.Generator().manual_seed(seed + soff)
                net.train()
                for _ in range(ep):
                    for ch in torch.randperm(len(tr_ids), generator=gen).split(bsz):
                        cg = ch.to(device); ig = tr_t[cg]
                        out = net(hist_g[ig], nat["bio_even"][ig], nat["pose_vec_world"][ig], nat["signed_knee"][ig], nat["quality"][ig], nat["m_pose"][ig], obs_dt=obs_dt_g[ig], fut_times_s=fut_s_g[ig])
                        loss = o2_joint_loss(out, truth_g[ig], w_tr[cg], act_weight=0.0)
                        if loss is not None:
                            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                return net.eval()

            torch.manual_seed(seed)
            m0 = _fit(PPTMainlineO2Forecaster("M0").to(device), ("motion.", "ctx_proj.", "cv_low_bonus", "central_", "pair_"), ep_m0, 2e-3, 1e-4, 0)
            torch.manual_seed(seed); m1u = PPTMainlineO2Forecaster("M1").to(device); _copy_matching(m1u, m0)
            m1u = _fit(m1u, ("pose_", "gate_", "central_", "pair_", "cv_low_bonus"), ep_m1, 1.3e-3, 3e-4, 12)
            torch.manual_seed(seed); m1m = MonotonicQualityM1Forecaster().to(device); _copy_matching(m1m, m0)
            m1m = _fit(m1m, ("pose_", "mono_", "central_", "pair_", "cv_low_bonus"), ep_m1, 1.3e-3, 3e-4, 15)
            mods = {"M0": m0, "M1_unconstrained": m1u, "M1_mono_q": m1m}
            for tc in TEST_CONDS:
                for a in arms:
                    tp, _, gt = eval_model_on_pack(mods[a], hist_g, packs[tc], obs_dt_g, fut_s_g, truth_np, te_ids, device)
                    top_s[(tc, a)][si, te_ids] = tp
                    if a != "M0":
                        gate_s[(tc, a)][si, te_ids] = gt
        print(f"[W18-MONO {tag}] seed={seed} Jit M1u={top_s[('Test_Jitter020','M1_unconstrained')][si].mean(0)[0]:.4f} M1m={top_s[('Test_Jitter020','M1_mono_q')][si].mean(0)[0]:.4f}", flush=True)

    all_m = np.ones(n, dtype=bool)
    def _pci(d2: np.ndarray) -> dict:
        return safe_site_clip_ci(d2, grp_arr, unit_arr) if is_pinns else {"location_equal": _loc_bootstrap_ci(d2, grp_arr), **fast_person_bootstrap_ci(d2, all_m, unit_arr)}

    cond_out = {}
    for tc in TEST_CONDS:
        cell = {a: {"group_equal_ade1_m": float(np.mean([top_s[(tc, a)].mean(0)[grp_arr == g, 0].mean() for g in groups])),
                    "group_equal_fde1_m": float(np.mean([top_s[(tc, a)].mean(0)[grp_arr == g, 1].mean() for g in groups]))} for a in arms}
        cell["M1_mono_minus_M1_unconstrained"] = _pci(top_s[(tc, "M1_mono_q")].mean(0) - top_s[(tc, "M1_unconstrained")].mean(0))
        cell["M1_mono_minus_M0"] = _pci(top_s[(tc, "M1_mono_q")].mean(0) - top_s[(tc, "M0")].mean(0))
        cond_out[tc] = cell
    q_np, mp_np = packs["Test_Natural"]["quality"].cpu().numpy(), packs["Test_Natural"]["m_pose"].cpu().numpy()
    g_aud = {
        "M1_unconstrained": summarize_gate_response(gate_s[("Test_Natural", "M1_unconstrained")], q_np, mp_np, q_buckets),
        "M1_mono_q": summarize_gate_response(gate_s[("Test_Natural", "M1_mono_q")], q_np, mp_np, q_buckets),
        "mean_gate_by_cond": {tc: {a: float(gate_s[(tc, a)].mean()) for a in ("M1_unconstrained", "M1_mono_q")} for tc in TEST_CONDS},
    }
    npz = {f"{tag}_{tc}_{a}_top1": top_s[(tc, a)].mean(0) for tc in TEST_CONDS for a in arms}
    return {"n_windows": n, "test_conditions": cond_out, "gate_audit": g_aud}, npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--causal-map-bin", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, required=True)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hw_prof = profile_all_models_hardware(dev, n_warmup=50, n_timed=250)
    print(f"[W18-HW] Profiled 14 models on {hw_prof['device_name']}", flush=True)

    dp = load_ppt_cohort_with_causal_maps(args.meta_dir, args.pose_npz, args.joint_zip, args.causal_map_bin)
    np_p = len(dp["history"])
    p_packs = build_pinns_stress_packs(dp, args.meta_dir, args.pose_npz, dev)
    p_res, p_npz = evaluate_monotonic_gate_remedy("pinns", dp["site"], dp["clip"], True, dp["history"].to(dev), dp["truth"].to(dev), dp["truth"].numpy(), p_packs, dp["q_buckets"], torch.full((np_p, 4), 0.1, dtype=torch.float64, device=dev), torch.arange(1, 31, dtype=torch.float64, device=dev)[None].expand(np_p, -1) * 0.1, 64, 45, 25, dev)

    dj = load_jrdb_10scene_rc2_rc3(args.repo_root)
    nj = len(dj["history"])
    j_packs = build_jrdb_stress_packs(dj, dev)
    j_res, j_npz = evaluate_monotonic_gate_remedy("jrdb", dj["location"], dj["actors"], False, dj["history"].to(dev), dj["truth"].to(dev), dj["truth"].numpy(), j_packs, dj["q_buckets"], torch.full((nj, 1), 0.4, dtype=torch.float64, device=dev), torch.arange(1, 8, dtype=torch.float64, device=dev)[None].expand(nj, -1) * 0.4, 256, 45, 28, dev)

    summary = {"experiment": "wave18_t4_latency_and_monotonic_gate_5seed", "device": str(dev), "seeds": list(SEEDS), "hardware_profile": hw_prof, "pinns_monotonic_gate_1482": p_res, "jrdb_monotonic_gate_5836": j_res}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(args.out_dir / "per_window_mono_gate_5seed.npz", **p_npz, **j_npz)
    dig = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out_dir.iterdir()) if p.is_file()}
    (args.out_dir / "sha256.json").write_text(json.dumps(dig, indent=2), encoding="utf-8")
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(args.out_dir.iterdir()):
            zf.write(p, arcname=p.name)
    print(f"DONE: {args.zip_path} sha256={hashlib.sha256(args.zip_path.read_bytes()).hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
