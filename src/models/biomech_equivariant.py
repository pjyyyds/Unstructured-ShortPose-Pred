"""O(2)-equivariant local-frame motion and view-invariant biomechanical pose forecaster."""
from __future__ import annotations

import numpy as np

AXIS_MIN_CHORD_M = 0.05
KP_CONF_MIN = 0.20
BIO_FEATURE_NAMES = (
    "ankle_span",
    "knee_cos_mean",
    "knee_cos_contrast",
    "leg_extension",
    "trunk_upright",
)


def causal_frenet_frame(history: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build causal O(2)-equivariant origin and (e_par, e_perp) basis from 5-frame xy."""
    h = np.asarray(history, dtype=np.float64)
    if h.ndim != 3 or h.shape[1:] != (5, 2):
        raise ValueError("Expected history shape (N, 5, 2)")
    origin = h[:, -1:, :]
    chords = (origin - h[:, :-1, :])[:, ::-1, :]
    norms = np.linalg.norm(chords, axis=-1)
    ok = norms >= (AXIS_MIN_CHORD_M - 1e-12)
    valid = ok.any(axis=1)
    idx = np.argmax(ok, axis=1)
    rows = np.arange(len(h))
    chosen = chords[rows, idx]
    denom = np.maximum(norms[rows, idx], 1e-12)[:, None]
    e_par = np.where(valid[:, None], chosen / denom, 0.0)
    e_perp = np.stack((-e_par[:, 1], e_par[:, 0]), axis=1)
    basis = np.stack((e_par, e_perp), axis=2)
    return origin, basis, valid


def parity_motion_features(history: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract parity-even (+1) and parity-odd (-1) local motion invariants."""
    origin, basis, valid = causal_frenet_frame(history)
    rel = np.einsum("ntw,nws->nts", history - origin, basis)
    vel = np.diff(rel, axis=1) / 0.1
    acc = np.diff(vel, axis=1) / 0.1
    last_speed = np.linalg.norm(history[:, -1] - history[:, -2], axis=-1) / 0.1
    path_len = np.linalg.norm(np.diff(history, axis=1), axis=-1).sum(axis=1)
    even = np.column_stack([
        rel[:, :-1, 0],
        np.abs(rel[:, :-1, 1]),
        vel[:, :, 0],
        np.abs(vel[:, :, 1]),
        acc[:, :, 0],
        last_speed,
        path_len,
    ])
    odd = np.column_stack([
        rel[:, :-1, 1],
        vel[:, :, 1],
        acc[:, :, 1],
    ])
    even = np.where(valid[:, None], even, 0.0)
    odd = np.where(valid[:, None], odd, 0.0)
    return even, odd, last_speed


def _knee_cos(xy: np.ndarray, good: np.ndarray, hip: int, knee: int, ank: int) -> tuple[np.ndarray, np.ndarray]:
    u = xy[:, :, hip] - xy[:, :, knee]
    v = xy[:, :, ank] - xy[:, :, knee]
    nu, nv = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
    ok = good[:, :, hip] & good[:, :, knee] & good[:, :, ank] & (nu > 1e-6) & (nv > 1e-6)
    cos = np.clip((u * v).sum(axis=-1) / np.where(ok, nu * nv, 1.0), -1.0, 1.0)
    return np.where(ok, cos, 0.0), ok


def biomech_features(kpts: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract camera-azimuth-invariant and O(2)-reflection-invariant 5-frame gait descriptors."""
    kp = np.asarray(kpts, dtype=np.float64)
    bx = np.asarray(bbox, dtype=np.float64)
    if kp.ndim != 4 or kp.shape[1:] != (5, 17, 3) or bx.shape != (len(kp), 5, 4):
        raise ValueError("Expected kpts (N,5,17,3) and bbox (N,5,4)")
    h = bx[..., 3]
    f_ok = np.isfinite(h) & (h > 1.0)
    good = np.isfinite(kp).all(axis=-1) & (kp[..., 2] >= KP_CONF_MIN) & f_ok[..., None]
    xy = np.where(good[..., None], kp[..., :2], 0.0) / np.where(f_ok, h, 1.0)[..., None, None]

    ank_ok = good[:, :, 15] & good[:, :, 16]
    ank_span = np.where(ank_ok, np.linalg.norm(xy[:, :, 15] - xy[:, :, 16], axis=-1), 0.0)
    c_l, ok_l = _knee_cos(xy, good, 11, 13, 15)
    c_r, ok_r = _knee_cos(xy, good, 12, 14, 16)
    k_ok = ok_l & ok_r
    k_mean = np.where(k_ok, 0.5 * (c_l + c_r), 0.0)
    k_diff = np.where(k_ok, np.abs(c_l - c_r), 0.0)
    leg_ok = good[:, :, 11] & good[:, :, 15] & good[:, :, 12] & good[:, :, 16]
    leg_ext = np.where(
        leg_ok,
        0.5 * (np.linalg.norm(xy[:, :, 11] - xy[:, :, 15], axis=-1) + np.linalg.norm(xy[:, :, 12] - xy[:, :, 16], axis=-1)),
        0.0,
    )
    tr_ok = good[:, :, [5, 6, 11, 12]].all(axis=-1)
    tr_up = np.where(tr_ok, -0.5 * (xy[:, :, 5, 1] + xy[:, :, 6, 1] - xy[:, :, 11, 1] - xy[:, :, 12, 1]), 0.0)

    seq = np.stack((ank_span, k_mean, k_diff, leg_ext, tr_up), axis=-1)
    masks = np.stack((ank_ok, k_ok, k_ok, leg_ok, tr_ok), axis=-1)
    all5 = masks.all(axis=1)
    weights = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)[None, :, None]
    slope = np.where(all5, (seq * weights).sum(axis=1) / 1.0, 0.0)
    tv = np.where(all5, np.abs(np.diff(seq, axis=1)).mean(axis=1) / 0.1, 0.0)
    last_val = np.where(masks[:, -1], seq[:, -1], 0.0)
    lower_conf = np.where(good[:, :, 11:], kp[:, :, 11:, 2], 0.0).mean(axis=(1, 2))
    q_bio = np.clip(masks[:, :, :4].mean(axis=(1, 2)) * lower_conf, 0.0, 1.0)
    feat = np.column_stack((last_val, slope, tv, all5.astype(np.float64)))
    return feat, q_bio


def _ridge_even(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    mu = x.mean(axis=0)
    sd = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
    z = (x - mu) / sd
    z1 = np.column_stack((np.ones(len(z)), z))
    reg = np.diag([0.0] + [alpha] * z.shape[1])
    w = np.linalg.solve(z1.T @ z1 + len(z) * reg, z1.T @ y)
    return {"mu": mu, "sd": sd, "b": w[0], "w": w[1:]}


def _pred_even(fit: dict, x: np.ndarray) -> np.ndarray:
    return fit["b"] + ((x - fit["mu"]) / fit["sd"]) @ fit["w"]


def _ridge_odd(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    rms = np.sqrt(np.maximum((x * x).mean(axis=0), 1e-12))
    z = x / rms
    w = np.linalg.solve(z.T @ z + len(z) * alpha * np.eye(z.shape[1]), z.T @ y)
    return {"rms": rms, "w": w}


def _pred_odd(fit: dict, x: np.ndarray) -> np.ndarray:
    return (x / fit["rms"]) @ fit["w"]


def last_two_cv(history: np.ndarray) -> np.ndarray:
    steps = np.arange(1, 31, dtype=np.float64)[None, :, None]
    return history[:, -1:, :] + (history[:, -1:, :] - history[:, -2:-1, :]) * steps


def fit_biomech_equivariant(
    history: np.ndarray,
    truth: np.ndarray,
    sites: np.ndarray,
    kpts: np.ndarray | None = None,
    bbox: np.ndarray | None = None,
    alpha_mot: float = 0.2,
    alpha_bio: float = 1.0,
) -> dict:
    """Fit O(2)-equivariant motion backbone + nested source-CV regime-gated biomechanical residual."""
    origin, basis, valid = causal_frenet_frame(history)
    even, odd, speed = parity_motion_features(history)
    anchor = last_two_cv(history)
    res_local = np.einsum("ntw,nws->nts", truth - anchor, basis)
    vm = valid

    def fit_motion_on(idx: np.ndarray) -> tuple[dict, dict]:
        m = idx & vm
        return _ridge_even(even[m], res_local[m, :, 0], alpha_mot), _ridge_odd(odd[m], res_local[m, :, 1], alpha_mot)

    uniq_sites = sorted(set(sites.tolist()))
    mot_grid = (0.0, 0.0625, 0.125, 0.1875, 0.25, 0.375)
    best_g_mot, best_ade = 0.125, np.inf
    oof_mot_par = np.zeros((len(history), 30), dtype=np.float64)
    oof_mot_perp = np.zeros((len(history), 30), dtype=np.float64)
    raw_par = np.zeros((len(history), 30), dtype=np.float64)
    raw_perp = np.zeros((len(history), 30), dtype=np.float64)
    for vs in uniq_sites:
        tr_in, va_in = (sites != vs), (sites == vs)
        fe, fo = fit_motion_on(tr_in)
        raw_par[va_in] = np.where(vm[va_in, None], _pred_even(fe, even[va_in]), 0.0)
        raw_perp[va_in] = np.where(vm[va_in, None], _pred_odd(fo, odd[va_in]), 0.0)
    for g in mot_grid:
        loc = np.stack((g * raw_par, g * raw_perp), axis=-1)
        pred = anchor + np.einsum("nts,nws->ntw", loc, basis)
        ade = np.mean([np.linalg.norm(pred[sites == s] - truth[sites == s], axis=-1).mean() for s in uniq_sites])
        if ade < best_ade:
            best_ade, best_g_mot = ade, g
    oof_mot_par, oof_mot_perp = best_g_mot * raw_par, best_g_mot * raw_perp
    fe_full, fo_full = fit_motion_on(np.ones(len(history), dtype=bool))
    model = {"fe": fe_full, "fo": fo_full, "g_mot": best_g_mot, "use_pose": kpts is not None}
    if kpts is None or bbox is None:
        return model

    bio, q_bio = biomech_features(kpts, bbox)
    rem_par = res_local[:, :, 0] - oof_mot_par
    best_bio = (0.0, 0.8, best_ade)
    for v_ref in (0.5, 0.8, 1.2):
        gate = np.where(vm, q_bio * np.exp(-((speed / v_ref) ** 2)), 0.0)
        gb_oof = np.zeros((len(history), 30), dtype=np.float64)
        for vs in uniq_sites:
            tr_in, va_in = (sites != vs) & (gate > 0.05), (sites == vs)
            fb = _ridge_even(bio[tr_in] * gate[tr_in, None], rem_par[tr_in], alpha_bio)
            gb_oof[va_in] = gate[va_in, None] * _pred_even(fb, bio[va_in] * gate[va_in, None])
        for g_bio in (0.0, 0.05, 0.10, 0.20, 0.35):
            loc = np.stack((oof_mot_par + g_bio * gb_oof, oof_mot_perp), axis=-1)
            pred = anchor + np.einsum("nts,nws->ntw", loc, basis)
            ade = np.mean([np.linalg.norm(pred[sites == s] - truth[sites == s], axis=-1).mean() for s in uniq_sites])
            if ade < best_bio[2] - 1e-7:
                best_bio = (g_bio, v_ref, ade)
    g_bio, v_ref, _ = best_bio
    gate_all = np.where(vm, q_bio * np.exp(-((speed / v_ref) ** 2)), 0.0)
    active = gate_all > 0.05
    fb_full = _ridge_even(bio[active] * gate_all[active, None], (res_local[active, :, 0] - best_g_mot * _pred_even(fe_full, even[active])), alpha_bio)
    model.update({"fb": fb_full, "g_bio": g_bio, "v_ref": v_ref})
    return model


def predict_biomech_equivariant(
    model: dict,
    history: np.ndarray,
    kpts: np.ndarray | None = None,
    bbox: np.ndarray | None = None,
) -> dict:
    """Predict O(2)-equivariant primary path and structured K=6 multimodal trajectories."""
    origin, basis, valid = causal_frenet_frame(history)
    even, odd, speed = parity_motion_features(history)
    anchor = last_two_cv(history)
    par = np.where(valid[:, None], model["g_mot"] * _pred_even(model["fe"], even), 0.0)
    perp = np.where(valid[:, None], model["g_mot"] * _pred_odd(model["fo"], odd), 0.0)
    gate = np.zeros(len(history), dtype=np.float64)
    if model.get("use_pose") and kpts is not None and bbox is not None and model.get("g_bio", 0.0) > 0.0:
        bio, q_bio = biomech_features(kpts, bbox)
        gate = np.where(valid, q_bio * np.exp(-((speed / model["v_ref"]) ** 2)), 0.0)
        par = par + model["g_bio"] * gate[:, None] * _pred_even(model["fb"], bio * gate[:, None])

    loc_primary = np.stack((par, perp), axis=-1)
    primary = anchor + np.einsum("nts,nws->ntw", loc_primary, basis)

    t_sec = np.arange(1, 31, dtype=np.float64)[None, :] * 0.1
    quad = 0.5 * (t_sec ** 2)
    m0 = primary
    m1 = np.where(valid[:, None, None], primary + 0.35 * quad[:, :, None] * basis[:, None, :, 0], origin)
    m2 = np.where(valid[:, None, None], origin + 0.35 * (primary - origin), origin)
    m3 = np.where(valid[:, None, None], primary + 0.25 * quad[:, :, None] * basis[:, None, :, 1], primary)
    m4 = np.where(valid[:, None, None], primary - 0.25 * quad[:, :, None] * basis[:, None, :, 1], primary)
    m5 = anchor
    modes = np.stack((m0, m1, m2, m3, m4, m5), axis=1)
    probs = np.tile(np.array([0.46, 0.10, 0.12, 0.11, 0.11, 0.10], dtype=np.float64), (len(history), 1))
    return {"primary": primary, "modes": modes, "probs": probs, "regime_gate": gate}
