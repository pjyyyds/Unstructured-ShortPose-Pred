"""Rebuild verified matched-history inputs and replay the four predictor variants on CPU."""
import hashlib
import io
import json

import numpy as np
import torch
from torch import nn

from src.data.features import build_pose_and_quality_features
from src.models.predictor import UnstructuredShortPosePredictor


CACHE_SHA = '59e59e3dfa6a2b5583ce4c3fbcb010866e793a3595e43cf691dbefbfcddc64f1'
PROTOCOL = dict(seed=42, epochs=28, batch_size=64, learning_rate=.002, min_lr_ratio=.05,
                weight_decay=.0001, gradient_clip=2.0)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def id_hash(keys):
    return sha(json.dumps(keys, separators=(',', ':')).encode())


def ids(arr):
    return list(zip(arr['clip'].tolist(), arr['person_id'].tolist(), arr['t0'].tolist()))


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name] for name in z.files}


def verify_sources(cache_bytes, census_path):
    assert sha(cache_bytes) == CACHE_SHA
    cache, census = load_npz(io.BytesIO(cache_bytes)), load_npz(census_path)
    current, original = ids(cache), ids(census)
    assert len(set(current)) == len(current) == 623 and current == sorted(current)
    assert len(set(original)) == len(original) == 1482
    strict = np.flatnonzero(census['strict_disjoint_20'])
    assert set(current) == {original[i] for i in strict}
    current_row = {key: i for i, key in enumerate(current)}
    ordered_rows = [current_row[original[i]] for i in strict]
    site_codes = census['site'][strict].tolist()
    groups = {s: np.array([r for r, name in zip(ordered_rows,site_codes) if name == s]) for s in sorted(set(site_codes))}
    assert len(groups) == 4 and len(set(cache['clip'].tolist())) == 47 and census['history_20_mask'][strict].all()
    assert np.isfinite(cache['xy']).all() and np.all(cache['xy'][:,19] == 0) and cache['pose_inferred'].all()
    expected = np.concatenate([census['history_20_xy'][strict], census['future_30_xy'][strict]], axis=1)
    np.testing.assert_array_equal(cache['xy'][ordered_rows], expected)
    for left, right in zip(current, current[1:]):
        assert left[:2] != right[:2] or right[2] - left[2] >= 50
    return cache, groups, current


def inputs(cache, t):
    pos = cache['xy'][:,20-t:20]
    motion = np.zeros((len(pos),t,9), dtype=np.float32)
    motion[:,:,:2] = pos - pos[:,-1:]
    velocity = np.diff(pos, axis=1) / .1
    motion[:,1:,2:4] = velocity
    motion[:,1:,4] = np.linalg.norm(velocity,axis=-1)
    motion[:,:,5] = 1
    motion[:,1:,6] = 1
    motion[:,1:,7] = .1
    motion[:,:,8] = 1
    samples = [build_pose_and_quality_features(k[-t:], b[-t:], np.ones(t), float(h))
               for k,b,h in zip(cache['hist_kpts'],cache['hist_bbox'],cache['img_h'])]
    xp, quality, _, buckets = zip(*samples)
    feat = dict(x_m=motion, x_p=np.stack(xp), q_vec=np.stack(quality),
                ego_type_oh=np.eye(8,dtype=np.float32)[cache['ego_type']])
    return {key: torch.from_numpy(value) for key,value in feat.items()}, np.asarray(buckets)


def measure(forecast, truth):
    distances = np.linalg.norm(forecast.astype(np.float64)-truth.astype(np.float64),axis=-1)
    return np.column_stack([distances.mean(axis=1),distances[:,-1]])


class ReplayConcat(nn.Module):
    """Independently implement learned [motion, present pose] projection and the absent-pose bypass."""
    def __init__(self, norm):
        super().__init__()
        self.out_norm = norm
        d, = norm.normalized_shape
        self.weight = nn.Parameter(torch.zeros(d, 2*d))

    def forward(self, h_m, h_p=None, q_vec=None, **unused):
        if h_p is None or q_vec is None:
            return self.out_norm(h_m), h_m.new_zeros((len(h_m),1))
        present = q_vec[:,0] > 0
        projected = h_m.clone()
        projected[present] = torch.cat([h_m[present],h_p[present]],dim=-1) @ self.weight.T
        return self.out_norm(projected), present[:,None].to(h_m.dtype)


def model_for(name):
    if name != "M1-concat":
        return UnstructuredShortPosePredictor(name)
    m = UnstructuredShortPosePredictor("M1a")
    m.quality_gate = ReplayConcat(m.quality_gate.out_norm)
    m.variant = name
    return m


@torch.inference_mode()
def forward(model_name, state, feat, rows):
    model = model_for(model_name).cpu().eval()
    model.load_state_dict(state,strict=True)
    assert all(torch.isfinite(v).all().item() for v in state.values())
    fields = ("x_m","ego_type_oh") if model_name=="M0" else ("x_m","ego_type_oh","x_p","q_vec")
    paths, probs = [], []
    for i in range(0,len(rows),31):
        block=rows[i:i+31]
        output=model({k:feat[k][block] for k in fields})
        paths.append(output['trajs'].cpu().numpy());probs.append(output['mode_probs'].cpu().numpy())
    p, q=np.concatenate(paths),np.concatenate(probs)
    assert p.shape==(len(rows),6,30,2) and q.shape==(len(rows),6) and np.isfinite(p).all() and np.isfinite(q).all()
    assert np.min(q)>=0 and np.max(q)<=1
    np.testing.assert_allclose(q.sum(1),1,atol=1e-6,rtol=0)
    return p,q
