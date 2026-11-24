"""Screen only the motion of the Social-Transmotion training recordings.
Reads directory metadata from the release; opens only matching HST train odometry.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from check_training_coordinates import ARTIFACTS, OUT, TrainingArchive


def main():
    st = TrainingArchive(*ARTIFACTS["tracks"][:2])
    odom = TrainingArchive(*ARTIFACTS["odometry"][:2])
    scenes = sorted(Path(name).name.removesuffix("_shift_0.ndjson") for name in st.training
                    if name.startswith("jrdb/data/train/") and name.endswith("_shift_0.ndjson"))
    assert len(scenes) == 10

    def measure(scene):
        name = f"odometry/train/{scene}.json"
        payload = odom.member(name)
        records = json.loads(payload)["odometry"]
        ordered = [records[k] for k in sorted(records)]
        p = np.asarray([[row["position"][i] for i in "xyz"] for row in ordered])
        q = np.asarray([[row["orientation"][i] for i in "xyzw"] for row in ordered])
        assert np.isfinite(p).all() and np.isfinite(q).all()
        q /= np.linalg.norm(q,axis=1,keepdims=True)
        angles = np.degrees(2*np.arccos(np.clip(np.abs(np.sum(q*q[0],axis=1)),0,1)))
        return scene, {"odometry_frames":len(p), "member_sha256":hashlib.sha256(payload).hexdigest(),
                       "max_xy_translation_from_first_m":float(np.linalg.norm(p[:,:2]-p[0,:2],axis=1).max()),
                       "max_orientation_change_from_first_deg":float(angles.max())}

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = dict(pool.map(measure, scenes))
    report = {"selection":"All and only source release jrdb/data/train/*_shift_0.ndjson stems. No held-out pose, trajectories or scores opened.",
              "wheel_odometry_source":ARTIFACTS["odometry"][0], "recordings":rows}
    (OUT/"train_split_odometry.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for scene, vals in rows.items():
        print(scene,vals["max_xy_translation_from_first_m"],vals["max_orientation_change_from_first_deg"])


if __name__=="__main__":
    main()
