"""Check the release's four-corner map against original/BEV video correspondence."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


CLIPS = (
    ("america_crossroad_summer_daytime_sunny", 0),
    ("america_crossroad_summer_daytime_sunny", 1),
    ("america_crossroad_summer_nightfall_sunny", 0),
    ("america_crossroad_summer_nightfall_sunny", 1),
)
FRAMES = (0, 60, 150)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def describe(values):
    return {"n": len(values), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95))}


def transform(points, matrix):
    hom = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if np.any(np.abs(hom[:, 2]) < 1e-10):
        raise ValueError("Point at infinity")
    return hom[:, :2] / hom[:, 2:]


def descriptors(image):
    keypoints, desc = cv2.SIFT_create().detectAndCompute(image, None)
    if desc is None:
        raise ValueError("No image descriptors")
    return np.array([k.pt for k in keypoints]), desc


def ratio_matches(source, destination):
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(source, destination, k=2)
    return {a.queryIdx: a.trainIdx for a, b in pairs if a.distance < .75 * b.distance}


def corresponding_points(warped, actual):
    src, src_desc = descriptors(warped)
    dst, dst_desc = descriptors(actual)
    forward = ratio_matches(src_desc, dst_desc)
    backward = ratio_matches(dst_desc, src_desc)
    ij = [(i, j) for i, j in forward.items() if backward.get(j) == i]
    if len(ij) < 4:
        raise ValueError("Insufficient geometric correspondence")
    a, b = np.array(ij, dtype=int).T
    return src[a], dst[b]


def read_gray(cap, frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, im = cap.read()
    if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != frame + 1:
        raise ValueError(f"Could not decode frame {frame}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def label_comparison(path, calib, warp, correction, width, height):
    # Raw Person world_x/world_y in these four American clips increase with BEV u/v.
    rows = []
    for line in path.read_text().splitlines():
        f = line.split()
        if f[1] == "Person" and int(f[0]) >= 1 and (int(f[0])-1) % 3 == 0:
            rows.append([float(f[i]) for i in (3, 4, 8, 9)])
    data = np.asarray(rows)
    span = np.array([calib["x_max"]-calib["x_min"], calib["y_max"]-calib["y_min"]])
    meters_per_pixel = span / [width - 1, height - 1]
    start = [calib["x_min"], calib["y_min"]]
    observed = transform(data[:, :2], warp)
    def errors(pixels):
        distances = np.linalg.norm(start + pixels * meters_per_pixel - data[:, 2:], axis=1)
        return describe(distances)
    return {"released_corners_m": errors(observed),
            "after_independent_image_alignment_m": errors(transform(observed, correction))}


def check_clip(metadata, bevs, condition, ident):
    base = Path("videos") / condition
    calpath = metadata / base / "calib.json"
    original = metadata / base / "videos" / f"ori_{ident}.mp4"
    bev = bevs / base / "videos" / f"bev_{ident}.mp4"
    label = metadata / "labels" / f"{condition}_{ident}.txt"
    cal = json.loads(calpath.read_bytes())
    width, height = cal["bev_width"], cal["bev_height"]
    # Correspondences are given in this order in all four released JSON files.
    world = np.array(cal["pts_3d"])
    corners = np.array([[0,height-1],[width-1,height-1],[width-1,0],[0,0]], dtype="f8")
    ground = np.column_stack((world[:,0]/cal["x_max"]*(width-1),
                              world[:,1]/cal["y_max"]*(height-1)))
    np.testing.assert_allclose(ground, corners, atol=.001, rtol=0)
    corners[:,1] = height - 1 - corners[:,1]
    warp = cv2.getPerspectiveTransform(np.asarray(cal["pts_2d"],dtype="f4"), corners.astype("f4"))
    frames = {}
    correction = None
    src, dst = cv2.VideoCapture(str(original)), cv2.VideoCapture(str(bev))
    try:
        assert src.isOpened() and dst.isOpened()
        assert abs(src.get(cv2.CAP_PROP_FPS)-30) < .1 and abs(dst.get(cv2.CAP_PROP_FPS)-30) < .1
        for frame in FRAMES:
            image, target = read_gray(src, frame), read_gray(dst, frame)
            if image.shape != (cal["v_size"],cal["u_size"]) or target.shape != (height,width):
                raise ValueError("Published calibration and decoded image sizes disagree")
            prewarped = cv2.warpPerspective(image, warp, (width,height))
            a,b = corresponding_points(prewarped,target)
            rec = {"matches":len(a),"released_corner_offset_px":describe(np.linalg.norm(a-b,axis=1))}
            if frame == FRAMES[0]:
                cv2.setRNGSeed(42)
                candidate, inliers = cv2.findHomography(a,b,cv2.RANSAC,3.0)
                rec["fit_inliers"] = int(inliers.sum()) if inliers is not None else 0
                if candidate is not None and rec["fit_inliers"] >= 12:
                    correction = candidate
            if correction is not None:
                rec["image_alignment_offset_px"] = describe(np.linalg.norm(transform(a,correction)-b,axis=1))
            frames[str(frame)] = rec
    finally:
        src.release()
        dst.release()
    return {"sha256": {p.name:sha(p) for p in (calpath,original,bev,label)},
            "frames":frames, "labels":label_comparison(label,cal,warp,correction,width,height) if correction is not None else None}


def run(metadata, bevs):
    cv2.setNumThreads(1)
    return {"protocol_sha256":sha(Path(__file__).with_name("PROTOCOL.md")),
            "opencv":cv2.__version__, "numpy":np.__version__,
            "clips":{f"{c}_{i}":check_clip(metadata,bevs,c,i) for c,i in CLIPS}}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("metadata","bev_root","output"):
        p.add_argument(key,type=Path)
    args=p.parse_args()
    report=run(args.metadata,args.bev_root)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for name, item in report["clips"].items():
        print(name,[(v["matches"],round(v["released_corner_offset_px"]["median"],3)) for v in item["frames"].values()],item["labels"])
