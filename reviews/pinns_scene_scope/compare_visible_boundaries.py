"""Compare finite, visible polylines in both directions using arc-length quadrature."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

RADII = (3, 5, 10)
INTERPRETATIONS = {
    "america_crossroad": "两套均包含远侧两车间的同一段路缘暗带。独立复核只画了这段。第一套另外标出了偏右的一段，且向左超过独立复核端点；单帧内偏右暗线在白色路面标记附近是否始终属于路缘较难确认。",
    "italy_park": "共有前排停车位前方同一条弧形窄石带。独立折线取靠停放车辆的铺面接缝，第一套在共同横向范围较靠近画面下方、更靠前景一侧。可见石带有宽度，不能从该帧确认为抬升路缘的上下缘。第一套另有两个向右的可见分段。",
    "japan_crossroad": "共有右侧路口转角的人行道边缘。独立复核将横过画面的架空线附近断开，并沿略靠车道的一侧记录短段；第一套转弯段稍靠人行道一侧，延伸更远，另画了转角以右与远侧街角的边缘。单帧难以分别确认路缘石的顶缘与底缘。",
    "thailand_street": "共有红白涂装路缘近端斜段。独立复核记录其接触低侧路面的底缘。第一套折线经过斜段靠上的圆角后靠近底缘，且继续覆盖远侧与排水格栅旁的其他段。两套近端未全程分别取独立的上缘与下缘。"
}


def segment_ends(polylines):
    pairs = [np.asarray(line, dtype=float) for line in polylines]
    if any(p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all() for p in pairs):
        raise ValueError("Invalid source polyline")
    start = np.concatenate([p[:-1] for p in pairs])
    end = np.concatenate([p[1:] for p in pairs])
    positive = np.any(end != start, axis=1)
    if not positive.any():
        raise ValueError("No segment of positive length")
    return start[positive], end[positive]


def sample_polylines(polylines, spacing):
    """Each connected polyline contributes its full length without joining separate strokes."""
    points, weights = [], []
    for line in polylines:
        start, end = segment_ends([line])
        offsets = np.linalg.norm(end - start, axis=1)
        breaks = np.r_[0, offsets.cumsum()]
        length = breaks[-1]
        n = int(np.ceil(length / spacing))
        cell = length / n
        positions = (np.arange(n) + 0.5) * cell
        indices = np.searchsorted(breaks[1:], positions, side="right")
        fraction = (positions - breaks[indices]) / offsets[indices]
        points.append(start[indices] + fraction[:, None] * (end - start)[indices])
        weights.append(np.full(n, cell))
    return np.concatenate(points), np.concatenate(weights)


def closest(points, lines):
    start, end = segment_ends(lines)
    direction = end - start
    offset = points[:, None, :] - start
    fraction = np.clip(np.einsum("nsi,si->ns", offset, direction)
                       / np.sum(direction**2, axis=1), 0, 1)
    distance = np.linalg.norm(offset - fraction[:, :, None] * direction, axis=-1)
    return np.min(distance, axis=1)


def directed(source, target, spacing):
    points, weights = sample_polylines(source, spacing)
    distances = closest(points, target)
    order = np.argsort(distances, kind="stable")
    cumulative = np.cumsum(weights[order])
    length = float(weights.sum())

    def quantile(q):
        return float(distances[order[np.searchsorted(cumulative, q * length)]])

    return {
        "source_strokes": len(source), "sample_points": len(points), "source_arc_length_px": length,
        "median_px": quantile(0.5), "p90_px": quantile(0.9),
        "source_arc_fraction_within_px": {
            str(r): float(weights[distances <= r].sum() / length) for r in RADII
        },
    }


def read(path):
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def compare(first, independent, spacing):
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Spacing must be positive")
    pilot, p_hash = read(first)
    second, i_hash = read(independent)
    sites = {}
    for site, annotation in second["annotations"].items():
        if (annotation["width"], annotation["height"], annotation["source_frame"]) != (1536, 864, 0):
            raise ValueError(f"Unexpected frame identity for {site}")
        clip = annotation["source_png"].removesuffix("_0.png")
        one = [r["points"] for r in pilot["clips"][clip]]
        two = annotation["polylines_xy"]
        sites[site] = {
            "clip": clip, "source_image_sha256": annotation["source_sha256"],
            "independent_to_first": directed(two, one, spacing),
            "first_to_independent": directed(one, two, spacing),
            "visual_interpretation": INTERPRETATIONS[site],
        }
    assert len(sites) == len(pilot['clips']) == 4
    return {
        "scope": "四处各一张日间固定机位首帧，两套独立 AI 选段。数值描述可见像素边缘的位置与覆盖差异。没有第二位人类标注，也没有人工真值、物理量测或米制误差。",
        "method": "每个可见折线单独按弧长分为等长单元，在每个单元中点取样；每个中点按其单元长度加权。目标是另一套线的全部有限线段（封闭端点），求到线段并集的精确欧氏最近距离。按来源总弧长计算经验分位数与各半径内弧长比例，间断处不补线，也不外推。远离另一套可见范围亦进入距离统计。",
        "maximum_spacing_px": spacing, "cutoffs_px": list(RADII), "images_wh": [1536, 864],
        "annotation_sha256": {"first": p_hash, "independent": i_hash}, "sites": sites,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("independent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spacing-px", type=float, default=0.25)
    args = parser.parse_args()
    result = compare(args.first, args.independent, args.spacing_px)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
