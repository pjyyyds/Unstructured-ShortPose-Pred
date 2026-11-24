"""Measure forecast-window proximity to four partial image annotations using only history."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_pixels(uv, matrix, y_sum):
    homogeneous = np.column_stack([uv, np.ones(len(uv))]) @ matrix.T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-8):
        raise ValueError("Annotation reaches the projective horizon")
    result = homogeneous[:, :2] / homogeneous[:, 2:3]
    result[:, 1] = y_sum - result[:, 1]
    return result


def nearest_distance(points, lines):
    segments = np.concatenate([np.stack([line[:-1], line[1:]], axis=1) for line in lines])
    starts, ends = np.moveaxis(segments, 1, 0)
    vectors = ends - starts
    squared = (vectors * vectors).sum(axis=1)
    if np.any(squared <= 0):
        raise ValueError("Duplicate adjacent vertices")
    offsets = points[:, None] - starts
    proportions = np.clip((offsets * vectors).sum(axis=2) / squared, 0, 1)
    nearest = starts + proportions[:, :, None] * vectors
    return np.linalg.norm(points[:, None] - nearest, axis=2).min(axis=1)


def evaluate(meta, pose_file, annotations):
    plan = json.loads(annotations.read_text())
    result = {}
    counts = {r: 0 for r in (2, 5)}
    with np.load(pose_file, allow_pickle=False) as poses:
        video_hashes = json.loads(str(poses['metadata_json']))['video_sha256']
        for clip, entries in plan['clips'].items():
            condition, number = clip.rsplit('_', 1)
            base = meta/'videos'/condition
            video = base/'videos'/f'ori_{number}.mp4'
            if digest(video) != video_hashes[clip]:
                raise ValueError(f'Incorrect camera source {clip}')
            cam_file = base/'calib.json'
            camera = json.loads(cam_file.read_text())
            matrix, _ = cv2.findHomography(np.asarray(camera['pts_2d'],dtype=float),
                                           np.asarray(camera['pts_3d'],dtype=float), method=0)
            control = np.asarray(camera['pts_2d'],np.float32)
            hull = cv2.convexHull(control)
            selected = poses['clip'] == clip
            persons = poses['person_id'][selected]; times = poses['t0'][selected]
            anchors = [(int(p), 3*int(t)+1) for p,t in zip(persons,times)]
            if not len(times) or 3*(min(times)-4) < plan['source_frame_zero_based']:
                raise ValueError('Missing windows or map frame follows earliest observation')
            labels = {}
            label_file = meta/'labels'/(clip+'.txt')
            for line in label_file.read_text().splitlines():
                row = line.split()
                if row[1].lower().startswith('per'):
                    key = int(row[2]),int(row[0])
                    if key in labels:
                        raise ValueError(f'Duplicate person and frame in {clip}')
                    labels[key] = np.array([float(row[k]) for k in (3,4,8,9)])
            end = np.stack([labels[key] for key in anchors])
            lines = [np.asarray(e['points'],float) for e in entries]
            ysum = camera['y_min'] + camera['y_max']
            points = end[:,2:]
            def summarize(dx, dy):
                world = [project_pixels(line+[dx,dy], matrix, ysum) for line in lines]
                separation = nearest_distance(points,world)
                return world,separation
            world, distance = summarize(0, 0)
            sensitivity = [summarize(dx,dy)[1] for dx in (-3,0,3) for dy in (-3,0,3)]
            n_in_range = {r:int((distance<=r).sum()) for r in counts}
            for r in counts: counts[r] += n_in_range[r]
            inside=lambda uv:cv2.pointPolygonTest(hull,tuple(map(float,uv)),False)>=0
            result[clip] = {
                'video_sha256':video_hashes[clip], 'calibration_sha256':digest(cam_file),'labels_sha256':digest(label_file),
                'source_frame_0based':plan['source_frame_zero_based'],
                'first_forecast_anchor_0based':int(3*min(times)), 'last_forecast_anchor_0based':int(3*max(times)),
                'windows':len(times),'tracklets':len(set(persons.tolist())),
                'line_count':len(lines),'vertex_count':sum(map(len,lines)),
                'vertices_inside_control_point_convex_hull':sum(inside(uv) for line in lines for uv in line),
                'image_anchors_inside_control_point_convex_hull':sum(inside(uv) for uv in end[:,:2]),
                'within_m_count':n_in_range,
                'within_m_count_range_for_common_image_shift_of_up_to_3_pixels':{
                    r:[min(int((z<=r).sum()) for z in sensitivity),max(int((z<=r).sum()) for z in sensitivity)] for r in counts},
                'ground_lines_native_annotation_xy_m':[
                    {'label':entry['label'],'points':line.round(6).tolist()} for entry,line in zip(entries,world)],
                'closest_annotated_edge_m_percentiles':dict(zip(('p25','median','p75'),np.percentile(distance,[25,50,75]).tolist())),
                'pixel_anchor_projection_to_released_xy_median_m':float(np.median(np.linalg.norm(project_pixels(end[:,:2],matrix,ysum)-points,axis=1))),
            }
    return {'scope':'Four actual daylight recordings. The first decoded frame is at 0 s for each. One AI reviewer traced selected visible, static edges. Distances reference the limited mapped segments only. Neither map completeness, independent metric accuracy, human agreement, obstruction nor forecast improvement is established.',
            'annotation_sha256':digest(annotations),'pose_cache_sha256':digest(pose_file),
            'units':'Approximate ground metres in native annotation axes. Camera pixels pass through the supplied four-point homography, then y maps to y_min+y_max-y. Window centres are the actual released pedestrian ground positions at the end of the five historical observations.',
            'sample_rule':'First lexicographic sunny daytime clip at each site in the 47-clip development corpus. No future frames, forecast errors, or other clips used for line placement.',
            'sensitivity':'Nine common image translations (+/-3 pixels per axis), evaluated separately per clip. Sensitivity does not estimate labelling error.',
            'windows':sum(x['windows'] for x in result.values()),'within_m_count':counts,'clips':result}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('metadata_root', type=Path);p.add_argument('pose_cache',type=Path)
    p.add_argument('annotation_json',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    a.output.write_text(json.dumps(evaluate(a.metadata_root,a.pose_cache,a.annotation_json),indent=2,allow_nan=False)+'\n')
