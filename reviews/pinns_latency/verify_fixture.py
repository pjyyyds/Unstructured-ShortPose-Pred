"""Independent source-level verification; imports no project library, model or detector."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np


def sha(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def read_cache(archive_path, member, expected_hash, original):
    with zipfile.ZipFile(archive_path) as z:
        raw = z.read(member)
    assert hashlib.sha256(raw).hexdigest() == expected_hash
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        data = {k:z[k] for k in ("clip","t0","person_id","hist_kpts","hist_bbox","pose_inferred",
                                   "ego_type","img_h","metadata_json")}
        data["history"] = z["hist_pos_relative"] if original else z["xy"][:, :20].copy()
    roster = defaultdict(dict)
    for i,(c,t,p) in enumerate(zip(data["clip"].tolist(), data["t0"].tolist(), data["person_id"].tolist())):
        assert p not in roster[c,t]
        roster[c,t][p] = i
    return data, roster


def group_counts(roster):
    return {str(n):sum(len(ids) >= n for ids in roster.values()) for n in (1,4,8,12)}


def parse_labels(file, end, cal):
    # Derivative orientation via signed polygon areas at the four corresponding calibration corners.
    def orientation(points):
        x,y = np.asarray(points).T
        return np.sign(x @ np.roll(y,-1) - y @ np.roll(x,-1))
    flip = orientation(cal["pts_2d"]) == orientation(cal["pts_3d"])
    frames = defaultdict(dict)
    with file.open() as src:
        for line in src:
            a = line.split()
            label = int(a[0])
            if label < 1 or label > 3*end+1 or (label-1)%3:
                continue
            t = (label-1)//3
            key = a[1],int(a[2])
            assert key not in frames[t] and key[0] in ("Car","Person")
            ox,oy,_,w,h,x,y = map(float,a[3:10])
            frames[t][key] = (np.array([x, cal["y_max"]-(y-cal["y_min"]) if flip else y]),
                              np.array([ox,oy-h*.5,w,h],dtype="f4"), int(float(a[10])))
    return frames, bool(flip)


def image_contours(video, cal, flip):
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS)-30) < .05
    ok, image = cap.read()
    cap.release()
    assert ok and image.shape[:2] == (cal["bev_height"], cal["bev_width"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray,(5,5),1.2),60,150)
    found,_ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in sorted(found,key=lambda c:cv2.arcLength(c,False),reverse=True)[:10]:
        if len(c) < 4 or cv2.arcLength(c,False) < .12*max(gray.shape):
            continue
        u,v = c[:,0].astype("f4").T
        x = cal["x_min"] + u/(gray.shape[1]-1)*(cal["x_max"]-cal["x_min"])
        y = cal["y_min"] + (1-v/(gray.shape[0]-1))*(cal["y_max"]-cal["y_min"])
        if flip:
            y = cal["y_max"] - (y-cal["y_min"])
        lines.append(np.stack([x,y],axis=-1))
    return lines, hashlib.sha256(image.tobytes()).hexdigest()


def local_map(lines, origin):
    origin = origin.astype("f4")
    out = np.zeros((16,10,11),dtype="f4")
    d = [np.linalg.norm(points-origin,axis=1).min() for points in lines]
    for i,idx in enumerate(sorted((k for k in range(len(d)) if d[k]<=20),key=lambda k:d[k])):
        p = lines[idx].astype("f8")
        s = np.r_[0.,np.cumsum(np.hypot(*np.diff(p,axis=0).T))]
        sample = np.linspace(0.,s[-1],10)
        out[i,:,:2] = np.stack([np.interp(sample,s,p[:,j]) for j in (0,1)],axis=1).astype("f4")-origin
        out[i,:,[2,9,10]] = 1
    return out


def history_features(xy, joints, box, height):
    n = len(xy)
    motion = np.zeros((n,9),dtype="f4")
    relative = xy-xy[-1]
    motion[:,:2] = relative
    v = np.diff(relative,axis=0)/.1
    motion[1:,2:4] = v
    motion[1:,4] = np.linalg.norm(v,axis=1)
    motion[:,[5,8]] = 1
    motion[1:,6] = 1
    motion[1:,7] = .1
    conf = np.clip(joints[:,:,2],0,1)
    good = (conf>=.2) & (box[:,3,None]>1)
    both = good[:,11] & good[:,12]
    root = np.where(both[:,None], (joints[:,11,:2]+joints[:,12,:2])*.5, box[:,:2])
    scale = box[:,3,None,None]
    pose = np.zeros((n,17,6),dtype="f4")
    pose[:,:,:2] = np.clip((joints[:,:,:2]-root[:,None])/scale,-3,3)*good[:,:,None]
    pose[:,:,2] = conf*good
    pose[:,:,3] = good
    pose[:,:,4] = both[:,None]
    pose[:,:,5] = np.log(np.maximum(box[:,3]/height,1e-4))[:,None]
    if not good.any():
        pose.fill(0.)
    def pair(ix):
        return [good[:,ix].mean(),(conf[:,ix]*good[:,ix]).sum()/(float(good[:,ix].sum())+1e-6)]
    q = [*pair(slice(None)),np.minimum(box[:,3]/height,1.).mean(),1.]
    qp = [v for ix in ([0,1,2,3,4,5,6],[5,6,7,8,9,10,11,12],[11,12,13,14,15,16]) for v in pair(ix)]
    return motion,pose,q,qp


def observed_neighbors(frames,t,pid):
    ego = ("Person",pid)
    at = frames[t]
    def state(key):
        tt = [j for j in range(t-4,t+1) if key in frames[j]]
        v = (at[key][0]-frames[tt[-2]][key][0])/(.1*(t-tt[-2])) if len(tt)>1 else np.zeros(2)
        return v,len(tt)
    candidates = []
    for k in at:
        if k==ego:
            continue
        delta = at[k][0]-at[ego][0]
        d = np.linalg.norm(delta)
        if 1e-5 < d <= 10.:
            candidates.append((d,k))
    e = np.zeros((12,9),dtype="f4")
    types = np.zeros((12,8),dtype="f4")
    mask = np.zeros(12,dtype="f4")
    ve,ne = state(ego)
    current_ids = []
    for slot,(d,k) in enumerate(sorted(candidates)[:12]):
        pos = at[k][0]-at[ego][0]
        vn,nn = state(k)
        valid = nn>1 and ne>1
        dv = vn-ve if valid else np.zeros(2)
        e[slot] = [*pos,*dv,d,-np.dot(pos,dv)/(d+1e-6),valid,nn/5,ne/5]
        typ = 6 if k[0]=="Car" else 2 if nn>1 and np.linalg.norm(vn)<.2 else 0
        types[slot,typ],mask[slot] = 1,1
        current_ids.append(k)
    return e,types,mask,current_ids


def run(work,repo):
    fix = work / "latency_audit/sync"
    proto = json.loads((fix/"protocol.json").read_bytes())
    assert sha(repo / proto["checkpoint"]) == proto["checkpoint_sha256"]
    h1 = proto["original_pose_member_sha256"]
    h2 = proto["matched_inputs_sha256"]
    old,full = read_cache(work/"shortpose_heldout.zip","pose.npz",h1,True)
    new,matched = read_cache(work/"matched_durations_gpu.zip","matched_observations.npz",h2,False)
    keys = sorted((c,t,p) for (c,t),ids in matched.items() if len(ids)>=12 for p in ids)
    scenes = sorted({(c,t) for c,t,_ in keys})
    assert len(keys)==24 and len(scenes)==2 and len(new["t0"])==623 and len(old["t0"])==1482
    assert group_counts(full)==proto["original_roster_qualifying_camera_ticks_by_minimum_targets"]
    assert group_counts(matched)==proto["matched_roster_qualifying_camera_ticks_by_minimum_targets"]
    m1,m2 = [json.loads(str(z["metadata_json"])) for z in (old,new)]
    assert m1["model_sha256"]==m2["onnx_sha256"] and m2["pose_parent_sha256"]==h1
    assert m1["label_and_calibration_digest_sha256"]==m2["data_sha256"]
    assert sha(work/"loso_rtmpose_m.onnx")==m1["model_sha256"]
    cv2.setNumThreads(1)
    ar = {}
    shapes = {"x_m":(None,9),"x_p":(None,17,6),"q_vec":(4,),"q_part":(6,),"ego_type_oh":(8,),
              "p_env":(16,10,11),"m_map":(1,),"nbr_edges":(12,9),"nbr_type_oh":(12,8),"nbr_mask":(12,)}
    for n in (5,10,20):
        file = fix/f"sync_T{n}.npz"
        assert sha(file)==proto["files_sha256"][file.name]
        with np.load(file,allow_pickle=False) as z:
            assert set(z.files)==set(shapes)|{"clip","t0","person_id","site"}
            assert list(zip(z["clip"],z["t0"],z["person_id"]))==keys
            ar[n] = {k:z[k] for k in z.files}
            for k,shape in shapes.items():
                a=z[k]
                assert a.shape == (24,*(n if i is None else i for i in shape))
                assert a.dtype==np.float32 and np.isfinite(a).all()
    scenes_out = []
    max_error = defaultdict(float)
    def verify(name, actual, expected, tol=0):
        delta = np.max(np.abs(np.asarray(actual,dtype=float)-np.asarray(expected,dtype=float)))
        assert delta <= tol, (name,delta,tol)
        max_error[name] = max(max_error[name],float(delta))
    for clip,t in scenes:
        condition,number = clip.rsplit("_",1)
        files = [f"videos/{condition}/calib.json",f"labels/{clip}.txt",f"videos/{condition}/videos/bev_{number}.mp4"]
        calib,label,video = [work/root/p for p,root in zip(files,["pinns_metadata","pinns_metadata","geometry_audit/bev_sources"])]
        for loc,p in zip(files,(calib,label,video)):
            assert sha(p)==proto["source_files_sha256"][loc]
        config=json.loads(calib.read_bytes())
        raw_video = calib.parent/"videos"/f"ori_{number}.mp4"
        assert sha(raw_video)==m1["video_sha256"][clip]
        with label.open() as src:
            first_person_number=min(int(a[0]) for line in src if (a:=line.split())[1]=="Person")
        frames,flip=parse_labels(label,t,config)
        assert t-19>0 and config["fps"]==30
        polys,pixelsha=image_contours(video,config,flip)
        scene={"clip":clip,"tick":t,"members":sorted(matched[clip,t]),"first_history_tick":t-19,
               "first_last_source_label_number_20":[3*(t-19)+1,3*t+1],"first_last_nominal_video_index_20":[3*(t-19),3*t],
               "source_camera_fps":config["fps"],"map_bev_frame":0,"map_frame_pixels_sha256":pixelsha,
               "native_video_bytes_match_pose_parent_sha256":m1["video_sha256"][clip],
               "first_person_source_label_number":first_person_number,
               "source_contours":len(polys),"ground_y_flipped":flip,"active_neighbor_slots":0,"slots_with_velocity":0,
               "source_target_neighbor_types":Counter(),"source_background_neighbor_slots":0,"unique_background_ids":set(),
               "nonempty_map_rows":0,"valid_map_polylines_across_rows":0,"pose_inferred_20":0,
               "joint_masks_by_T":Counter(),"ego_types":Counter(),"observed_source_types_t0":dict(Counter(k[0] for k in frames[t]))}
        for ii,(c,_,pid) in enumerate(keys):
            if c!=clip:
                continue
            si,li = matched[clip,t][pid],full[clip,t][pid]
            coords = np.stack([frames[j]["Person",pid][0] for j in range(t-19,t+1)])
            boxes = np.stack([frames[j]["Person",pid][1] for j in range(t-19,t+1)])
            pose = new["hist_kpts"][si]
            faces = new["pose_inferred"][si]
            verify("boxes_20_from_source",boxes,new["hist_bbox"][si])
            verify("motion_20_from_source",coords-coords[-1],new["history"][si],1e-10)
            verify("motion_5_source_and_old",coords[-5:]-coords[-1],old["history"][li],1e-10)
            verify("joints_5_match_parent",pose[-5:],old["hist_kpts"][li])
            verify("boxes_5_match_parent",boxes[-5:],old["hist_bbox"][li])
            verify("inferred_5_match_parent",faces[-5:],old["pose_inferred"][li])
            assert faces.all() and new["img_h"][si]==old["img_h"][li]==config["v_size"]
            ego = 2 if frames[t]["Person",pid][2]==1 or np.linalg.norm(coords[-1]-coords[-2])/.1<.2 else 0
            assert old["ego_type"][li]==new["ego_type"][si]==ego
            scene["ego_types"][str(ego)]+=1
            scene["pose_inferred_20"]+=int(faces.sum())
            env=local_map(polys,coords[-1]); edges,types,mask,ids=observed_neighbors(frames,t,pid)
            scene["active_neighbor_slots"]+=len(ids)
            scene["slots_with_velocity"]+=int(edges[:,6].sum())
            scene["source_target_neighbor_types"].update(k[0] for k in ids)
            external=[k for k in ids if k[0]!="Person" or k[1] not in matched[clip,t]]
            scene["source_background_neighbor_slots"]+=len(external)
            scene["unique_background_ids"].update(external)
            scene["nonempty_map_rows"]+=int(env[:,:,10].any())
            scene["valid_map_polylines_across_rows"]+=int(env[:,0,10].sum())
            for n,z in ar.items():
                assert z["site"][ii]==config["location"]+"_"+config["scene"]
                mot,xp,q,qp = history_features(coords[-n:],pose[-n:],boxes[-n:],config["v_size"])
                scene["joint_masks_by_T"][str(n)]+=int(xp[:,:,3].sum())
                for name,expected,tol in (("x_m",mot,1e-6),("x_p",xp,1e-6),("q_vec",q,1e-6),("q_part",qp,1e-6),
                    ("ego_type_oh",np.eye(8)[ego],0),("p_env",env,1e-5),("m_map",[1.],0),
                    ("nbr_edges",edges,1e-6),("nbr_type_oh",types,0),("nbr_mask",mask,0)):
                    verify(name,z[name][ii],expected,tol)
        scene["unique_background_ids"] = len(scene["unique_background_ids"])
        pub=next(e for e in proto["scene_support"] if e["site"]==config["location"]+"_"+config["scene"])
        assert [len(scene["members"]),len(polys),scene["active_neighbor_slots"],scene["source_target_neighbor_types"]["Car"],scene["nonempty_map_rows"]]==[pub[k] for k in ("targets","scene_contours","neighbor_slots","car_slots","map_rows_with_points")]
        scene["distinct_sampled_targets"] = len(scene.pop("members"))
        scenes_out.append(scene)
    return {"protocol_sha256":sha(fix/"protocol.json"),"files_sha256":proto["files_sha256"],"numpy":np.__version__,"opencv":cv2.__version__,
            "origin_rtmpose_sha256":m1["model_sha256"],"matched_cache_sha256":h2,"original_pose_sha256":h1,
            "checkpoint_sha256":proto["checkpoint_sha256"],"source_files_sha256_verified":proto["source_files_sha256"],
            "original_camera_ticks_by_minimum_roster":group_counts(full),"matched_camera_ticks_by_minimum_roster":group_counts(matched),
            "same_24_keys_in_all_three_durations":True,"ten_model_inputs_plus_four_identity_arrays_only":True,
            "max_absolute_differences":dict(max_error),"scenes":scenes_out,
            "limits":["Enrollment inherits complete future labels from the 623-window study; no future measurement contributes to any exported feature.",
                      "Native annotations use 3*t+1 per frozen loader and pose images use frame index 3*t; the selected American clip contains source label frame 0. A possible one-frame image/label shift remains.",
                      "BEV frame-zero Canny contours are image edges. Recomputed pixels depend on the decode and OpenCV stack.",
                      "No inference, latency measurement, or output prediction was performed."]}


if __name__=="__main__":
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("work",type=Path)
    args.add_argument("repo",type=Path)
    args.add_argument("output",type=Path)
    a=args.parse_args()
    a.output.write_text(json.dumps(run(a.work,a.repo),indent=2,ensure_ascii=False,allow_nan=False)+"\n")
