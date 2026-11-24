"""Rebuild two frame-zero BEV maps against native PINNS label axes while freezing every other feature."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np


OLD_BUNDLE_SHA = "362efbaa576340495749b0820f398d768b82e000e714492265b91183ef061873"
OLD_PROTOCOL_SHA = "b3f33a5b8a4216fb4240d4b3412145315a510d0bd4eaad21cce542de7137f52c"
EXPECTED_CHECKPOINT_SHA = "5069c5be520f2308dd190c01cd68427ba8bdb321d405fc963f5436e911d30689"
LENGTHS = (5, 10, 20)
IDENTITY = ("clip", "t0", "person_id", "site")


def require(ok, msg):
    if not ok:raise ValueError(msg)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def dump(d):
    return (json.dumps(d, indent=2, ensure_ascii=True, allow_nan=False)+"\n").encode()


def check_source(path, rel, manifest):
    if sha(path.read_bytes()) != manifest["source_files_sha256"][rel]:
        raise ValueError(f"Source identity disagreement: {rel}")


def capture_polys(path, c):
    cap=cv2.VideoCapture(str(path))
    try:
        require(cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS)-30)<.05, "invalid source video")
        ok,image=cap.read()
        require(ok and image.shape[:2]==(c["bev_height"],c["bev_width"]), "frame zero missing or size mismatch")
    finally:cap.release()
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edge=cv2.Canny(cv2.GaussianBlur(gray,(5,5),1.2),60,150)
    polys,_=cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    chosen=sorted(polys,key=lambda p:cv2.arcLength(p,False),reverse=True)[:10]
    chosen=[p[:,0] for p in chosen if len(p)>=4 and cv2.arcLength(p,False)>=.12*max(gray.shape)]
    return chosen,sha(image.tobytes())


def geometries(pixels,c,convert,flip,version):
    geom=[]
    for poly in pixels:
        w,h=c["bev_width"],c["bev_height"]
        points=convert(poly,w,h,c,flip,version)
        # Separate computation of the two source conventions. Preserve project's float32 arithmetic.
        u,v=poly.astype("f4").T
        old=version=="legacy_raster"
        direct=np.column_stack((c["x_min"]+u/(w-1 if old else w)*(c["x_max"]-c["x_min"]),
                c["y_min"]+(1-v/(h-1) if old else v/h)*(c["y_max"]-c["y_min"])))
        if flip:direct[:,1]=c["y_max"]-(direct[:,1]-c["y_min"])
        np.testing.assert_array_equal(points,direct)
        path=points.astype("f8")
        arc=np.r_[0.,np.cumsum(np.hypot(*np.diff(path,axis=0).T))]
        require(arc[-1]>0,"degenerate contour")
        targets=np.linspace(0,arc[-1],10)
        points10=np.column_stack([np.interp(targets,arc,path[:,j]) for j in range(2)]).astype("f4")
        geom.append((points,points10))
    return geom


def local(geometry,center):
    m=np.zeros((16,10,11),"f4")
    dist=[(float(np.linalg.norm(vertices-center,axis=1).min()),i) for i,(vertices,_) in enumerate(geometry)]
    ids=[i for d,i in sorted(dist) if d<=20][:16]
    for index,i in enumerate(ids):
        m[index,:,:2]=geometry[i][1]-center
        m[index,:,[2,9,10]]=1.
    return m,ids


def centers(path,t,flip,c):
    found={}
    for line in path.read_bytes().splitlines():
        f=line.split()
        if f[1]==b"Person" and int(f[0])==3*t+1:
            pid=int(f[2])
            require(pid not in found,"duplicate person in native source frame")
            x,y=float(f[8]),float(f[9])
            if flip:y=c["y_max"]-(y-c["y_min"])
            found[pid]=np.array([x,y],dtype="f4")
    return found


def rebuild(bundle,metadata,bev,repo,dest):
    cv2.setNumThreads(1)
    require(sha(bundle.read_bytes())==OLD_BUNDLE_SHA,"expected original private timing input")
    sys.path.insert(0,str(repo.resolve()))
    from src.data.pinns_bev import bev_to_track_xy
    from src.data.pinns_extractor import _check_right_handed_homography
    from src.data import pinns_bev,pinns_extractor
    require(pinns_extractor.cv2 is not None,"OpenCV is required for the camera orientation check")
    require(Path(pinns_bev.__file__).resolve()==repo.resolve()/"src/data/pinns_bev.py","helper imported outside the nominated repo")
    with zipfile.ZipFile(bundle) as z:
        require(z.testzip() is None,"input CRC mismatch")
        proto_raw=z.read("sync/protocol.json"); require(sha(proto_raw)==OLD_PROTOCOL_SHA,"original protocol SHA")
        protocol=json.loads(proto_raw)
        require(protocol["checkpoint_sha256"]==EXPECTED_CHECKPOINT_SHA, "original checkpoint differs")
        past={}
        for t in LENGTHS:
            name=f"sync_T{t}.npz"; content=z.read("sync/"+name)
            require(sha(content)==protocol["files_sha256"][name], "input snapshot SHA")
            with np.load(io.BytesIO(content),allow_pickle=False) as a:past[t]={k:a[k] for k in a.files}
    base=past[5]
    for t in LENGTHS:
        for k in IDENTITY+("p_env","m_map"):
            np.testing.assert_array_equal(base[k],past[t][k])
    require(len(base["clip"])==24,"expected 24 unique person moments")
    groups={}
    for i,(clip,tick) in enumerate(zip(base["clip"].tolist(),base["t0"].tolist())):
        groups.setdefault((clip,tick),[]).append(i)
    require(len(groups)==2 and all(len(v)==12 for v in groups.values()),"two 12-target moments required")
    updated=np.empty_like(base["p_env"]); active=np.empty_like(base["m_map"])
    scene_support=[]; diagnostics=[]
    for (clip,tick), ix in sorted(groups.items()):
        require(tick-19>0,"frame zero must precede the longest entire history")
        condition,number=clip.rsplit("_",1)
        lp=f"labels/{clip}.txt";cp=f"videos/{condition}/calib.json";vp=f"videos/{condition}/videos/bev_{number}.mp4"
        for root,rel in [(metadata,lp),(metadata,cp),(bev,vp)]:check_source(root/rel,rel,protocol)
        c=json.loads((metadata/cp).read_bytes())
        require((c["u_axis"],c["v_axis"])==("x","-y") and c["x_max"]>c["x_min"] and c["y_max"]>c["y_min"],"unvalidated source axes")
        flip=_check_right_handed_homography(c)
        pixels,px_sha=capture_polys(bev/vp,c)
        old,new=(geometries(pixels,c,bev_to_track_xy,flip,v) for v in ("legacy_raster","source_label"))
        xy=centers(metadata/lp,tick,flip,c)
        require(len(set(base["person_id"][ix].tolist()))==12,"duplicate group identity")
        diag=Counter(); site=str(base["site"][ix[0]])
        for row in ix:
            origin=xy[int(base["person_id"][row])]
            first,oi=local(old,origin); last,ni=local(new,origin)
            np.testing.assert_array_equal(first,base["p_env"][row])
            require(float(base["m_map"][row,0])==float(bool(pixels)), "legacy scene availability")
            updated[row],active[row,0]=last,float(bool(pixels))
            diag.update(old_slots=len(oi),source_label_slots=len(ni),new_slot_memberships=len(set(ni)-set(oi)),
                        removed_slot_memberships=len(set(oi)-set(ni)),people_with_different_set=int(set(oi)!=set(ni)),
                        people_with_different_order_or_set=int(oi!=ni),people_with_changed_map=int(not np.array_equal(first,last)))
        old_support=next(s for s in protocol["scene_support"] if s["site"]==site)
        support=dict(old_support,map_rows_with_points=int((updated[ix,:,:,10]>0).any(axis=(1,2)).sum()))
        require(len(pixels)==old_support["scene_contours"],"wrong frame-zero contour membership")
        scene_support.append(support)
        diagnostics.append(dict(site=site,targets=len(ix),raw_map_frame=0,source_bgr_frame_sha256=px_sha,
            camera_handedness_flip=bool(flip),scene_pixel_contours=len(pixels),**diag))
    dest.mkdir(parents=True,exist_ok=False)
    new_hashes={}
    for t in LENGTHS:
        d=dict(past[t],p_env=updated,m_map=active)
        name=f"sync_T{t}.npz"; out=dest/name
        np.savez_compressed(out,**d)
        with np.load(out,allow_pickle=False) as z:
            require(set(z.files)==set(past[t]) and len(z.files)==14,"unexpected input schema")
            for k in z.files:
                value=z[k]
                if k not in ("p_env","m_map"):
                    old=past[t][k]
                    require(value.dtype==old.dtype and value.shape==old.shape and value.tobytes()==old.tobytes(),"non-map input changed")
        new_hashes[name]=sha(out.read_bytes())
    diag={"source_implementation":"src/data/pinns_bev.py", "converter_sha256":sha(Path(pinns_bev.__file__).read_bytes()),
          "preparation_sha256":sha(Path(__file__).read_bytes()),"opencv":cv2.__version__,"numpy":np.__version__,
          "untouched_fields_bytewise_per_duration":sorted(set(base)-{"p_env","m_map"}),"untouched_field_count":12,
          "old_local_map_exact_from_raw_first_frame":True,"m_map_exact_as_parent":bool(np.array_equal(active,base["m_map"])),
          "scene_comparison":diagnostics}
    new_proto=dict(protocol,version="source_label_sync_v2",files_sha256=new_hashes,scene_support=scene_support,
        map="Image Canny contours from each clip's own frame zero. BEV u/W and v/H are scaled into released raw tracking axes, then the actual loader's handedness flag is applied. A fresh 20 m radius selection and arclength resampling are done per person. Image edges have unverified road semantics.",
        parent={"private_bundle_sha256":OLD_BUNDLE_SHA,"protocol_sha256":OLD_PROTOCOL_SHA,"files_sha256":protocol["files_sha256"]},
        map_correction=diag)
    (dest/"protocol.json").write_bytes(dump(new_proto))
    return new_proto


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("original_private_bin","metadata","bev_root","repo","destination"):p.add_argument(name,type=Path)
    a=p.parse_args();r=rebuild(a.original_private_bin,a.metadata,a.bev_root,a.repo,a.destination)
    print("Input protocol",sha((a.destination/"protocol.json").read_bytes()))
    print(json.dumps(r["map_correction"]["scene_comparison"],indent=2))
