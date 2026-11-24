"""Independent scene contour, coordinate and support replay on the fixed PINNS causal-map cohort."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def read_label(path):
    ticks=set()
    people={}
    for line in path.read_bytes().splitlines():
        s=line.split(); f=int(s[0])-1
        if f<0 or f%3:continue
        tick=f//3;ticks.add(tick)
        if s[1]==b"Person":
            k=tick,int(s[2])
            assert k not in people
            people[k]=np.array([float(s[8]),float(s[9])],dtype=np.float32)
    return np.array(sorted(ticks)),people


def split_ticks(ticks, indexes, id):
    lo,hi=int(ticks.min()),int(ticks.max())
    if len(indexes)>=3:
        n=len(indexes); final=max(1,round(.15*n)); val=max(1,round(.15*n))
        if final+val>=n: final,val=1,1
        ntrain=n-val-final; seq=indexes.index(id)
        if seq<ntrain: return ticks, np.empty(0,int)
        guard=60 if seq in (ntrain,ntrain+val) else 0
        return np.empty(0,int),ticks[ticks>=lo+guard]
    net=max(max(hi-15,lo+3)-lo-120,3);first=max(1,round(.7*net));second=max(1,round(.5*max(net-first,2)))
    train=ticks[ticks<=lo+first]
    held=ticks[((ticks>=lo+first+60)&(ticks<=lo+first+60+second))|(ticks>=lo+first+120+second)]
    return train,held


def extract_pixels(path, indexes, h, w):
    cap=cv2.VideoCapture(str(path))
    frames=[]
    try:
        assert cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS)-30)<.05
        for k in indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(k))
            ok,pic=cap.read();assert ok and pic.shape[:2]==(h,w)
            frames.append(pic)
    finally:cap.release()
    gray=cv2.cvtColor(np.median(np.stack(frames),axis=0).astype("u1"),cv2.COLOR_BGR2GRAY)
    edged=cv2.Canny(cv2.GaussianBlur(gray,(5,5),1.2),60,150)
    polys,_=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    rank=sorted(polys,key=lambda p:cv2.arcLength(p,False),reverse=True)[:10]
    return [p[:,0].astype("i4") for p in rank if len(p)>=4 and cv2.arcLength(p,False)>=.12*max(h,w)],sha(gray.tobytes())


def convert(lines,cal,correct):
    w,h=cal["bev_width"],cal["bev_height"]
    assert (cal["u_axis"],cal["v_axis"])==("x","-y") and w>1 and h>1
    assert cal["x_max"]>cal["x_min"] and cal["y_max"]>cal["y_min"]
    out=[]
    for q in lines:
        u,v=q.astype("f4").T
        x=cal["x_min"]+(u/(w if correct else w-1))*(cal["x_max"]-cal["x_min"])
        ratio=v/h if correct else 1-v/(h-1)
        y=cal["y_min"]+ratio*(cal["y_max"]-cal["y_min"])
        pts=np.column_stack([x,y])
        double=pts.astype("f8")
        s=np.r_[0.,np.cumsum(np.hypot(*np.diff(double,axis=0).T))]
        targets=np.linspace(0,s[-1],10)
        interp=np.column_stack([np.interp(targets,s,double[:,j]) for j in range(2)]).astype("f4")
        out.append((pts,interp))
    return out


def local(lines,xy):
    ten=np.zeros((16,10,11),np.float32)
    valid=[]
    for i,(line,_) in enumerate(lines):
        d=float(np.min(np.linalg.norm(line-xy,axis=1)))
        if d<=20:valid.append((d,i))
    ids=[]
    for row,(_,idx) in enumerate(sorted(valid)[:16]):
        ids.append(idx)
        ten[row,:,:2]=lines[idx][1]-xy
        ten[row,:,[2,9,10]]=1.
    return ten,ids


def lines_digest(lines):
    h=hashlib.sha256()
    for p in lines:
        h.update(np.array(p.shape,dtype="<i4").tobytes())
        h.update(p.astype("<i4").tobytes())
    return h.hexdigest()


def inc(counter,eligible,old_ids,new_ids):
    counter.update(windows=1,source_eligible=int(eligible),old_nonempty=int(bool(old_ids)),new_nonempty=int(bool(new_ids)),
                   old_polylines=len(old_ids),new_polylines=len(new_ids),old_only=int(bool(old_ids) and not new_ids),
                   new_only=int(bool(new_ids) and not old_ids),changed_contour_set=int(set(new_ids)!=set(old_ids)),
                   changed_ordered_contour_list=int(new_ids!=old_ids))


def run(data,bev,original,compare,field):
    cv2.setNumThreads(1)
    assert file_hash(original/"maps.npz")=="47453819b0542cefadb0122685d5a45876596191c21d0a855d35d596c8ee8fe7"
    ref=json.loads((original/"manifest.json").read_bytes())
    with np.load(original/"maps.npz",allow_pickle=False) as z:
        clips,pids,ticks,splits,elig,prev=[z[k] for k in ("clip","person_id","t0","split","source_eligible","p_env_image")]
    keys=list(zip(clips.tolist(),pids.tolist(),ticks.tolist()))
    assert len(set(keys))==len(keys)==7283 and len(set(clips))==47
    expected=np.zeros_like(prev)
    prior_max=0.
    ledger={}
    groups=defaultdict(list); counts=defaultdict(Counter)
    for clip in set(clips):
        condition,i=clip.rsplit("_",1);groups[condition].append(int(i))
    for name in sorted(set(clips)):
        cond,id=name.rsplit("_",1)
        label=data/"labels"/(name+".txt"); config=data/"videos"/cond/"calib.json";c=json.loads(config.read_bytes())
        site=c["location"]+"_"+c["scene"]
        r=ref["sources"][name]
        assert file_hash(label)==r["label_sha256"] and file_hash(config)==r["calibration_sha256"]
        times,people=read_label(label)
        train,held=split_ticks(times,sorted(groups[cond]),int(id))
        source=train[train<=train[0]+60] if len(train) else train
        if len(held):source=source[source<int(held.min())-19]
        if len(source)<7:source=np.empty(0,int)
        assert source.tolist()==r["reference_10hz"]
        raw=(source[np.linspace(0,len(source)-1,7,dtype=int)]*3).tolist() if len(source) else []
        assert raw==r["raw_image_frames"]
        uv=[];gray_sha=None
        if raw:
            path=bev/"videos"/cond/"videos"/f"bev_{id}.mp4"
            assert file_hash(path)==r["bev_sha256"]
            uv,gray_sha=extract_pixels(path,raw,c["bev_height"],c["bev_width"])
        else:assert r["bev_sha256"] is None
        # All 22 current conditions keep raw ground Y in the frozen loader (independent scale audit).
        old_lines,new_lines=convert(uv,c,False),convert(uv,c,True)
        idx=np.flatnonzero(clips==name)
        for i in idx:
            t,p=int(ticks[i]),int(pids[i]);xy=people[t,p]
            allowed=bool(len(source) and t>source[-1])
            assert allowed==bool(elig[i])
            if allowed:assert max(raw)<=3*(t-1)
            old,old_ids=local(old_lines if allowed else [],xy)
            new,new_ids=local(new_lines if allowed else [],xy)
            prior_max=max(prior_max,float(np.abs(prev[i]-old).max()))
            assert np.array_equal(prev[i],old),"Independent old-order reproduction failed"
            expected[i]=new
            for group in ("all",site,str(splits[i]),site+"/"+str(splits[i]),name):
                inc(counts[group],allowed,old_ids,new_ids)
        ledger[name]={"source_frames":raw,"gray_median_sha256":gray_sha,"contour_count":len(uv),
            "integer_contour_vertices":sum(len(x) for x in uv),"contour_geometry_sha256":lines_digest(uv),
            "corrected_local_point_mask_sha256":sha(expected[idx,:,:,10].astype("u1").tobytes()),
            "corrected_local_tensor_float32_sha256":sha(expected[idx].astype("<f4").tobytes()),
            "support":dict(counts[name])}
    delta={"sha256_corrected_tensor_float32_in_parent_order":sha(expected.astype("<f4").tobytes()),
           "sha256_point_mask_uint8_in_parent_order":sha(expected[:,:,:,10].astype("u1").tobytes()),
           "original_exact_max_abs_diff":prior_max}
    if compare is not None:
        with np.load(compare,allow_pickle=False) as z:
            kk=list(zip(z["clip"].tolist(),z["person_id"].tolist(),z["t0"].tolist()));assert len(set(kk))==len(keys) and set(kk)==set(keys)
            lookup={k:i for i,k in enumerate(kk)}
            got=z[field][[lookup[k] for k in keys]]
        delta.update(paired_candidate_sha256=file_hash(compare),paired_candidate_field=field,
            paired_max_absolute_difference=float(np.abs(got-expected).max()),
            paired_support_identical=bool(np.array_equal(got[:,:,:,10],expected[:,:,:,10])),
            paired_elements_bitwise_equal_fraction=float(np.mean(got==expected)))
    return {"protocol":"Original 7,283 ordered keys, reference SHA and eligibility held fixed. Compare legacy N-1/reflected vs actual released raw BEV N/direct formula.",
            "numpy":np.__version__,"opencv":cv2.__version__,"script_sha256":file_hash(Path(__file__)),
            "old_manifest_sha256":file_hash(original/"manifest.json"),"old_maps_sha256":ref["npz_sha256"],"tensor_validation":delta,
            "clips":len(ledger),"clips_using_pixels":sum(v["gray_median_sha256"] is not None for v in ledger.values()),
            "source_frames_total":sum(len(v["source_frames"]) for v in ledger.values()),
            "contours_total":sum(v["contour_count"] for v in ledger.values()),
            "contour_vertices_total":sum(v["integer_contour_vertices"] for v in ledger.values()),
            "support":{k:dict(v) for k,v in counts.items() if k not in ledger},"per_clip":ledger,
            "limits":"Camera imagery and per-person tensors are held locally. No prediction or forecast loss was computed. Recorded contour points are image edges, with unknown road semantics."}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("metadata","bev_root","old_sidecar","output"):p.add_argument(name,type=Path)
    p.add_argument("--compare-npz",type=Path);p.add_argument("--compare-field",default="p_env_image")
    a=p.parse_args();r=run(a.metadata,a.bev_root,a.old_sidecar,a.compare_npz,a.compare_field)
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print(r["clips"],r["clips_using_pixels"],r["source_frames_total"],r["contours_total"],r["support"]["all"],r["tensor_validation"])
