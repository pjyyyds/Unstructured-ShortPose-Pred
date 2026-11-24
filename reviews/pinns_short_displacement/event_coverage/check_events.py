"""Reconstruct first annotated, short-path PINNS support and read frozen seed-42 candidate tracks."""
import argparse
from collections import defaultdict
from hashlib import sha256
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile
import numpy as np

EVENTS = ("start","stop","left","right","maintain")
NAMES = ("M0","M1")
HISTORY = "caa8f0959f28f2d69a6f16cc9df115ab6222d0b6eb3297edd575615fef81b94c"
CANDIDATES = "388920f757f0f76d688f281e232d20f2c1afe4178e50771d6f9d0f6146f61748"
PARENT = "0bf2745ab2d5ddcf5f5fbb5d330e6c6cc800625cec301f4c7105d031bdb49d2d"
RERANK = "b24ac2bb1ad3d6742be2cbe16b927e42ef9a1a54a17ce2c08be3e92fa286b871"
FIRST_SHORT = "198c634d9c5d447a14b05327ffd51d738962cf71a628dfcce571551e4e092684"
PROTOCOL = "78ffdcfd4587399ad5d413d184a61ac08dd92642f8c69b86af6969e9d5a74480"


def sha(payload):
    return sha256(payload).hexdigest()


def ordered_keys(keys):
    return sha(json.dumps(keys,separators=(",",":")).encode())


def keys_of(z):
    return list(zip(z["clip"].tolist(),z["person_id"].tolist(),z["t0"].tolist()))


def rules(repo):
    # A second already-verified implementation works directly in complete-track velocities.
    spec = importlib.util.spec_from_file_location("frozen_event_implementation", repo/"reviews/pinns_event_coverage/verify_scores.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.insert(0,str(repo.resolve()))
    from src.data.actions import compute_action_labels
    def validated(h,f):
        out = np.asarray(module.label_complete(h,f),dtype=bool)
        expected,valid = compute_action_labels(h,np.ones(5),f,np.ones(30))
        assert valid==1 and np.array_equal(out,expected)
        return out
    return validated


def mask_and_support(repo,work,label):
    source = work/"pinns_matched_histories.npz"
    assert sha(source.read_bytes())==HISTORY
    with np.load(source,allow_pickle=False) as z:
        keys,sites=keys_of(z),z["site"]
        hist,future=z["history_20_xy"][:,-5:],z["future_30_xy"]
        assert z["history_20_mask"][:,-5:].all() and len(set(keys))==len(keys)==1482
        onset=z["first_person_label_t"]
    first_raw={}
    for c in sorted({k[0] for k in keys}):
        rows=defaultdict(set)
        for l in (work/"pinns_metadata/labels"/(c+".txt")).read_text().splitlines():
            a=l.split()
            raw=int(a[0])-1
            if a[1]=="Person" and raw>=0 and raw%3==0:
                rows[int(a[2])].add(raw//3)
        first_raw.update({(c,p):min(t) for p,t in rows.items()})
    first=np.array([first_raw[c,p]==t-4 for c,p,t in keys])
    assert np.array_equal(onset,[first_raw[c,p] for c,p,t in keys]) and sum(first)==449
    dist=[math.fsum(math.dist(a,b) for a,b in zip(h[:-1],h[1:])) for h in hist]
    subset=first&(np.array(dist)<.5)
    selection=[k for k,m in zip(keys,subset) if m]
    design=json.loads((repo/"reviews/pinns_short_displacement/history_support.json").read_bytes())["design"]
    assert ordered_keys(keys)==design["ordered_keys_sha256"]
    assert ordered_keys(selection)==design["history_only_groups"]["first_short"]["ordered_keys_sha256"]==FIRST_SHORT
    assert len(selection)==len(set(k[:2] for k in selection))==323
    truth=np.array([label(h,f) for h,f in zip(hist[subset],future[subset])])
    row_sites=sites[subset];clips=np.array([k[0] for k in selection])
    def support(m):
        return {"windows":int(m.sum()),"clips":len(set(clips[m])),"truth":{n:{"positive":int((m & truth[:,e]).sum()),
            "positive_clips":len(set(clips[m & truth[:,e]])),"negative":int((m & ~truth[:,e]).sum()),
            "negative_clips":len(set(clips[m & ~truth[:,e]]))} for e,n in enumerate(EVENTS)}}
    groups={"pooled":np.ones(len(truth),dtype=bool),**{s:row_sites==s for s in sorted(set(row_sites.tolist()))}}
    support_out={g:support(m) for g,m in groups.items()}
    return keys,sites,hist,subset,truth,groups,support_out


def confusion(y,pred):
    return {"tp":int((y & pred).sum()),"fp":int((~y & pred).sum()),"fn":int((y & ~pred).sum()),
            "tn":int((~y & ~pred).sum())}


def grouped_counts(truth,flags,top,groups):
    """flags [models, windows, modes, events]; top holds saved maximum-probability indices."""
    assert flags.shape==(2,len(truth),6,5) and top.shape==(2,len(truth)) and (top>=0).all() and (top<6).all()
    any_mode=flags.any(axis=2)
    best=flags[np.arange(2)[:,None],np.arange(len(truth))[None,:],top]
    out={}
    for site,use in groups.items():
        out[site]={}
        for e,n in enumerate(EVENTS):
            y=truth[use,e]
            per_arm={}
            for a,arm in enumerate(NAMES):
                p,b=any_mode[a,use,e],best[a,use,e]
                assert not (b & ~p).any()
                per_arm[arm]={"top1":confusion(y,b),"any_of_six":confusion(y,p),
                              "positive_with_candidate_but_top1_missed":int((y & p & ~b).sum())}
            p,q=any_mode[:,use,e]
            both={"both_no_candidate":int((y & ~p & ~q).sum()),"M0_only":int((y & p & ~q).sum()),
                  "M1_only":int((y & ~p & q).sum()),"both_with_candidate":int((y & p & q).sum())}
            assert sum(both.values())==int(y.sum())
            out[site][n]={"models":per_arm,"paired_any_on_true_positives":both}
    return out


def read_forecasts(repo,work,keys,sites,hist,selection,label):
    # This function is only called when the approved analysis protocol matches its expected SHA.
    original=json.loads((repo/"results/pinns_heldout_replay.json").read_bytes())
    candidates,parent,ranked=[work/p for p in ("anchored_six_candidates.zip","shortpose_heldout_five_seeds.zip","reranked_modes.zip")]
    for p,h in zip((candidates,parent,ranked),(CANDIDATES,PARENT,RERANK)):
        assert sha(p.read_bytes())==h
    retained=np.flatnonzero(selection)
    flags=np.empty((2,len(retained),6,5),bool);choice=np.empty((2,len(retained)),int)
    with ZipFile(candidates) as z,ZipFile(parent) as weights,ZipFile(ranked) as rr:
        reg=json.loads(z.read("run.json"))["registration"]; man=json.loads(z.read("sha256.json"))
        assert reg["archive_sha256"]==PARENT and reg["ordered_keys_sha256"]==ordered_keys(keys)
        assert json.loads(z.read("scores.json"))["training_seed"]==42
        with np.load(io.BytesIO(rr.read("reranked_paired_windows.npz")),allow_pickle=False) as r:
            assert keys_of(r)==keys and r["models"].tolist()==list(NAMES)
            indices=r["mode_indices"][r["seeds"].tolist().index(42),:,:,r["index_columns"].tolist().index("network_top1")]
        for s in sorted(set(sites.tolist())):
            ix=np.flatnonzero(sites==s); kept=np.flatnonzero(selection[ix]); assigned=np.searchsorted(retained,ix[kept])
            k=[keys[i] for i in ix];assert ordered_keys(k)==reg["folds"][s]["test_key_sha256"]
            member=s+"/heldout_private.npz"; raw=z.read(member);assert sha(raw)==man[member]
            with np.load(io.BytesIO(raw),allow_pickle=False) as a:
                assert keys_of(a)==k
                for m,name in enumerate(NAMES):
                    assert sha(weights.read("seed_42/"+s+"/"+name+".pt"))==original["sites"][s]["checkpoint_sha256"][name]
                    prefix="original_"+name+"_"
                    paths=a[prefix+"paths"][kept];pr=a[prefix+"probabilities"][kept]
                    assert paths.shape==(len(kept),6,30,2) and pr.shape==(len(kept),6)
                    assert np.isfinite(paths).all() and np.isfinite(pr).all() and (pr>=0).all()
                    np.testing.assert_allclose(pr.sum(1),1,rtol=0,atol=1e-6)
                    best=pr.argmax(1);assert np.array_equal(best,indices[m,ix[kept]])
                    choice[m,assigned]=best
                    for n,i in enumerate(kept):
                        flags[m,assigned[n]]=[label(hist[ix[i]],path) for path in paths[n]]
    return flags,choice


def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo",type=Path);p.add_argument("work",type=Path);p.add_argument("out",type=Path)
    p.add_argument("--support-only",action="store_true")
    a=p.parse_args()
    protocol=a.repo/"reviews/pinns_short_displacement/event_coverage/PROTOCOL.md"
    if sha(protocol.read_bytes())!=PROTOCOL:
        raise ValueError("Expected the registered 323-window protocol from commit 86eeadd")
    label=rules(a.repo)
    keys,sites,hist,selection,truth,groups,support=mask_and_support(a.repo,a.work,label)
    doc={"ordered_first_short_keys_sha256":FIRST_SHORT,"history_sha256":HISTORY,"support":support}
    if not a.support_only:
        f,top=read_forecasts(a.repo,a.work,keys,sites,hist,selection,label)
        doc.update(protocol_sha256=PROTOCOL,archive_sha256=CANDIDATES,rerank_sha256=RERANK,
                   weight_parent_sha256=PARENT,seed=42,models=list(NAMES),results=grouped_counts(truth,f,top,groups))
    a.out.write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    run()
