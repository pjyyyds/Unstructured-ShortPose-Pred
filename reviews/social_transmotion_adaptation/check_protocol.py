"""Pinned, inspectable PINNS horizon/adapter check. No training or real forecasting."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import tokenize

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
SOURCE='87ab7809431d943f1b94a62d92323641e2474d5030ca6569ef11888e08cc78f4'
POSE='b493a47536664ddb4fd66603c1bbf6b2c2e33f553520382027b3ae96b53447d0'
LABELS='0b276b34ab1b01dae91f67c56bfaea5cc931511b05b4fc1c92686a891e6f6801'
ORDER='f5ee511a54ad818499d3dde6213f7f3b49332943ff2fc93d93aed87d316dbb91'


def model_class(source, adapted):
    """Original verified JRDB file. Only its numeric temporal extents change for PINNS."""
    raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE: raise ValueError('Wrong upstream model source')
    if adapted:
        # Each affected executable 9/21 is a time extent (see audit); keep other integers and comments.
        subst={'9':'5','21':'35'}
        tokens=[]
        for tok in tokenize.tokenize(io.BytesIO(raw).readline):
            if tok.type==tokenize.NUMBER and tok.string in subst: tok=tok._replace(string=subst[tok.string])
            tokens.append(tok)
        raw=tokenize.untokenize(tokens)
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'model_jrdb.py';p.write_bytes(raw)
        spec=importlib.util.spec_from_file_location('pinned_transmotion',p)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.TransMotion,hashlib.sha256(raw).hexdigest()


def build_observed(history,center_box,box_on):
    """Target only; neither future ground truth nor future neighbor identities are accepted."""
    if history.ndim!=3 or history.shape[1:]!=(5,2) or center_box.shape!=history.shape[:-1]+(4,):
        raise ValueError('Expected five measured ground positions and xywh boxes')
    if not np.isfinite(history).all() or not np.isfinite(center_box).all():
        raise ValueError('Nonfinite history')
    if box_on and not (center_box[:,:,2:]>0).all(): raise ValueError('Missing observed box')
    tokens=np.zeros((len(history),5,2,4),dtype=np.float32)
    tokens[:,:,0,:2]=history-history[:,-1:]
    if box_on:
        ltwh=center_box.copy()
        ltwh[:,:,:2]-=.5*center_box[:,:,2:]
        tokens[:,:,1,:]=.25*(ltwh-ltwh[:,-1:])
    return tokens


def smoke(source):
    torch.manual_seed(17)
    torch.set_num_threads(2)
    res={}
    for adapted,history,future in [(False,9,12),(True,5,30)]:
        cls,code=model_class(source,adapted)
        net=cls(tok_dim=history*2+future,nhid=128,nhead=4,dim_feedfwd=1024,nlayers_local=6,
                nlayers_global=3,dropout=.1,output_scale=1,obs_and_pred=history+future,
                num_tokens=2,device='cpu').eval()
        # 2 synthetic people for native; one target for the cohort-matched adaptation.
        people=1 if adapted else 2
        x=torch.randn(2,history,2*people,4)
        mask=torch.zeros(2,people,dtype=torch.bool)
        with torch.no_grad(): y=net(x,mask)
        assert y.shape==(2,history+future,1,2) and torch.isfinite(y).all()
        res['pinns_adapted' if adapted else 'native']=dict(code_sha256=code,synthetic_people=people,
            input_shape=list(x.shape),full_shape=list(y.shape),future_shape=list(y[:,history:].shape),
            parameters=sum(v.numel() for v in net.parameters()))
    res['torch']=str(torch.__version__)
    return res


def data_check(metadata,pose):
    from scripts.benchmark_heldout_sites import load_site_windows
    from scripts.run_heldout_seed_sensitivity import json_digest,SITE_COUNTS
    if hashlib.sha256(pose.read_bytes()).hexdigest()!=POSE: raise ValueError('Wrong pose/box file')
    groups,_,sha=load_site_windows(metadata)
    assert sha==LABELS
    keys=[(r.clip,r.person_id,r.t0) for s in sorted(groups) for r in groups[s]]
    assert json_digest(keys)==ORDER and {s:len(v) for s,v in groups.items()}==SITE_COUNTS
    with np.load(pose,allow_pickle=False) as data:
        cache=list(zip(data['clip'].tolist(),data['person_id'].tolist(),data['t0'].tolist()))
        assert len(set(cache))==len(keys) and set(cache)==set(keys)
        lookup={k:i for i,k in enumerate(cache)};rows=[lookup[k] for k in keys]
        h=np.stack([r.xy[:5] for s in sorted(groups) for r in groups[s]])
        np.testing.assert_array_equal(h,data['hist_pos_relative'][rows])
        boxes=data['hist_bbox'][rows]
        valid=(boxes[...,2:]>0).all(-1)
    motion=build_observed(h,boxes,False)
    assert not motion[:,:,1,:].any() and not motion[:,-1].any()
    with_boxes=build_observed(h[valid.all(1)],boxes[valid.all(1)],True)
    assert not with_boxes[:,-1].any()
    return dict(source_label_sha256=sha,source_pose_sha256=POSE,ordered_keys_sha256=ORDER,site_windows=SITE_COUNTS,
                complete_box_windows=int(valid.all(1).sum()),valid_history_boxes=int(valid.sum()),
                motion_tokens_shape=list(motion.shape),complete_box_tokens_shape=list(with_boxes.shape),
                real_model_forwards=0,real_training_steps=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'source/model_jrdb.py')
    parser.add_argument('--metadata',type=Path);parser.add_argument('--pose',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'compatibility.json')
    args=parser.parse_args()
    if bool(args.metadata)!=bool(args.pose):parser.error('Supply both metadata and pose for a cohort check')
    report={'synthetic':smoke(args.source)}
    if args.metadata:report['cohort']=data_check(args.metadata,args.pose)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
