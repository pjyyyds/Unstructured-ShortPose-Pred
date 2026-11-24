"""Audit original observations, stored candidate paths and site-cluster summaries. No model forward or training."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

SEEDS = (42,1024,2025,3407,4096)
LENGTHS = (5,10,20)
MODELS = ('M0','M1','M1b','M1-concat')
BINS = ('clear','partial','heavy')
SCORES = ('ade_m','fde_m')
def sha(data): return hashlib.sha256(data).hexdigest()
def keys(z): return list(zip(z['clip'].tolist(),z['person_id'].tolist(),z['t0'].tolist()))
def pair(x): return {m:v.tolist() for m,v in zip(SCORES,x)}

def labels(raw, t):
    box=raw['hist_bbox'][:, -t:, 3]
    frame_valid=box>1
    height=raw['img_h'].astype(np.float64)
    nr=frame_valid.sum(1)
    conf=raw['hist_kpts'][:, -t:, :, 2]
    assert raw['pose_inferred'][:,-t:].all() and np.isfinite(conf).all()
    conf=np.clip(conf,0.,1.)
    good=(conf>=0.2)&frame_valid[:,:,None]
    n=good.sum((1,2))
    sum_conf=(conf*good).sum((1,2),dtype=np.float32)
    q1=(n/(t*17)).astype(np.float32).astype(float)
    q2=(sum_conf/(n+1e-6)).astype(np.float32).astype(float)
    scaled=np.clip(box.astype(float)/height[:,None],0,1)
    q3=np.divide((scaled*frame_valid).sum(1),nr,where=nr>0,out=np.zeros(len(n))).astype(np.float32).astype(float)
    def classify(h):
        rules=np.stack(((q1>=.85)&(q2>=.6)&(h>=.08),
                ((q1>=.50)&(q1<.85))|((q1>=.85)&(q2>=.40)&(q2<.60)&(h>=.04)),
                (q1<.50)|(q2<.40)|(h<.04)),1)
        op=np.where(rules[:,0],0,np.where(rules[:,2],2,1));return rules,op
    rules,op=classify(q3)
    # Audit the common alternate calculation, averaging float32 scaled boxes before storing q3.
    low=(np.minimum(box/height.astype(np.float32)[:,None],1)*frame_valid).sum(1,dtype=np.float32)
    q3low=np.divide(low,nr,where=nr>0,out=np.zeros(len(n))).astype(np.float32).astype(float)
    _,other=classify(q3low)
    drift=[]
    for i in np.flatnonzero(op!=other):
        drift.append(dict(key=keys(raw)[i],valid_joints=int(n[i]),possible_joints=17*t,
            sum_valid_confidence_f32=float(sum_conf[i]),q1=float(q1[i]),q2=float(q2[i]),
            image_height=float(height[i]),box_heights_pixels=box[i].tolist(),
            sum_box_heights_f64=float(box[i].astype(float).sum()),
            normalized_mean_f64=float(scaled[i].mean()),
            canonical_q3_f32_as_double=float(q3[i]),alternate_q3_f32_as_double=float(q3low[i]),
            canonical=BINS[op[i]],alternate=BINS[other[i]]))
    return op,dict(literal_match_counts={str(m):int(np.sum(rules.sum(1)==m)) for m in range(4)},
                   low_precision_related_changed_windows=drift,threshold_008_side_changes=int(np.sum((q3>=.08)!=(q3low>=.08))))

def extract(work,cache,sites,keys_in_order,reference):
    locations=sorted(set(sites));ref=reference['provenance']
    err=np.full((5,3,4,len(sites),2),np.nan);provenance={};maxgap=0.;count={'from_saved_paths':0,'from_error_only_checkpoints':0}
    def consume(archive,seed,ts):
        nonlocal maxgap
        with zipfile.ZipFile(io.BytesIO(archive)) as arc:
            if 'cpu_T20.zip' in arc.namelist():
                for member in ('cpu_T20.zip','matched_fusion_cpu_T20.zip'):consume(arc.read(member),seed,(20,))
                return
            member='folds.json' if 'folds.json' in arc.namelist() else 'fold_ids.json'
            folds=json.loads(arc.read(member))
            for path in arc.namelist():
                if not path.endswith('.pt'):continue
                bits=path.split('/');site,arm=bits[-2],Path(bits[-1]).stem
                t=int(bits[-3][1:]) if len(bits)>2 else ts[0]
                assert t in ts and site in locations and arm in MODELS
                kg=folds[site];ids=kg.get('test_keys',kg.get('test_ids'))
                pos={k:i for i,k in enumerate(keys_in_order)};rows=[pos[tuple(k)] for k in ids]
                assert len(set(rows))==len(rows) and set(rows)==set(np.flatnonzero(sites==site))
                ck=torch.load(io.BytesIO(arc.read(path)),weights_only=True,map_location='cpu')
                rec=ck.get('predictions',ck);observed=rec['errors'].numpy().astype(float)
                ss,tt,aa=SEEDS.index(seed),LENGTHS.index(t),MODELS.index(arm)
                assert np.isnan(err[ss,tt,aa,rows]).all()
                if 'trajs' in rec:
                    p=rec['trajs'].numpy().astype(float);q=rec['probs'].numpy()
                    selected=p[np.arange(len(rows)),np.argmax(q,axis=1)]
                    y=cache['xy'][rows,20:].astype(np.float32).astype(float)
                    errvector=selected-y
                    d=np.hypot(errvector[:,:,0],errvector[:,:,1])
                    recomputed=np.stack((d.mean(1),d[:,-1]),1)
                    gap=float(np.max(np.abs(recomputed-observed)));maxgap=max(gap,maxgap)
                    assert gap<2e-5,(path,gap)
                    count['from_saved_paths']+=1
                else:
                    # The 24 seed-42 M0/M1 base checkpoints only saved top1 errors.
                    count['from_error_only_checkpoints']+=1
                err[ss,tt,aa,rows]=observed
    for seed in SEEDS:
        if seed==42:
            bags=[(f'cpu_T{t}.zip',(t,),ref['seed42_sources_T5_T10'][str(t)]['base']['archive_sha256']) for t in (5,10)]
            bags += [(f'matched_fusion_cpu_T{t}.zip',(t,),ref['seed42_sources_T5_T10'][str(t)]['fusion']['archive_sha256']) for t in (5,10)]
            bags += [('matched_same_worker_T20.zip',(20,),ref['seed42_source_T20']['bundle_sha256'])]
        else:
            bags=[(f'multiseed_s{seed}_all.zip',LENGTHS,ref['verified_archives_and_replay'][str(seed)][0]['archive_sha256'])]
        for file,ts,expected in bags:
            original=(work/file).read_bytes();assert sha(original)==expected
            provenance[file]=expected;consume(original,seed,ts)
    assert np.isfinite(err).all() and sum(count.values())==240
    return err,dict(source_sha256=provenance,checkpoint_counts=count,maximum_path_vs_stored_error_m=maxgap)

def summarize(errors,cv,subgroups,site_names,clips):
    means=[];cvmeans=[];by_site={};counts={}
    boot=np.zeros((20000,4));rng=np.random.Generator(np.random.PCG64(2026))
    for site in site_names:
        ix=subgroups[site];cs=sorted(set(clips[ix]));counts[site]={'windows':len(ix),'clips':len(cs)}
        if not len(ix):continue
        e=errors[:,:,ix,:];m=np.stack([np.stack([x.mean(0) for x in seed]) for seed in e])
        means.append(m);cvmeans.append(cv[ix].mean(0))
        by_site[site]=dict(zip(MODELS,map(pair,m.mean(0))));by_site[site]['last_pair_cv']=pair(cvmeans[-1])
        paired=np.concatenate((e[:,1].mean(0)-e[:,0].mean(0),e[:,1].mean(0)-e[:,2].mean(0)),axis=1)
        sums=np.array([paired[clips[ix]==c].sum(0) for c in cs]);ns=np.array([np.sum(clips[ix]==c) for c in cs])
        draw=rng.integers(0,len(cs),(20000,len(cs)))
        # Multiplicities by replicate and cluster; denominator follows resampled windows.
        hits=np.zeros((len(draw),len(cs)),dtype=np.int16)
        for j in range(len(cs)):hits[:,j]=(draw==j).sum(1)
        boot+=((hits@sums)/(hits@ns)[:,None])/len(site_names)
    cell=dict(windows=sum(v['windows'] for v in counts.values()),clips=sum(v['clips'] for v in counts.values()),counts=counts,by_site=by_site,four_site=None)
    if len(means)==len(site_names):
        per_seed=np.array(means).mean(0);m=per_seed.mean(0)
        ps={str(s):dict(zip(MODELS,map(pair,per_seed[k]))) for k,s in enumerate(SEEDS)}
        allm=dict(zip(MODELS,map(pair,m)));allm['last_pair_cv']=pair(np.mean(cvmeans,0))
        ci=np.quantile(boot,[.025,.975],axis=0).T;cmp={}
        for i,ai in enumerate((0,2)):
            cmp['M1_minus_'+MODELS[ai]]=dict(mean=pair(m[1]-m[ai]),clip_95=pair(ci[i*2:i*2+2]))
        cell['four_site']=dict(models=allm,paired=cmp,per_seed=ps)
    return cell

def verify(work,repo):
    with zipfile.ZipFile(work/'cpu_T5.zip') as archive:
        raw_history=archive.read('matched_observations.npz')
    assert sha(raw_history)=='59e59e3dfa6a2b5583ce4c3fbcb010866e793a3595e43cf691dbefbfcddc64f1'
    assert sha((work/'pinns_matched_histories.npz').read_bytes())=='caa8f0959f28f2d69a6f16cc9df115ab6222d0b6eb3297edd575615fef81b94c'
    with np.load(io.BytesIO(raw_history),allow_pickle=False) as d:
        cache={k:d[k] for k in ('clip','person_id','t0','img_h','hist_bbox','hist_kpts','pose_inferred','xy')}
    order=keys(cache);assert len(order)==len(set(order))==623
    with np.load(work/'pinns_matched_histories.npz',allow_pickle=False) as c:
        km=keys(c);pos={k:i for i,k in enumerate(km)}
        ix=[pos[k] for k in order];assert set(ix)==set(np.flatnonzero(c['strict_disjoint_20']))
        sites=c['site'][ix];assert c['history_20_mask'][ix].all()
        np.testing.assert_array_equal(c['history_20_xy'][ix],cache['xy'][:,:20])
        np.testing.assert_array_equal(c['future_30_xy'][ix],cache['xy'][:,20:])
    clips=cache['clip'];site_names=sorted(set(sites));assert len(site_names)==4 and len(set(clips))==47
    orig=json.loads((repo/'results/pinns_matched_fusion_five_seed.json').read_bytes())
    values,metadata=extract(work,cache,sites,order,orig)
    pred=cache['xy'][:,19:20]+np.diff(cache['xy'][:,18:20],axis=1)*np.arange(1,31)[None,:,None]
    e=pred-cache['xy'][:,20:];d=np.hypot(e[:,:,0],e[:,:,1]);cv=np.stack((d.mean(1),d[:,-1]),1)
    r=dict(identity=metadata,windows=len(order),clips=len(set(clips)),sites=len(site_names),by_duration={},transitions={},pooled_transitions={},history_validation={})
    roster={}
    for i,t in enumerate(LENGTHS):
        roster[t],qc=labels(cache,t);r['history_validation'][str(t)]=qc
        groups={}
        for g,n in enumerate((*BINS,'overall')):
            mask=roster[t]==g if n!='overall' else np.ones(len(clips),bool)
            groups[n]=summarize(values[:,i],cv,{s:np.flatnonzero((sites==s)&mask) for s in site_names},site_names,clips)
        r['by_duration'][str(t)]=dict(literal_match_counts=qc['literal_match_counts'],groups=groups)
    for a,b in ((5,10),(5,20),(10,20)):
        pairname=f'{a}_to_{b}';m={}
        for s in site_names:
            ix=np.flatnonzero(sites==s);ct=Counter(zip(roster[a][ix],roster[b][ix]))
            m[s]=[[ct[i,j] for j in range(3)] for i in range(3)]
        r['transitions'][pairname]=m
        total=np.array(list(m.values())).sum(0);r['pooled_transitions'][pairname]=dict(matrix=total.tolist(),changed=int(total.sum()-np.trace(total)))
    return r

def compare(result,actual):
    diffs=[]
    def rec(x,y):
        if isinstance(x,dict):
            assert set(x)==set(y),(x.keys(),y.keys())
            for k in x:rec(x[k],y[k])
        elif x is None or isinstance(x,str):assert x==y
        else:
            a,b=np.asarray(x),np.asarray(y);assert a.shape==b.shape
            delta=np.max(np.abs(a-b),initial=0);diffs.append(delta)
            if a.dtype.kind in 'iu':assert delta==0
            else:assert delta<2e-10,delta
    for n in ('by_duration','transitions'):rec(result[n],actual[n])
    return {'numeric_comparisons':len(diffs),'maximum_absolute_difference':float(max(diffs))}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('repo',type=Path);p.add_argument('output',type=Path);p.add_argument('--compare',type=Path);p.add_argument('--private-threshold-rows',action='store_true',help='include keys only in a privately held report')
    a=p.parse_args();r=verify(a.work,a.repo)
    if a.compare:r['comparison_with_production']=compare(r,json.loads(a.compare.read_bytes()))
    if not a.private_threshold_rows:
        for entry in r['history_validation'].values():
            for row in entry['low_precision_related_changed_windows']:
                row.pop('key');row.pop('box_heights_pixels')
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(r['pooled_transitions']);print(r['identity']['checkpoint_counts'],r['identity']['maximum_path_vs_stored_error_m'])
    for t,v in r['by_duration'].items():
        print(t,[(k,g['windows'],g['four_site']['paired']['M1_minus_M0']) for k,g in v['groups'].items() if k!='overall'])
