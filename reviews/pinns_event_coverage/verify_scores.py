"""Independent complete-track event reconstruction and clip-resampling check; no model dependency."""
from hashlib import sha256
import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np

CATEGORIES=('start','stop','left','right','maintain')
NAMES=('M0','M1')


def first_run(mask):
    hits=np.flatnonzero(mask[:-2]&mask[1:-1]&mask[2:])
    return int(hits[0]) if len(hits) else None


def label_complete(hist,future):
    p=np.concatenate((hist,future)).astype(float)
    vel=np.diff(p,axis=0)/0.1
    # velocity row 0 ends at original tick 1, so last three historical rows have indices 1..3.
    smooth=np.vstack((vel[0],(vel[0]+vel[1])/2.,(vel[:-2]+vel[1:-1]+vel[2:])/3.))
    speed=np.hypot(smooth[:,0],smooth[:,1])
    before=speed[1:4].mean()
    start=bool(before<=.2 and first_run(speed[4:]>=.6) is not None)
    stop=bool(before>=.6 and first_run(speed[4:]<=.2) is not None)
    left=right=False
    # Reconstruct positive-speed intervals including tick 2; each turns relative to its own start heading.
    moving=speed[1:]>=.5
    boundaries=np.flatnonzero(np.diff(np.r_[False,moving,False].astype(int)))
    for begin,end in boundaries.reshape(-1,2):
        begin+=1;end+=1
        if end-begin<3:continue
        velocity=smooth[begin:end];angles=np.unwrap(np.arctan2(velocity[:,1],velocity[:,0]))
        delta=angles[max(0,4-begin):]-angles[0]
        if len(delta)<3:continue
        l=first_run(delta>=np.pi/6);r=first_run(delta<=-np.pi/6)
        if l is not None or r is not None:
            left=r is None or (l is not None and l<=r);right=not left;break
    return [start,stop,left,right,not(start or stop or left or right)]


def audit(inp,result_path,repo):
    import sys
    sys.path.insert(0,str(repo.resolve()))
    from src.data.actions import compute_action_labels
    def certified_label(h,f):
        own=label_complete(h,f)
        published,valid=compute_action_labels(h,np.ones(5),f,np.ones(30))
        assert valid==1 and np.array_equal(own,published)
        return own
    r=json.loads(result_path.read_bytes());assert r['windows']==1482
    with np.load(inp/'pinns_matched_histories.npz',allow_pickle=False) as c:
        h=c['history_20_xy'][:,-5:];f=c['future_30_xy'];site=c['site'];clip=c['clip']
        keys=list(zip(clip.tolist(),c['person_id'].tolist(),c['t0'].tolist()))
    y=np.array([certified_label(a,b) for a,b in zip(h,f)])
    emit=np.empty((2,1482,6,5),bool);selected=np.empty((2,1482),int);geo=np.empty((2,1482,7))
    with ZipFile(inp/'anchored_six_candidates.zip') as z:
        for sn in sorted(set(site)):
            idx=np.flatnonzero(site==sn)
            with np.load(io.BytesIO(z.read(sn+'/heldout_private.npz')),allow_pickle=False) as q:
                assert [keys[i] for i in idx]==list(zip(q['clip'].tolist(),q['person_id'].tolist(),q['t0'].tolist()))
                for a,name in enumerate(NAMES):
                    arr=q['original_'+name+'_paths'];which=q['original_'+name+'_probabilities'].argmax(1)
                    delta=np.linalg.norm(arr-f[idx,None],axis=3)
                    per=delta.mean(2);imin=per.argmin(1);rr=np.arange(len(idx))
                    geo[a,idx]=np.column_stack((per[rr,which],delta[rr,which,-1],per.min(1),delta[:,:,-1].min(1),delta[rr,imin,-1],per[rr,which]-per.min(1),which==imin))
                    selected[a,idx]=which
                    for j,i in enumerate(idx):
                        for k in range(6):emit[a,i,k]=certified_label(h[i],arr[j,k])
    with ZipFile(inp/'reranked_modes.zip') as z,np.load(io.BytesIO(z.read('reranked_paired_windows.npz')),allow_pickle=False) as q:
        kk=q['methods'].tolist();ii=q['index_columns'].tolist();tt=kk.index('network_top1');oo=kk.index('oracle_minade_mode')
        x=q['errors'];cv=x[0,0,:,kk.index('last_two_velocity')];ind=q['mode_indices'];seeds=q['seeds'].tolist()
        metrics=np.stack([x[:,:,:,tt,0],x[:,:,:,tt,1],x[:,:,:,oo,0],x[:,:,:,oo,1],x[:,:,:,tt,0]-x[:,:,:,oo,0],ind[:,:,:,ii.index('network_top1')]==ind[:,:,:,ii.index('oracle_minade_mode')]],-1)
    peak=0.;ncheck=0
    def ck(actual,expected):
        nonlocal peak,ncheck
        d=abs(np.asarray(actual)-np.asarray(expected));peak=max(peak,d.max(initial=0));ncheck+=d.size
        np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-12)
    geom=('top1_ADE_m','top1_FDE_m','minADE6_m','minFDE6_m','minADE_candidate_FDE_m','top1_minus_minADE_m','top1_is_minADE')
    five=('top1_ADE_m','top1_FDE_m','minADE6_m','minADE_candidate_FDE_m','top1_minus_minADE_m','top1_is_minADE')
    for e,name in enumerate(CATEGORIES):
        item=r['events'][name];pos=y[:,e];rng=np.random.Generator(np.random.PCG64(2026));bs=[]
        for s in sorted(set(site)):
            eligible=(site==s)&pos
            u=np.unique(clip[eligible]);blocks=[eligible&(clip==c) for c in u]
            d=np.column_stack(((emit[1,:,:,e].max(1).astype(int)-emit[0,:,:,e].max(1))*100,(metrics[:,1,:,0]-metrics[:,0,:,0]).mean(0)))
            sums=np.stack([d[m].sum(0) for m in blocks]);size=np.array([m.sum() for m in blocks])
            sampled=rng.integers(len(u),size=(20000,len(u)));bs.append(sums[sampled].sum(1)/size[sampled].sum(1)[:,None])
            for a,model in enumerate(NAMES):
                g=item['models'][model]['seed42_by_site'][s]
                for title,mask in [('positive',eligible),('negative',(site==s)&~pos),('all',site==s)]:
                    b=emit[a,mask,:,e];rr=np.arange(mask.sum());cn=g['rates'][title]['counts']
                    assert cn==dict(windows=len(b),top1_event_windows=int(b[rr,selected[a,mask]].sum()),any_event_windows=int((b.sum(1)>0).sum()),zero_event_windows=int((b.sum(1)==0).sum()),event_candidate_flags=int(b.sum()),candidate_denominator=b.size)
                ck(geo[a,eligible].mean(0),[g['positive_geometry'][k] for k in geom])
                ck(cv[eligible].mean(0),g['positive_direct_CV_m'])
                for i,seed in enumerate(seeds):
                    ck(metrics[i,a,eligible].mean(0),[g['five_seed_positive_geometry'][str(seed)][k] for k in five])
        ci=np.quantile(np.mean(bs,0),[.025,.975],axis=0).T
        ck(ci,[item['paired']['four_site_equal'][k]['paired_clip_95'] for k in ('any_event_recall_M1_minus_M0_pp','top1_ADE_M1_minus_M0_m')])
    return dict(source_result_sha256=sha256(result_path.read_bytes()).hexdigest(),verifier_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),complete_ground_truth_tracks=1482,complete_candidate_tracks=17784,
                independent_per_track_event_bits_equal=(1482+17784)*5,site_event_model_denominator_groups_exact=5*4*2*3,numerical_recomputations=ncheck,max_absolute_geometry_or_interval_difference=float(peak),
                method='Independent complete-track finite-difference velocities, three-frame causal means and sustained-velocity/relative-heading conditions. Compared all 120 site/event/model/truth-stratum counts and all prespecified intervals by direct index resampling, alongside 5-seed site-conditional geometric means.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','input','scores','output'):parser.add_argument(k,type=Path)
    a=parser.parse_args();doc=audit(a.input,a.scores,a.repo);a.output.write_text(json.dumps(doc,indent=2)+'\n');print(doc)
