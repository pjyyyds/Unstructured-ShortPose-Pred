"""Score already archived candidate coordinates and forecast errors without a model forward."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from src.data.actions import compute_action_labels

EVENTS=('start','stop','left','right','maintain')
ARMS=('M0','M1')
SEEDS=(42,1024,2025,3407,4096)
HISTORY='caa8f0959f28f2d69a6f16cc9df115ab6222d0b6eb3297edd575615fef81b94c'
RERANK='b24ac2bb1ad3d6742be2cbe16b927e42ef9a1a54a17ce2c08be3e92fa286b871'
CANDIDATES='388920f757f0f76d688f281e232d20f2c1afe4178e50771d6f9d0f6146f61748'
PARENT='0bf2745ab2d5ddcf5f5fbb5d330e6c6cc800625cec301f4c7105d031bdb49d2d'
PROTOCOL='de8abe39b12d91e7db452718ec6f3f1b736a6d6606951f9243446eb2cc004328'
GEOMETRY=('top1_ADE_m','top1_FDE_m','minADE6_m','minFDE6_m','minADE_candidate_FDE_m','top1_minus_minADE_m','top1_is_minADE')
FIVE=('top1_ADE_m','top1_FDE_m','minADE6_m','minADE_candidate_FDE_m','top1_minus_minADE_m','top1_is_minADE')
H_MASK=np.ones(5);F_MASK=np.ones(30)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def jhash(keys):
    return sha(json.dumps(keys,separators=(',',':')).encode())


def ids(z):
    return list(zip(z['clip'].tolist(),z['person_id'].tolist(),z['t0'].tolist()))


def require(a,description):
    if not a:raise ValueError(description)


def near(a,b,atol,rtol=0):
    np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)
    return float(abs(np.array(a)-np.array(b)).max(initial=0))


def events(history,paths):
    vals=np.zeros((len(history),paths.shape[1],5),bool)
    for i,(xp,fs) in enumerate(zip(history,paths)):
        for k,f in enumerate(fs):
            lab,valid=compute_action_labels(xp,H_MASK,f,F_MASK)
            require(valid==1.,'Incomplete trajectory')
            vals[i,k]=lab
    return vals


def counts(top,emits,mask):
    e=emits[mask]
    return {'windows':len(e),'top1_event_windows':int(top[mask].sum()),'any_event_windows':int(e.any(1).sum()),
            'zero_event_windows':int((~e.any(1)).sum()),'event_candidate_flags':int(e.sum()),'candidate_denominator':int(e.size)}


def fractions(top,emits,member,pos):
    pp=member&pos;nn=member&~pos
    out={}
    for title,mask in [('positive',pp),('negative',nn),('all',member)]:
        c=counts(top,emits,mask);num=c['windows']
        out[title]=dict(counts=c,top1_rate=float(c['top1_event_windows']/num) if num else None,
                        any_rate=float(c['any_event_windows']/num) if num else None,
                        zero_rate=float(c['zero_event_windows']/num) if num else None,
                        candidate_fraction=float(c['event_candidate_flags']/c['candidate_denominator']) if num else None)
    return out


def bootstrap(d,positive,sites,clips):
    rng=np.random.default_rng(2026);draw=[];per_site={}
    for s in sorted(set(sites)):
        mask=(sites==s)&positive;u=np.unique(clips[mask]);n=len(u)
        if not n:
            per_site[s]=None;continue
        units=[np.flatnonzero(mask&(clips==c)) for c in u]
        indices=rng.integers(n,size=(20000,n))
        mult=np.stack([(indices==i).sum(1) for i in range(n)],axis=1)
        denominator=mult @ np.array([len(rows) for rows in units])
        rep=(mult @ np.stack([d[idx].sum(axis=0) for idx in units]))/denominator[:,None]
        draw.append(rep);per_site[s]=d[mask].mean(0).tolist()
    if len(draw)!=len(set(sites)):return dict(by_site=per_site,four_site_equal=None)
    mean=np.mean(list(per_site.values()),axis=0)
    q=np.quantile(np.mean(draw,0),[.025,.975],axis=0).T
    return dict(by_site=per_site,four_site_equal=dict(zip(('any_event_recall_M1_minus_M0_pp','top1_ADE_M1_minus_M0_m'),
                    [dict(estimate=float(a),paired_clip_95=b.tolist()) for a,b in zip(mean,q)])))


def assemble(args):
    md=args.repo/'reviews/pinns_event_coverage/PROTOCOL.md'
    require(sha(md.read_bytes())==PROTOCOL,'Unregistered protocol')
    paths={k:args.input/n for k,n in [('history','pinns_matched_histories.npz'),('rerank','reranked_modes.zip'),('candidates','anchored_six_candidates.zip'),('parent','shortpose_heldout_five_seeds.zip')]}
    require({k:sha(p.read_bytes()) for k,p in paths.items()}==dict(history=HISTORY,rerank=RERANK,candidates=CANDIDATES,parent=PARENT),'Input bytes changed')
    with np.load(paths['history'],allow_pickle=False) as z:
        keys=ids(z);sites=z['site'];hist=z['history_20_xy'][:,-5:];future=z['future_30_xy']
        require(z['history_20_mask'][:,-5:].all() and len(set(keys))==len(keys)==1482,'Incomplete history')
    clips=np.array([k[0] for k in keys]);names=sorted(set(sites));assert len(names)==4 and len(set(clips))==47
    with ZipFile(paths['rerank']) as rz:
        require(rz.testzip() is None,'Forecast chain CRC')
        raw=rz.read('reranked_paired_windows.npz');rrep=json.loads(rz.read('reranked_mode_scores.json'))
        require(sha(raw)==rrep['paired_windows_sha256'] and rrep['archive_sha256']==PARENT,'Forecast chain source')
        with np.load(io.BytesIO(raw),allow_pickle=False) as q:
            require(ids(q)==keys and tuple(q['seeds'])==SEEDS and tuple(q['models'])==ARMS and q['metrics'].tolist()==['ade_m','fde_m'],'Forecast schema or keys')
            method=q['methods'].tolist();idxname=q['index_columns'].tolist()
            t,o,c=[method.index(k) for k in ('network_top1','oracle_minade_mode','last_two_velocity')]
            ix=q['mode_indices'];err=q['errors'];top_indices=ix[:,:,:,idxname.index('network_top1')];or_indices=ix[:,:,:,idxname.index('oracle_minade_mode')]
            require(err.shape==(5,2,len(keys),4,2) and np.isfinite(err).all(),'Archived error shape')
            eq=top_indices==or_indices
            five=np.stack((err[:,:,:,t,0],err[:,:,:,t,1],err[:,:,:,o,0],err[:,:,:,o,1],err[:,:,:,t,0]-err[:,:,:,o,0],eq),axis=-1)
            cv=err[0,0,:,c];require(np.all(err[:,:,:,c]==cv),'Direct velocity disagrees among seeds')
    official=json.loads((args.repo/'results/pinns_heldout_implied_actions.json').read_bytes())
    modes=json.loads((args.repo/'results/pinns_heldout_mode_scores.json').read_bytes())
    orig=json.loads((args.repo/'results/pinns_heldout_replay.json').read_bytes())
    truth=events(hist,future[:,None])[:,0]
    extras={};n=len(keys)
    for e,name in enumerate(EVENTS):
        assert truth[:,e].sum()==official['pooled']['classes'][name]['positive_windows']
        assert len(np.unique(clips[truth[:,e]]))==official['pooled']['classes'][name]['clips_with_positives']
    assert (truth.sum(1)>1).sum()==17
    all_geometry=np.empty((2,n,7));all_events=np.empty((2,n,6,5),bool);top_ev=np.empty((2,n,5),bool)
    gpu_mismatch=0.;published_mode_drift=0.;checked_indices=0;all_paths=0
    for s in names:
        for e,name in enumerate(EVENTS):
            mask=(sites==s)&truth[:,e];support=official['sites'][s]['classes'][name]
            assert mask.sum()==support['positive_windows'] and len(set(clips[mask]))==support['clips_with_positives']
    with ZipFile(paths['candidates']) as az,ZipFile(paths['parent']) as parent:
        require(az.testzip() is None and parent.testzip() is None,'Candidate/weight CRC')
        reg=json.loads(az.read('run.json'))['registration'];man=json.loads(az.read('sha256.json'))
        assert reg['archive_sha256']==PARENT and reg['ordered_keys_sha256']==jhash(keys)
        require(json.loads(az.read('scores.json'))['training_seed']==42,'Wrong seed')
        for s in names:
            rows=np.flatnonzero(sites==s);site_keys=[keys[i] for i in rows]
            assert reg['folds'][s]['test_key_sha256']==jhash(site_keys)
            raw=az.read(s+'/heldout_private.npz');assert sha(raw)==man[s+'/heldout_private.npz']
            with np.load(io.BytesIO(raw),allow_pickle=False) as q:
                assert ids(q)==site_keys
                for arm_index,arm in enumerate(ARMS):
                    assert sha(parent.read('seed_42/'+s+'/'+arm+'.pt'))==orig['sites'][s]['checkpoint_sha256'][arm]
                    stem='original_'+arm+'_'
                    p,prob=q[stem+'paths'],q[stem+'probabilities'];nr=len(rows);rr=np.arange(nr)
                    assert p.shape==(nr,6,30,2) and prob.shape==(nr,6) and np.isfinite(p).all() and np.isfinite(prob).all() and (prob>=0).all()
                    near(prob.sum(1),np.ones(nr),1e-6)
                    xt=p.astype(np.float64)-future[rows,None]
                    d=np.hypot(xt[...,0],xt[...,1]);ade=d.mean(-1);fde=d[:,:,-1]
                    choose=prob.argmax(1);oracle=ade.argmin(1)
                    np.testing.assert_array_equal(choose,top_indices[0,arm_index,rows])
                    np.testing.assert_array_equal(oracle,or_indices[0,arm_index,rows]);checked_indices+=2*nr;all_paths+=nr*6
                    geo=np.stack((ade[rr,choose],fde[rr,choose],ade[rr,oracle],fde.min(1),fde[rr,oracle],ade[rr,choose]-ade[rr,oracle],choose==oracle),axis=-1)
                    gpu_mismatch=max(gpu_mismatch,near(geo[:,[0,1,2,4]],five[0,arm_index,rows,:4],2e-4,2e-5))
                    pub=modes['heldout_site_by_seed']['42'][s][arm]
                    published_mode_drift=max(published_mode_drift,near(geo.mean(0)[[0,1,2,3]],[pub[k] for k in ('ADE_1','FDE_1','minADE_6','minFDE_6')],2e-4,2e-5))
                    emit=events(hist[rows],p);all_events[arm_index,rows]=emit;top_ev[arm_index,rows]=emit[rr,choose];all_geometry[arm_index,rows]=geo
                    weighted=np.einsum('nk,nke->ne',prob.astype(float)/prob.sum(1,dtype=float)[:,None],emit)
                    expected=official['sites'][s]['methods'][arm]['per_class']
                    for e,name in enumerate(EVENTS):
                        gt=truth[rows,e];bit=weighted[:,e]>=.5
                        exp=expected[name]
                        for count,val in [('tp',(gt&bit).sum()),('fp',(bit&~gt).sum()),('fn',(gt&~bit).sum())]:
                            assert int(val)==exp[count],(s,arm,name,count)
    assert checked_indices==5928 and all_paths==17784
    result={}
    for e,name in enumerate(EVENTS):
        pos=truth[:,e];event_result={};seed_scores={}
        for a,arm in enumerate(ARMS):
            e6=all_events[a,:,:,e];t1=top_ev[a,:,e]
            grouped={};averages={};fmean=[]
            for site in names:
                member=sites==site;pp=member&pos
                stats=fractions(t1,e6,member,pos)
                gp=all_geometry[a,pp].mean(0);cv_mean=cv[pp].mean(0)
                fout=five[:,a][:,pp,:].mean(axis=1);fmean.append(fout)
                grouped[site]=dict(rates=stats,positive_geometry=dict(zip(GEOMETRY,gp.tolist())),positive_direct_CV_m=cv_mean.tolist(),
                                   five_seed_positive_geometry={str(seed):dict(zip(FIVE,row.tolist())) for seed,row in zip(SEEDS,fout)})
            for which in ('positive','negative','all'):
                for key in ('top1_rate','any_rate','zero_rate','candidate_fraction'):
                    values=[grouped[s]['rates'][which][key] for s in names]
                    averages[which+'_'+key]=float(np.mean(values)) if all(v is not None for v in values) else None
            pooled=fractions(t1,e6,np.ones(n,bool),pos)
            gmacro={k:float(np.mean([grouped[s]['positive_geometry'][k] for s in names])) for k in GEOMETRY}
            seed_equal=np.mean(fmean,0)
            event_result[arm]=dict(seed42_by_site=grouped,seed42_conditional_four_site_rates=averages,
                                  seed42_positive_four_site_geometry=gmacro,seed42_pooled_counts={k:v['counts'] for k,v in pooled.items()},
                                  five_seed_positive_geometry_by_seed={str(seed):dict(zip(FIVE,vec.tolist())) for seed,vec in zip(SEEDS,seed_equal)},
                                  five_seed_positive_geometry=dict(zip(FIVE,seed_equal.mean(0).tolist())))
        support={}
        for s in ['all',*names]:
            member=np.ones(n,bool) if s=='all' else sites==s
            support[s]={title:dict(windows=int(mask.sum()),clips=len(set(clips[mask]))) for title,mask in [('positive',member&pos),('negative',member&~pos)]}
        paired=np.stack(((all_events[1,:,:,e].any(1).astype(float)-all_events[0,:,:,e].any(1))*100,
                         (five[:,1,:,0]-five[:,0,:,0]).mean(0)),axis=1)
        result[name]=dict(support=support,models=event_result,
                          four_site_direct_CV_positive_m=dict(zip(('ade_m','fde_m'),np.mean([cv[(sites==s)&pos].mean(0) for s in names],axis=0).tolist())),
                          paired=bootstrap(paired,pos,sites,clips))
    return dict(frozen_git_commit='6dd78f7a4f77fb1d81603f53c67b8c205053e9f9',protocol_sha256=PROTOCOL,score_script_sha256=sha(Path(__file__).read_bytes()),action_rules_sha256=sha((args.repo/'src/data/actions.py').read_bytes()),
                provenance=dict(six_path_reference_kaggle=373619572,five_seed_errors_kaggle=373583293,history_npz_sha256=HISTORY,
                six_path_archive_sha256=CANDIDATES,five_seed_archive_sha256=RERANK,parent_weight_archive_sha256=PARENT),windows=n,clips=len(set(clips)),sites=names,
                seeds=SEEDS,multilabel_windows=17,verification=dict(seed42_paths_read=all_paths,exact_top1_and_oracle_indices=checked_indices,
                  max_seed42_abs_error_m_against_original_independent_Kaggle=gpu_mismatch,max_seed42_abs_site_metric_error_against_published=published_mode_drift,
                  exact_previously_published_thresholded_event_confusion_counts=4*2*5*3,source_weights_sha_verified=8),
                units='ADE/FDE in metres. Rates/fractions in [0,1], recall contrasts in percentage points. By-site raw numerators and denominators available in seed42_by_site/rates.',
                caveat='Observed sites, archived fits and overlapping true kinematic labels. Five-seed archive does not contain full candidate trajectories or independent minimum FDE. Seed42 only for candidate event consistency and minFDE6.',events=result)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','input','output'):p.add_argument(k,type=Path)
    a=p.parse_args();res=assemble(a);a.output.write_text(json.dumps(res,indent=2,allow_nan=False)+'\n')
    print('Verified:',res['verification'])
