"""Inference-free scoring for the registered joint-action archive. Actual decisions run only on Kaggle."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from selector import select, eligibility
from src.data.actions import compute_action_labels

HERE=Path(__file__).resolve().parent
SOURCE='b84eaaadcaab4518ded15c83e9d746030e03693e89dfe9e23b15ef2f1fc8587e'
CENSUS='caa8f0959f28f2d69a6f16cc9df115ab6222d0b6eb3297edd575615fef81b94c'
LABELS='0b276b34ab1b01dae91f67c56bfaea5cc931511b05b4fc1c92686a891e6f6801'
KEYS='f5ee511a54ad818499d3dde6213f7f3b49332943ff2fc93d93aed87d316dbb91'
PLAN='85e873537eb09f508b85df64de323d0d994236a64615bc2703d151ec9fb021e3'
SIZES={'america_crossroad':816,'italy_park':84,'japan_crossroad':432,'thailand_street':150}
ARMS=('no_head','auxiliary','end_to_end','detached')
EVENTS=('start','stop','left','right','maintain')
FIELDS=('ade_m','fde_m')


def digest(b):
    return hashlib.sha256(b).hexdigest()


def jhash(value):
    return digest(json.dumps(value,separators=(',',':'),sort_keys=True).encode())


def check(test,message):
    if not test:raise ValueError(message)


def packed_json(path,doc):
    path.write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')


def source_histories(census,metadata):
    if census:
        check(digest(census.read_bytes())==CENSUS,'Unexpected history archive')
        with np.load(census,allow_pickle=False) as q:
            keys=list(zip(q['clip'].tolist(),q['person_id'].tolist(),q['t0'].tolist()))
            check(q['history_20_mask'][:,-5:].all(),'Missing five observed frames')
            return keys,q['site'],q['history_20_xy'][:,-5:],q['future_30_xy']
    from scripts.benchmark_heldout_sites import load_site_windows
    groups,_,raw_digest=load_site_windows(metadata)
    check(raw_digest==LABELS,'Unexpected official labels/calibration')
    windows=[w for s in sorted(groups) for w in groups[s]]
    xy=np.stack([w.xy for w in windows])
    return [(w.clip,w.person_id,w.t0) for w in windows],np.array([s for s in sorted(groups) for _ in groups[s]]),xy[:,:5],xy[:,5:]


def preflight(bundle,repo,census,metadata):
    check(digest((HERE/'PROTOCOL.md').read_bytes())==PLAN and digest(bundle.read_bytes())==SOURCE,'Wrong plan or source predictions')
    source_rules=repo/'src/data/actions.py'
    check(digest(source_rules.read_bytes())=='4b1300c579377a6013355fb6920779ef56972781dd1b548a19f9f8f940af7646','Wrong event rules')
    baseline=json.loads((repo/'results/pinns_individual_joint_actions.json').read_bytes())
    keys,sites,hist,future=source_histories(census,metadata)
    require_counts={s:int((sites==s).sum()) for s in sorted(set(sites))}
    check(jhash(keys)==KEYS and len(set(keys))==1482 and require_counts==SIZES and len({k[0] for k in keys})==47,'Changed cohort')
    with ZipFile(bundle) as z:
        check(z.testzip() is None and len(set(z.namelist()))==len(z.namelist()),'Invalid source ZIP')
        man=json.loads(z.read('sha256.json'));check(set(z.namelist())==set(man)|{'sha256.json'},'Source manifest member list')
        for fn,sha in man.items():check(digest(z.read(fn))==sha,'Changed '+fn)
        plan=json.loads(z.read('preregistration.json'))
        check(plan['ordered_keys_sha256']==KEYS and plan['data_sha256']==LABELS and plan['action_labels']==list(EVENTS),'Unexpected original registration')
        check(set(SIZES)==set(plan['folds']),'Missing fold')
        for s,count in SIZES.items():
            idx=np.flatnonzero(sites==s);other=sorted(set(SIZES)-{s})
            train=[list(keys[i]) for sn in other for i in np.flatnonzero(sites==sn)]
            test=[list(keys[i]) for i in idx];fk=json.loads(z.read(f'{s}/fold_keys.json'))
            check(fk==dict(train_sites=other,train_keys=train,test_keys=test) and jhash(test)==plan['folds'][s]['test_ids_sha256'] and jhash(train)==plan['folds'][s]['train_ids_sha256'],'Wrong held-out identities')
            check(not ({k[0] for k in train}&{k[0] for k in test}),'Clip crossed folds')
            with np.load(io.BytesIO(z.read(f'{s}/targets.npz')),allow_pickle=False) as q:
                check(list(zip(q['clip'].tolist(),q['person_id'].tolist(),q['t0'].tolist()))==[keys[i] for i in idx],'Targets not matched')
                check(q['y_act'].shape==(count,5) and q['m_act'].all() and q['valid_by_class'].all() and q['fut_mask'].all(),'Missing targets')
                np.testing.assert_array_equal(future[idx].astype(np.float32),q['fut_xy'])
            for arm in ARMS:
                with np.load(io.BytesIO(z.read(f'{s}/{arm}.npz')),allow_pickle=False) as q:
                    check(q['trajs'].shape==(count,6,30,2) and q['mode_probs'].shape==(count,6) and q['mode_logits'].shape==(count,6) and q['errors'].shape==(count,2),'Invalid forecast shape')
                    check(('act_probs' in q)==(arm!='no_head') and ('act_logits' in q)==(arm!='no_head'),'Unexpected head')
                    if arm!='no_head':check(q['act_probs'].shape==q['act_logits'].shape==(count,5),'Head shape')
    return keys,sites,hist,future,baseline


def errs(paths,future):
    delta=paths.astype(float)-future.astype(float)
    d=np.hypot(delta[...,0],delta[...,1]);return np.stack((d.mean(-1),d[:,-1]),1)


def vec(v):
    return dict(zip(FIELDS,np.asarray(v).tolist()))


def stats(mask,labels,before,after,flags):
    cond={}
    for j,name in enumerate(EVENTS):
        part={}
        for title,v in [('positive',labels[:,j]),('negative',~labels[:,j])]:
            rows=mask&v;old=before[rows,j];new=after[rows,j]
            part[title]=dict(windows=int(rows.sum()),original_event=int(old.sum()),reranked_event=int(new.sum()),any_mode_event=int(flags[rows,:,j].any(1).sum()),
                event_gained=int((~old&new).sum()),event_lost=int((old&~new).sum()),original_rate=float(old.mean()) if len(old) else None,
                reranked_rate=float(new.mean()) if len(new) else None)
        cond[name]=part
    return cond


def interval(d,sites,clips):
    rng=np.random.default_rng(2026);all_rep=[]
    for s in sorted(set(sites)):
        rows=np.flatnonzero(sites==s);u=np.unique(clips[rows]);units=[rows[clips[rows]==c] for c in u]
        sample=rng.integers(len(u),size=(20000,len(u)))
        sums=np.stack([d[idx].sum(0) for idx in units]);counts=np.array([len(idx) for idx in units])
        all_rep.append(sums[sample].sum(1)/counts[sample].sum(1)[:,None])
    return vec(np.quantile(np.mean(all_rep,0),[.025,.975],axis=0).T)


def main(args):
    if not args.check_only and not Path('/kaggle/working').is_dir():
        raise RuntimeError('Use --check-only locally; real ranking and scoring run only on Kaggle')
    keys,sites,hist,future,baseline=preflight(args.bundle,args.repo,args.census,args.metadata)
    if args.check_only:
        print('Archive schema and source identities passed: 1,482 histories, 4 folds, 16 frozen model outputs, no scores computed.')
        return
    n=len(keys);clips=np.array([k[0] for k in keys]);rows=np.arange(n)
    truth=np.stack([compute_action_labels(h,np.ones(5),f,np.ones(30))[0] for h,f in zip(hist,future)]).astype(bool)
    check(truth.sum(0).tolist()==[49,53,214,208,975] and not (truth[:,:2]&~eligibility(hist)).any(),'Changed ground-truth event rules')
    cv_path=hist[:,-1:]+(hist[:,-1]-hist[:,-2])[:,None]*np.arange(1,31)[None,:,None]
    direct=errs(cv_path,future.astype(np.float32))
    macro=lambda v:np.mean([v[sites==s].mean(0) for s in SIZES],0)
    private=dict(clip=clips,person_id=np.array([k[1] for k in keys]),t0=np.array([k[2] for k in keys]),site=sites,truth_events=truth,cv_errors=direct)
    output={}
    gaps={}
    with ZipFile(args.bundle) as z:
        official=json.loads(z.read('report_all.json'))
        for arm in ARMS:
            old=np.empty(n,int);new=old.copy();of=np.empty((n,2));nf=of.copy();oracle=of.copy()
            flags=np.zeros((n,6,5),bool);old_pi=np.empty(n);new_pi=old_pi.copy()
            for s in SIZES:
                idx=np.flatnonzero(sites==s);ri=np.arange(len(idx))
                with np.load(io.BytesIO(z.read(f'{s}/targets.npz')),allow_pickle=False) as q:
                    np.testing.assert_array_equal(truth[idx],q['y_act'])
                    stored_y=q['fut_xy']
                with np.load(io.BytesIO(z.read(f'{s}/{arm}.npz')),allow_pickle=False) as q:
                    paths=q['trajs'];pi=q['mode_probs'];logits=q['mode_logits'].astype(float);expected=np.exp(logits-logits.max(1,keepdims=True))
                    np.testing.assert_allclose(expected/expected.sum(1,keepdims=True),pi,rtol=2e-6,atol=2e-7)
                    head=None
                    if arm!='no_head':
                        head=q['act_probs'];logit=q['act_logits'].astype(float)
                        np.testing.assert_allclose(1/(1+np.exp(-logit)),head,atol=2e-7,rtol=2e-6)
                    # All true labels and future positions are outside this call.
                    selected,score,events,allow=select(hist[idx],paths,pi,head)
                    old[idx]=pi.argmax(1);new[idx]=selected;flags[idx]=events
                    if arm=='no_head':np.testing.assert_array_equal(old[idx],new[idx])
                    check(np.isfinite(score).all() and np.isfinite(paths).all(),'Nonfinite selection score or path')
                    of[idx]=errs(paths[ri,old[idx]],stored_y);nf[idx]=errs(paths[ri,new[idx]],stored_y)
                    np.testing.assert_allclose(of[idx],q['errors'],atol=5e-6,rtol=1e-6)
                    gaps[s+'/'+arm]=float(abs(of[idx]-q['errors']).max())
                    for field,expected_metric in zip(FIELDS,of[idx].mean(0)):
                        np.testing.assert_allclose(expected_metric,official['sites'][s][arm][field],atol=5e-6,rtol=1e-6)
                        np.testing.assert_allclose(expected_metric,baseline['sites'][s][arm][field],atol=5e-6,rtol=1e-6)
                    dist=np.linalg.norm(paths.astype(float)-stored_y[:,None].astype(float),axis=-1)
                    oracle[idx]=np.stack((dist.mean(2).min(1),dist[:,:,-1].min(1)),1)
                    old_pi[idx]=pi[ri,old[idx]];new_pi[idx]=pi[ri,new[idx]]
            selected_before=flags[rows,old];selected_after=flags[rows,new];d=nf-of
            by={}
            for s in ['pooled',*SIZES]:
                m=np.ones(n,bool) if s=='pooled' else sites==s
                matrix=np.zeros((6,6),int);np.add.at(matrix,(old[m],new[m]),1)
                by[s]=dict(windows=int(m.sum()),clips=len(set(clips[m])),original=vec(of[m].mean(0)),reranked=vec(nf[m].mean(0)),cv=vec(direct[m].mean(0)),minADE_and_independent_minFDE=vec(oracle[m].mean(0)),
                    switched=int((old[m]!=new[m]).sum()),mode_changes_original_row_new_column=matrix.tolist(),event_counts=stats(m,truth,selected_before,selected_after,flags))
            rates={}
            for event in EVENTS:
                rates[event]={}
                for label in ('positive','negative'):
                    rates[event][label]={key:float(np.mean([by[s]['event_counts'][event][label][key] for s in SIZES])) for key in ('original_rate','reranked_rate')}
            output[arm]=dict(sites_and_pooled=by,four_site_m=dict(original=vec(macro(of)),reranked=vec(macro(nf)),direct_cv=vec(macro(direct)),reranked_minus_original=vec(macro(d)),
                reranked_minus_CV=vec(macro(nf-direct)),original_minus_CV=vec(macro(of-direct)),entire_unchanged_candidate_set_oracle=vec(macro(oracle))),
                four_site_switch_fraction=float(macro((old!=new).astype(float))),four_site_original_probability_before_after=[float(macro(old_pi)),float(macro(new_pi))],
                four_site_event_rates=rates)
            if arm=='detached':output[arm]['primary_paired_clip_interval_m']=interval(d,sites,clips)
            private.update({arm+'_original_index':old.astype('i1'),arm+'_selected_index':new.astype('i1'),arm+'_original_error_m':of,arm+'_selected_error_m':nf})
    args.output.mkdir(parents=True,exist_ok=True)
    arr=args.output/'paired_private.npz';np.savez_compressed(arr,**private)
    summary=dict(protocol_sha256=PLAN,source_archive_sha256=SOURCE,registered_keys_sha256=KEYS,input_histories=CENSUS if args.census else LABELS,
                 observed_windows=1482,sites=SIZES,labels=EVENTS,units='m, except dimensionless fractions/probabilities and counts',
                 primary='detached reranked minus original four-site top-path ADE; conditional paired whole-clip interval. Only one seed.',
                 selection_scope='Only history, six fixed candidates, original mode probabilities and four qualified action marginals. No new fit, network forward or probability calibration.',
                 max_rebuilt_vs_saved_top1_error_m=max(gaps.values()),paired_private_sha256=digest(arr.read_bytes()),arms=output)
    packed_json(args.output/'rerank_scores.json',summary)
    target=args.output/'action_consistency_rerank.zip'
    with ZipFile(target,'w',ZIP_DEFLATED) as z:
        for fn in ('PROTOCOL.md','selector.py','run_rerank.py','test_selector.py'):z.write(HERE/fn,fn)
        z.write(arr,arr.name);z.write(args.output/'rerank_scores.json','rerank_scores.json')
    print('Returned',target,'SHA256',digest(target.read_bytes()))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for f in ('repo','bundle','output'):p.add_argument(f,type=Path)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--census',type=Path);group.add_argument('--metadata',type=Path)
    p.add_argument('--check-only',action='store_true');main(p.parse_args())
