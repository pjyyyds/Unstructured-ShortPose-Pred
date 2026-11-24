"""Group previously scored five-frame forecasts by the PINNS authors' named recording sites."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from scripts.score_pinns_annotation_history import errors_from_bundle
from scripts.benchmark_heldout_sites import load_site_windows
from scripts.compare_pinns_velocity_estimators import predictions
from scripts.analyze_heldout_pairs import paired_clip_draws, REPLICATES, BOOTSTRAP_SEED
from scripts.train_pinns_heldout_sites import ROOT
from src.data.actions import compute_action_labels

CITIES={'america_crossroad':'Wyoming, USA','italy_park':'Pistoia, Italy','japan_crossroad':'Tokyo, Japan','thailand_street':'Surat Thani, Thailand'}
GROUPS={'park_and_street_two_site_equal':['italy_park','thailand_street'],
        'crossroads_two_site_equal':['america_crossroad','japan_crossroad'],'all_four_sites_equal':sorted(CITIES)}
ARMS=('no_head','auxiliary','end_to_end','detached')
FIELDS=('ade_m','fde_m')
JOINT='b84eaaadcaab4518ded15c83e9d746030e03693e89dfe9e23b15ef2f1fc8587e'


def sha(data): return hashlib.sha256(data).hexdigest()


def metric(x): return dict(zip(FIELDS,np.asarray(x).tolist()))


def run(data,reranked,joint):
    groups,counts,source=load_site_windows(data)
    assert set(groups)==set(CITIES)
    keys=[(w.clip,w.person_id,w.t0) for s in sorted(groups) for w in groups[s]]
    xyz=np.stack([w.xy for s in sorted(groups) for w in groups[s]])
    sites=np.asarray([s for s in sorted(groups) for w in groups[s]])
    clips=np.asarray([key[0] for key in keys])
    raw,info=errors_from_bundle(reranked,keys)
    estimates={'frozen_M0_five_seed_mean':raw[:,0,:,0,:].mean(0),'frozen_M1_five_seed_mean':raw[:,1,:,0,:].mean(0)}
    cv=predictions(xyz[:,:5])['last_two']; distances=np.linalg.norm(cv-xyz[:,5:],axis=-1)
    estimates['last_two_velocity']=np.column_stack([distances.mean(1),distances[:,-1]])
    np.testing.assert_allclose(raw[0,0,:,2],estimates['last_two_velocity'],atol=2e-6,rtol=2e-6)
    for a in ARMS:estimates['joint_'+a]=np.full((len(keys),2),np.nan)
    assert sha(joint.read_bytes())==JOINT
    published=json.loads((ROOT/'results/pinns_individual_joint_actions.json').read_text())
    assert published['provenance']['archive_sha256']==JOINT
    y=np.stack([compute_action_labels(x[:5],np.ones(5),x[5:],np.ones(30))[0] for x in xyz]).astype(bool)
    with zipfile.ZipFile(joint) as z:
        assert z.testzip() is None
        for s in sorted(CITIES):
            ix=np.flatnonzero(sites==s)
            with np.load(io.BytesIO(z.read(s+'/targets.npz')),allow_pickle=False) as f:
                assert list(zip(f['clip'].tolist(),f['person_id'].tolist(),f['t0'].tolist()))==[keys[i] for i in ix]
                np.testing.assert_array_equal(f['y_act'],y[ix]);np.testing.assert_array_equal(f['fut_xy'],xyz[ix,5:].astype(np.float32))
                assert f['valid_by_class'].all() and f['fut_mask'].all()
            for a in ARMS:
                with np.load(io.BytesIO(z.read(f'{s}/{a}.npz')),allow_pickle=False) as f:
                    top=f['trajs'][np.arange(len(ix)),f['mode_probs'].argmax(1)]
                    d=np.linalg.norm(top.astype(float)-xyz[ix,5:].astype(np.float32).astype(float),axis=2)
                    e=np.column_stack([d.mean(1),d[:,-1]]);np.testing.assert_allclose(e,f['errors'],atol=1e-11,rtol=0)
                estimates['joint_'+a][ix]=e
                r=published['sites'][s][a]
                np.testing.assert_allclose(e.mean(0),[r[m] for m in FIELDS],atol=1e-12,rtol=0)
    assert all(np.isfinite(x).all() for x in estimates.values())
    pairs={k+'_minus_CV':v-estimates['last_two_velocity'] for k,v in estimates.items() if k!='last_two_velocity'}
    pairs['frozen_M1_minus_M0_five_seed_mean']=estimates['frozen_M1_five_seed_mean']-estimates['frozen_M0_five_seed_mean']
    for a,b in [('auxiliary','no_head'),('end_to_end','auxiliary'),('detached','end_to_end')]:
        pairs['joint_'+a+'_minus_'+b]=estimates['joint_'+a]-estimates['joint_'+b]
    rng=np.random.default_rng(BOOTSTRAP_SEED); stats={}; draws={}; index={}
    stacked=np.column_stack(list(pairs.values()))
    for s in sorted(groups):
        ix=np.flatnonzero(sites==s);index[s]=ix
        clip_person={(keys[i][0],keys[i][1]) for i in ix}
        p=counts[s];assert len(clip_person)==p['clip_person_tracks_with_windows']
        d=paired_clip_draws(stacked[ix],clips[ix],rng).reshape(REPLICATES,len(pairs),2)
        draws[s]=d
        stats[s]={'city':CITIES[s],'windows':len(ix),'clips':len(set(clips[ix])),'within_clip_person_ids':len(clip_person),
                  'action_support':{'valid_windows':len(ix),'start_positive_windows':int(y[ix,0].sum()),'stop_positive_windows':int(y[ix,1].sum())}}
    def summarize(names):
        result={k:metric(np.mean([v[index[s]].mean(0) for s in names],0)) for k,v in estimates.items()}
        draws_averaged=np.mean([draws[s] for s in names],axis=0)
        ci=np.quantile(draws_averaged,[.025,.975],axis=0)
        diffs={key:{'difference':metric(np.mean([v[index[s]].mean(0) for s in names],0)),
                    'paired_clip_95pct':dict(zip(FIELDS,ci[:,j,:].T.tolist()))} for j,(key,v) in enumerate(pairs.items())}
        return {'sites':names,'windows':sum(stats[s]['windows'] for s in names),
                'clips':sum(stats[s]['clips'] for s in names),'within_clip_person_ids':sum(stats[s]['within_clip_person_ids'] for s in names),
                'means':result,'paired_contrasts':diffs}
    by_site={s:{**stats[s],**summarize([s])} for s in sorted(CITIES)}
    by_group={name:summarize(chosen) for name,chosen in GROUPS.items()}
    for s,row in by_site.items():
        for v in ('M0','M1'):
            expect=info['five_seed_means_by_site'][s][v]['methods']['network_top1']
            np.testing.assert_allclose(list(row['means']['frozen_'+v+'_five_seed_mean'].values()),[expect[m] for m in FIELDS],atol=2e-12,rtol=0)
    return {'scope':'Post-hoc regrouping of the same four already studied LOSO development sites. The two crossed-road locations, one park and one street do not separately identify the effect of scene type, country, geometry, camera or traffic. Park and street labels describe acquisition sites; lane marking visibility has not been coded for each window.',
            'source_paper':'https://arxiv.org/html/2605.25947v1','source_section':'Figures 7 and 8 (a–d) and surrounding discussion.',
            'units':'Fitted ground coordinates in metres; five real 10 Hz observations, complete 30-step 3s forecast. All trajectories scored by probability-selected mode; deterministic CV uses the last two ground points. People counts refer to (clip, person ID) tracks. Person identities across videos are not linked.',
            'aggregation':'Mean windows within site. Equal site weights within each specified group. For existing M0/M1 first average the five fixed training seeds per window. Joint arms use a separate seed-42 individual-only experiment, with common motion/pose backbone. The no_head joint control is not the existing motion-only M0.',
            'uncertainty':{'draws':REPLICATES,'rng_seed':BOOTSTRAP_SEED,'method':'Sample whole clips with replacement within each fixed site; identical draws across methods and metrics. Paired contrasts positive when first named method has larger error; point and percentile interval conditional on these fitted models, same fixed sites and test windows. No correction across descriptive comparisons.'},
            'provenance':{'window_ids_sha256':hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest(),
                          'labels_calibration_sha256':source,'reranked_paired_windows_sha256':info['paired_windows_sha256'],'joint_archive_sha256':JOINT,
                          'old_model_seeds':info['seeds'],'joint_model_seed':42},'by_site':by_site,'grouped':by_group}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','reranked','joint','output'):p.add_argument(k,type=Path)
    a=p.parse_args();r=run(a.data,a.reranked,a.joint);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    for k,v in r['grouped'].items():print(k,v['windows'],v['means'])
