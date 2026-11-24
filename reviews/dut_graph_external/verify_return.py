"""Check a returned ZIP locally; optional independent replay of already-stored predictions on Kaggle CPU."""
import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

D=Path(__file__).resolve().parent
ANGLES=('0','90','180','270','mirror_y')
SITES=('america_crossroad','italy_park','japan_crossroad','thailand_street')
ARMS=('M1','M2b_real','M2b_empty','real_edges_off','real_drop_car','real_drop_people')
EXTRA='real_reserve_one_car'
PAIRS=[('M2b_real','M1'),('M2b_real','M2b_empty'),('M2b_empty','M1')]+[(k,'M2b_real') for k in ARMS[3:]]
def sha(b):return hashlib.sha256(b).hexdigest()
def compare(a,b,tol=1e-10):np.testing.assert_allclose(a,b,rtol=0,atol=tol)
def values(d):return np.array([d['ade_m'],d['fde_m']])

def paired_check(e,select,c,expected):
    strata=sorted(set(c[select]));assert int(select.sum())==expected['windows'] and len(strata)==expected['clips']
    if not strata:return
    means=np.stack([e[select&(c==s)].mean(0) for s in strata])
    compare(means.mean(0),values(expected['clip_equal']))
    r=np.random.default_rng(2026).integers(0,len(strata),size=(20000,len(strata)))
    weights=np.stack([(r==i).sum(1) for i in range(len(strata))],axis=1)
    ci=np.percentile(weights@means/len(strata),[2.5,97.5],axis=0)
    compare(ci.T,values(expected['paired_clip_95']))
    for i,name in enumerate(strata):compare(means[i],values(expected['by_clip'][name]['difference']))

def run(bundle,cache,weights,replay):
    result={'bundle_sha256':sha(bundle.read_bytes())}
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        manifest=json.loads(z.read('sha256.json'));assert set(manifest)==set(z.namelist())-{'sha256.json'}
        for name,h in manifest.items():assert sha(z.read(name))==h
        a=json.loads(z.read('aggregate.json'))
        assert a['pose_cache_sha256']==sha(cache.read_bytes())
        assert a['source_training_archive_sha256']==sha(weights.read_bytes())=='0b3e590b8d7cb2bf99670fd27d3bdbf2a40fa0e77bf29f7ac0423b73535589f8'
        for name,h in a['runner_and_protocol'].items():assert sha((D/name).read_bytes())==h==manifest['sources/'+name]
        assert a['conditions']==list(ANGLES) and a['arms']==list(ARMS) and a['history']==json.loads((D/'historical_denominators.json').read_bytes())
        with zipfile.ZipFile(weights) as w:
            for site in SITES:
                for arm in ARMS[:3]: assert sha(w.read(f'{site}_{arm}.pt'))==a['source_weights'][site][arm]
        with np.load(io.BytesIO(z.read('private_graph.npz')),allow_pickle=False) as g:
            keys=json.loads(g['keys_json'].item());clips=np.asarray([s for s,p,t in keys]);sub=g['reserve_changed'];car=g['pre'][:,1]>0
            assert sha(json.dumps(keys,separators=(',',':')).encode())=='02c44cf55e5ca8690dc1fe9ac9c076af44dc89f11629b8ccc9e725e49e4580ec'
            assert len(keys)==3066 and sub.sum()==810 and car.sum()==1202
            assert np.array_equal(sub,car&~(g['source']==1).any(1))
            with zipfile.ZipFile(cache) as src, np.load(io.BytesIO(src.read('private.npz')),allow_pickle=False) as pose:
                assert json.loads(pose['keys_json'].item())==keys
                np.testing.assert_array_equal(g['frames'],pose['source_frame_1based'])
                np.testing.assert_array_equal(g['pose_mask'],pose['matched'])
        members=[f'paths_{a}_{s}.npz' for a in ANGLES for s in SITES]
        assert set(members)=={k for k in z.namelist() if k.startswith('paths_')}
        result['checked_prediction_files']=len(members);result['real_scores_recomputed']=False
        if not replay:return result
        if not Path('/kaggle/working').is_dir():raise RuntimeError('Score replay requires Kaggle')
        with np.load(io.BytesIO(z.read('errors_and_targets.npz')),allow_pickle=False) as saved:
            y=saved['truth'];rows=np.arange(len(y));every=np.ones(len(y),bool)
            for angle in ANGLES:
                arms=ARMS+(EXTRA,) if angle=='0' else ARMS;ee={arm:[] for arm in arms}
                for si,site in enumerate(SITES):
                    with np.load(io.BytesIO(z.read(f'paths_{angle}_{site}.npz')),allow_pickle=False) as preds:
                        for arm in arms:
                            p,q,k=[preds[arm+'_'+n] for n in ('p','q','top')]
                            assert k.shape==(len(y),) and p.shape==(len(y),6,30,2) and np.isfinite(p).all()
                            np.testing.assert_array_equal(q.argmax(1),k);compare(q.sum(1),1,2e-6)
                            d=p[rows,k].astype(float)-y
                            dist=np.hypot(d[:,:,0],d[:,:,1]);err=np.column_stack((dist.mean(1),dist[:,-1]))
                            compare(err,saved[angle+'_'+arm][si])
                            ee[arm].append(err)
                        if angle=='0':
                            for n in ('p','q','top'):np.testing.assert_equal(preds[EXTRA+'_'+n][~sub],preds['M2b_real_'+n][~sub])
                means={arm:np.mean(ee[arm],0) for arm in arms}
                for arm,v in means.items():
                    byclip=np.stack([v[clips==s].mean(0) for s in sorted(set(clips))])
                    compare(byclip.mean(0),values(a['all_axes'][angle]['subgroups']['all'][arm]['clip_equal']))
                if angle=='0':
                    for left,right in PAIRS:paired_check(means[left]-means[right],every,clips,a['native_primary_and_secondary'][left+'_minus_'+right])
                    for name,mask in zip(('all','affected','precap_any_car'),(every,sub,car)):paired_check(means[EXTRA]-means['M2b_real'],mask,clips,a['native_vehicle_slot'][name])
            result['real_scores_recomputed']=True
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('pose_cache',type=Path);p.add_argument('source_weights',type=Path)
    p.add_argument('--replay-on-kaggle',action='store_true');x=p.parse_args();print(json.dumps(run(x.bundle,x.pose_cache,x.source_weights,x.replay_on_kaggle),indent=2))
