"""Paired, source-locked evaluation of the original and corrected causal PINNS image maps."""
import argparse
import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import platform
import tempfile
import types
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


REGISTRATION = '746793f69286e6fbfe5114158ae1451a676b767a74781f56fd7ad181fa401405'
FROZEN_PY = 'acdccb0f87c690357e54478285476f81a1913045dc0800d7a4ef5df586bcba8a'
ORIGINAL_ROSTER = '6777f1824e35f30f25e890bb35dbb98864801d59e1b28677573cf2f8cb09497f'
ORIGINAL_MAP_ARRAYS = '2b1873f056dfba34b79fd939da4cf6aa89c352d0d11f78988068072db14b5e90'
ORIGINAL_MAPS = '47453819b0542cefadb0122685d5a45876596191c21d0a855d35d596c8ee8fe7'
ORIGINAL_REFERENCES = '87983d8e23a0853ef6b7bbe68959c7d3e604b5319d486fc663d21424890408bd'
ORIGINAL_REPLAY = 'af471b7be832e47b74eee0d4232b15cba177393262468cf06166eabaed90cc7e'
METHODS = ('motion', 'availability', 'image_geometry', 'occupancy_geometry')
VERSIONS = ('legacy_raster', 'source_label')
METRICS = ('ade_m', 'fde_last_valid_m', 'fde_exact_3s_m')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def jhash(data):
    return sha(json.dumps(data, sort_keys=True, separators=(',', ':')).encode())


def frozen_math(code):
    """Use the byte-locked public implementation, including its original arithmetic and bootstrap."""
    require(sha(code) == FROZEN_PY, 'Original evaluator fingerprint')
    functions = {'geometry_features', 'array_digest', 'fit_horizons', 'forecast', 'errors', 'clip_interval'}
    constants = {'GEOM', 'ALPHA', 'REPLICATES', 'SEED'}
    module = ast.parse(code)
    nodes = [n for n in module.body if (
        isinstance(n, ast.FunctionDef) and n.name in functions
        or isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in constants)]
    require(len(nodes) == len(functions) + len(constants), 'Frozen definitions')
    env = {'np': np, 'hashlib': hashlib, 'json': json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'frozen_evaluate_pinns_causal_geometry.py', 'exec'), env)
    return types.SimpleNamespace(**{k: env[k] for k in functions | constants})


def build_features(motion, eligible, image, hull, image_mask, hull_mask):
    for a, m in ((image, image_mask), (hull, hull_mask)):
        require(np.isfinite(a).all() and np.all(a[~m] == 0) and not (m & ~eligible).any(), 'Missing geometry or source leak')
    presence = np.column_stack((eligible, image_mask, hull_mask))
    base = np.column_stack((motion, presence))
    feats = dict(motion=motion, availability=base, image_geometry=np.column_stack((base, image)),
                 occupancy_geometry=np.column_stack((base, hull)))
    return presence, feats


def paired_design(a, b, maps, manifests, funcs, replay):
    """Read only historical features, identifiers, source metadata and availability before model fitting."""
    require(manifests[a]['protocol']=='causal_same_clip_training_reference_v1', 'Legacy acquisition protocol')
    require(manifests[b]['protocol']=='causal_same_clip_training_reference_v2_source_label'
            and manifests[b]['bev_convention']==b, 'Corrected acquisition protocol')
    require(manifests[a]['sources'] == manifests[b]['sources'] and jhash(manifests[a]['sources']) == ORIGINAL_REFERENCES,
            'Any scene source, training frame, calibration, or label fingerprint changed')
    keys = list(zip(maps[a]['clip'].tolist(), maps[a]['person_id'].tolist(), maps[a]['t0'].tolist()))
    require(len(keys)==len(set(keys))==7283 and jhash(keys)==ORIGINAL_ROSTER, 'Window cohort')
    for field in ('clip', 'person_id', 't0', 'split'):
        np.testing.assert_array_equal(maps[a][field], replay[field])
    for field in ('clip','person_id','t0','split','source_eligible','p_env_hull','m_map_hull'):
        np.testing.assert_array_equal(maps[a][field], maps[b][field])
    clips = maps[a]['clip']; splits = maps[a]['split']; sites = replay['site']
    require(Counter(splits.tolist()) == {'train':5686,'val':1100,'test':497} and len(set(clips))==47 and len(set(sites))==4,
            'Original split or site roster')
    require(set(clips)==set(manifests[a]['sources']), 'Unknown clip')
    v = replay['velocity']; motion = replay['inputs_motion']
    require(v.shape==(7283,2) and motion.shape==(7283,14) and np.isfinite(motion).all(), 'Original five-frame motion')
    np.testing.assert_array_equal(v, motion[:, -2:])
    for clip, source in manifests[a]['sources'].items():
        ix=clips==clip
        require((sites[ix]==source['site']).all(), 'Source site differs')
        ref=source['reference_10hz']; raw=source['raw_image_frames']; held=source['earliest_held_trigger_10hz']
        expected=maps[a]['t0'][ix]>max(ref) if ref else np.zeros(ix.sum(), dtype=bool)
        np.testing.assert_array_equal(expected, maps[a]['source_eligible'][ix])
        if ref:
            require(len(ref)>=7 and len(set(raw))==7 and max(ref)-min(ref)<=60, 'Noncanonical reference set')
            require(held is None or max(ref)<held-19, 'Source reaches held history')
            if expected.any():
                require(max(raw)<=3*(maps[a]['t0'][ix][expected].min()-1), 'Video frame after historical cutoff')
    available={}; features={}; identities={}
    hull, hm = funcs.geometry_features(maps[a]['p_env_hull'], v)
    for name in VERSIONS:
        frames=maps[name]
        np.testing.assert_array_equal(hm, frames['m_map_hull'].astype(bool))
        image, im=funcs.geometry_features(frames['p_env_image'],v)
        np.testing.assert_array_equal(im,frames['m_map_image'].astype(bool))
        eligible=frames['source_eligible'].astype(bool)
        available[name],features[name]=build_features(motion,eligible,image,hull,im,hm)
        identities[name]=funcs.array_digest(frames)
    require(identities[a]==ORIGINAL_MAP_ARRAYS and identities[b]!=identities[a], 'Collapsed conventions or invalid original')
    np.testing.assert_array_equal(available[a],replay['availability'])
    for method in METHODS:
        np.testing.assert_allclose(features[a][method],replay['inputs_'+method],rtol=0,atol=1e-12)
    return clips,splits,sites,available,features,identities


def map_support(mask, available):
    a,b=(available[name][mask] for name in VERSIONS)
    i,j=a[:,1],b[:,1]
    return {'windows':int(mask.sum()),'training_reference':int(a[:,0].sum()),'independent_occupancy':int(a[:,2].sum()),
            'image_old_only':int((i & ~j).sum()),'image_new_only':int((~i & j).sum()),
            'image_both':int((i & j).sum()),'image_neither':int((~i & ~j).sum()),
            'without_image_geometry_old':int((~i).sum()),'without_image_geometry_new':int((~j).sum())}


def score_rows(err, mask):
    arr=err[mask]
    result={}
    for i,name in enumerate(METRICS):
        values=arr[:,i];use=np.isfinite(values);n=int(use.sum())
        result[name]={'sum_m':float(values[use].sum()),'windows':n,'mean_m':float(values[use].mean()) if n else None}
    return result


def fits(funcs, feats, cv, truth, valid, train):
    pred={'ols_five':cv}; parameters={}
    for method,x in feats.items():
        learned=funcs.fit_horizons(x,truth-cv,valid,train)
        pred[method]=funcs.forecast(x,cv,learned)
        parameters.update({method+'_'+k:v for k,v in learned.items()})
    return pred, parameters


def compare_scores(paired, funcs, sites, clips, avail, valid, splits, truth):
    metrics={name:{m:funcs.errors(z,truth,valid) for m,z in preds.items()} for name,preds in paired.items()}
    old,new=(metrics[n] for n in VERSIONS)
    delta_old=old['image_geometry']-old['availability']
    delta_new=new['image_geometry']-new['availability']
    contrasts={'corrected_image_minus_corrected_availability':delta_new,
               'legacy_image_minus_legacy_availability':delta_old,
               'difference_in_differences':delta_new-delta_old}
    for m in METHODS:
        contrasts['corrected_minus_legacy_'+m]=new[m]-old[m]
    def group(mask):
        return {'support':map_support(mask,avail),'clips':len(set(clips[mask])), 'future_point_denominator':int(valid[mask].sum()),
                'errors':{v:{m:score_rows(e,mask) for m,e in model.items()} for v,model in metrics.items()},
                'paired':{k:score_rows(x,mask) for k,x in contrasts.items()}}
    result={}
    for split in ('val','test'):
        mask=splits==split
        result[split]={'pooled':group(mask), 'by_site':{s:group(mask & (sites==s)) for s in sorted(set(sites))},
                       'by_clip':{c:group(mask & (clips==c)) for c in sorted(set(clips[mask]))}}
    test=splits=='test'
    result['registered_test_intervals']={k:funcs.clip_interval(contrasts[k][test,0],sites[test],clips[test]) for k in
        ('corrected_image_minus_corrected_availability','difference_in_differences')}
    return result


def analyze(private_zip, lock_path, original_script, protocol_path, *, score):
    lock=json.loads(lock_path.read_bytes())
    require(sha(protocol_path.read_bytes())==REGISTRATION, 'Preregistered protocol')
    require(sha(private_zip.read_bytes())==lock['private_input_sha256'], 'Private bundle identity')
    funcs=frozen_math(original_script.read_bytes())
    if score:
        require(Path('/kaggle/working').is_dir(), 'Real fitting and scoring are restricted to Kaggle')
    with ZipFile(private_zip) as pack:
        require(pack.testzip() is None and set(pack.namelist())==set(lock['member_sha256']), 'Private package roster/CRC')
        def member(name):
            blob=pack.read(name)
            require(sha(blob)==lock['member_sha256'][name], 'Package fingerprint '+name)
            return blob
        replay_bytes=member('legacy_replay.npz')
        require(sha(replay_bytes)==ORIGINAL_REPLAY, 'Original Kaggle replay changed')
        manifests={n:json.loads(member(n+'/manifest.json')) for n in VERSIONS}
        maps={}
        for name in VERSIONS:
            raw=member(name+'/maps.npz')
            require(sha(raw)==manifests[name]['npz_sha256'], 'Map payload / manifest disagreement')
            with np.load(io.BytesIO(raw),allow_pickle=False) as z: maps[name]={k:z[k] for k in z.files}
        require(manifests[VERSIONS[0]]['npz_sha256']==ORIGINAL_MAPS, 'Original map archive changed')
    with np.load(io.BytesIO(replay_bytes),allow_pickle=False) as replay:
        clips,splits,sites,available,features,identities=paired_design(*VERSIONS,maps,manifests,funcs,replay)
        del maps
        require(identities==lock['map_array_sha256'],'Public corrected map tensor identity')
        report={'registration_sha256':REGISTRATION,'private_input_sha256':lock['private_input_sha256'],'members_sha256':lock['member_sha256'],
                'script_sha256':sha(Path(__file__).read_bytes()),'original_evaluator_sha256':FROZEN_PY,'roster_sha256':ORIGINAL_ROSTER,
                'reference_sources_sha256':ORIGINAL_REFERENCES,'map_array_sha256':identities,'numpy':str(np.__version__),
                'python':platform.python_version(),'platform':platform.platform(),'scored':score,
                'design':{s:map_support(splits==s,available) for s in ('train','val','test')}}
        if not score:
            return report, {}
        truth=replay['truth_relative'];valid=replay['future_mask'];v=replay['velocity']
        require(truth.shape==(7283,30,2) and valid.shape==(7283,30) and (valid.sum(1)>=15).all() and np.isfinite(truth).all(), 'Original targets')
        cv=v[:,None]*(0.1*np.arange(1,31))[None,:,None];train=splits=='train'
        private={k:replay[k] for k in ('clip','person_id','t0','site','split','historical_mask','truth_relative','future_mask','velocity')}
        paired={}
        legacy_peak={}
        for version in VERSIONS:
            name=version+'__'
            pred,params=fits(funcs,features[version],cv,truth,valid,train)
            for method,p in pred.items():
                require(np.isfinite(p).all(), 'Nonfinite predictions')
                if version==VERSIONS[0]:
                    expected=replay['prediction_'+method]
                    np.testing.assert_allclose(p,expected,atol=2e-8,rtol=2e-10)
                    legacy_peak[method]=float(np.abs(p-expected).max())
                private[name+'prediction_'+method]=p
            private.update({name+k:x for k,x in params.items()})
            private.update({name+'inputs_'+k:x for k,x in features[version].items()})
            private[name+'map_availability']=available[version]
            paired[version]=pred
        for m in ('ols_five','motion'):
            np.testing.assert_array_equal(paired[VERSIONS[0]][m],paired[VERSIONS[1]][m])
        unchanged_controls={}
        for m in ('availability','occupancy_geometry'):
            x0,x1=(features[v][m] for v in VERSIONS)
            same_rows=(x0==x1).all(axis=1)
            if same_rows[train].all():
                for name in ('mean','scale','intercept','weights'):
                    np.testing.assert_array_equal(private[VERSIONS[0]+'__'+m+'_'+name],private[VERSIONS[1]+'__'+m+'_'+name])
                np.testing.assert_array_equal(paired[VERSIONS[0]][m][same_rows],paired[VERSIONS[1]][m][same_rows])
            unchanged_controls[m]={'training_inputs_identical':bool(same_rows[train].all()),
                                   'test_input_changed_windows':int(((~same_rows)&(splits=='test')).sum())}
        report['matched_control_checks']=unchanged_controls
        report['max_abs_legacy_predictions_vs_original_m']=legacy_peak
        report['scores']=compare_scores(paired,funcs,sites,clips,available,valid,splits,truth)
        return report,private


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('input','lock','original-evaluator','protocol','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--score',action='store_true')
    a=p.parse_args()
    report,private=analyze(a.input,a.lock,a.original_evaluator,a.protocol,score=a.score)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.output.parent) as tmp:
        root=Path(tmp)
        payload={'aggregate.json':(json.dumps(report,indent=2,allow_nan=False)+'\n').encode(),
                 'input_identities.json':a.lock.read_bytes(),'protocol.md':a.protocol.read_bytes(),
                 'runner.py':Path(__file__).read_bytes(),'original_evaluator.py':a.original_evaluator.read_bytes()}
        if a.score:
            with (root/'private_replay.npz').open('wb') as stream:
                np.savez_compressed(stream,**private)
            payload['private_replay.npz']=(root/'private_replay.npz').read_bytes()
        manifest={k:sha(v) for k,v in payload.items()}
        payload['sha256.json']=json.dumps(manifest,sort_keys=True).encode()
        with ZipFile(root/'result.zip','w',ZIP_DEFLATED,compresslevel=4) as archive:
            for n,v in payload.items(): archive.writestr(n,v)
        with ZipFile(root/'result.zip') as archive: require(archive.testzip() is None,'Output CRC')
        (root/'result.zip').replace(a.output)
    print(json.dumps(report['scores']['registered_test_intervals'] if a.score else report['design']),flush=True)


if __name__=='__main__':
    main()
