"""Verify an already-returned private score archive, without importing the scoring program."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import numpy as np

F=('ade_m','fde_m')
R=('first','independent')
RULES=('probability','normal','full_velocity','tangent')
PLAN='b435feb70bfe2f2f0876d80beeac38a85d4b6a988974b97a1e2737f57936a22c'
LOCK='863c7d956ffcca22e5553bd4415632bf27d4bd3808e2eeda6d57eb5119a6fdc3'
def digest(x):return hashlib.sha256(x).hexdigest()
def near(x,y):
    np.testing.assert_allclose(x,y,atol=1e-10,rtol=1e-10)

def crosses(path,lines,hull):
    # Solve the two-segment parametric intersection; open intervals reject vertex/tangent touches.
    for line in lines:
        xy=np.asarray(line['xy']);keep=np.asarray(line['inside'],bool);seg=np.diff(xy,axis=0)
        for i,v in enumerate(seg):
            if hull and not (keep[i] and keep[i+1]):continue
            p,q=path[:-1],path[1:];d=q-p;off=xy[i]-p
            det=d[:,0]*v[1]-d[:,1]*v[0];nz=det!=0
            if not np.any(nz):continue
            time=(off[nz,0]*v[1]-off[nz,1]*v[0])/det[nz]
            arc=(off[nz,0]*d[nz,1]-off[nz,1]*d[nz,0])/det[nz]
            if np.any((time>0)&(time<1)&(arc>0)&(arc<1)):return True
    return False

def verify(root,bundle):
    with zipfile.ZipFile(bundle) as z:
        expected={'sha256.json','PROTOCOL.md','inputs_manifest.json','support.json','private_selected_modes.json','visible_boundary_forecasts.json','evaluate_visible_boundaries.py'}
        assert z.testzip() is None and set(z.namelist())==expected and len(z.namelist())==len(expected)
        manifest=json.loads(z.read('sha256.json'));assert set(manifest)==expected-{'sha256.json'}
        assert all(digest(z.read(k))==v for k,v in manifest.items())
        assert digest(z.read('PROTOCOL.md'))==PLAN and z.read('evaluate_visible_boundaries.py')==(root/'evaluate_visible_boundaries.py').read_bytes()
        inp=json.loads(z.read('inputs_manifest.json'))
        assert inp==json.loads((root/'inputs_manifest.json').read_text())
        assert all(inp[f]==digest((root/p).read_bytes()) for f,p in [('observations_and_candidates','observed_and_candidates.npz'),('sealed_targets','sealed_targets.npz')])
        locked=json.loads(z.read('private_selected_modes.json'));r=json.loads(z.read('visible_boundary_forecasts.json'))
        assert r['locked_choices_sha256']==digest(z.read('private_selected_modes.json'))
    core=[[e['key'],*[{'reason':e[r]['reason'],'modes':e[r]['modes']} for r in R]] for e in locked]
    assert digest(json.dumps(core,sort_keys=True,separators=(',',':')).encode())==LOCK
    with np.load(root/'observed_and_candidates.npz',allow_pickle=False) as z:
        clips=z['clip'];people=z['person'];anchor=z['hist'][:,-1];paths=z['paths'].astype(float)+anchor[:,None,None,:];maps=json.loads(str(z['maps']))
        keys=[list(k) for k in zip(clips.tolist(),people.tolist(),z['t0'].tolist())]
    with np.load(root/'sealed_targets.npz',allow_pickle=False) as z:
        assert json.loads(str(z['keys']))==keys==[e['key'] for e in locked];future=z['future']
    points=len(clips);score_checks=0;cross_checks=0;stats={}
    for reader in R:
        modes={k:[e[reader]['modes'][k] for e in locked] for k in RULES}
        errors={}
        for k,choice in modes.items():
            ds=np.hypot(*(paths[np.arange(points),choice]-future).transpose(2,0,1))
            errors[k]=np.stack([ds.mean(1),ds[:,-1]],axis=1)
        eligible=np.array([e[reader]['reason']=='supported' for e in locked])
        both=np.array([all(e[rr]['reason']=='supported' for rr in R) for e in locked])
        for name,mask in [('all_including_fallback',np.ones(points,bool)),('supported',eligible),('unsupported',~eligible),('both_maps_supported',both)]:
            block=r['readers'][reader]['scores'][name];indices=np.flatnonzero(mask);assert len(indices)==block['windows']
            for k,e in errors.items():
                sb=[]
                for s,v in block['by_clip'].items():
                    idx=np.flatnonzero(mask&(clips==s));assert len(idx)==v['windows'] and len(set(people[idx]))==v['tracks']
                    if not len(idx):assert v['scores'][k] is None;continue
                    pair=e[idx].mean(0);near(pair,[v['scores'][k][f] for f in F]);sb.append(pair);score_checks+=2
                    if k!='normal':
                        delta=(errors['normal'][idx]-e[idx]).mean(0)
                        near(delta,[v['normal_minus'][k][f] for f in F]);score_checks+=2
                if not sb:assert block['contributing_clip_equal'][k] is None;continue
                assert len(sb)==block['contributing_clips']
                near(np.mean(sb,0),[block['contributing_clip_equal'][k][f] for f in F]);score_checks+=2
                if k!='normal':
                    vals=block['normal_minus_contributing_clip_equal'][k]
                    local=block['contributing_clip_equal']
                    near([vals[f] for f in F],[local['normal'][f]-local[k][f] for f in F])
                    near((errors['normal'][indices]-e[indices]).mean(0),[block['pooled_normal_minus_by_window'][k][f] for f in F]);score_checks+=4
        for h,nm in [(False,'all_drawn_segments'),(True,'segments_inside_calibration_hull')]:
            events=[]
            for i,s in enumerate(clips):
                conditions=maps[s][reader];origin=anchor[i:i+1]
                y=crosses(np.vstack([origin,future[i]]),conditions,h)
                p=[crosses(np.vstack([origin,x]),conditions,h) for x in paths[i]]
                events.append({'truth':y,'mixed_candidates':any(p) and not all(p),**{k:p[v[i]] for k,v in modes.items()}})
            def compare(idx,ref):
                assert ref['windows']==len(idx) and all(ref[k]==sum(int(events[i][k]) for i in idx) for k in events[0])
                return len(events[0])
            result=r['readers'][reader]['strict_intersections'][nm]
            cross_checks+=compare(np.arange(points),result['all'])+compare(np.flatnonzero(eligible),result['supported'])
            for s,sr in result['by_clip'].items():cross_checks+=compare(np.flatnonzero(clips==s),sr)
        stats[reader]=r['readers'][reader]['scores']['all_including_fallback']['normal_minus_contributing_clip_equal']['probability']
    return {'archive_sha256':digest(bundle.read_bytes()),'protocol_sha256':PLAN,'locked_history_only_decisions_sha256':LOCK,'windows':points,
            'verified_error_summary_scalars':score_checks,'verified_intersection_summary_counts':cross_checks,'primary_or_sensitivity_four_clip_normal_minus_probability_m':stats}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input_root',type=Path);p.add_argument('returned',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();out=verify(args.input_root,args.returned);args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
