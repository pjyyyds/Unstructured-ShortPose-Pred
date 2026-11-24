"""Select frozen modes using histories and finite map edges; only then open the sealed targets."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from collections import Counter
import numpy as np

READERS=('first','independent')
RULES=('probability','normal','full_velocity','tangent')
PROTOCOL='b435feb70bfe2f2f0876d80beeac38a85d4b6a988974b97a1e2737f57936a22c'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def read(path):return json.loads(path.read_text())
def finite_stats(x):
    v=np.asarray(x,dtype=float)
    return None if not len(v) else dict(zip(('min','median','p90','max'),np.percentile(v,[0,50,90,100]).tolist()))

def frame(point,lines,anchor_inside):
    best=None
    for no,line in enumerate(lines):
        xy=np.array(line['xy'],float);vectors=np.diff(xy,axis=0);length=np.linalg.norm(vectors,axis=1)
        if np.any(length<=0):raise ValueError('Invalid map edge')
        frac=np.clip(np.einsum('ij,ij->i',point-xy[:-1],vectors)/length**2,0,1)
        dist=np.linalg.norm(point-xy[:-1]-frac[:,None]*vectors,axis=1)
        j=int(dist.argmin())
        if best is None or dist[j]<best[0]:
            best=(float(dist[j]),no,j,float(frac[j]),xy,length,line['inside'])
    distance,no,j,f,xy,length,inside=best;arcs=np.r_[0,np.cumsum(length)]
    pos=arcs[j]+f*length[j];lo=pos-.5;hi=pos+.5
    record={'distance_m':distance,'line':no,'segment':j}
    if not anchor_inside: reason='anchor_outside_control_hull'
    elif distance>5:reason='beyond_5m'
    elif lo<0 or hi>arcs[-1]:reason='insufficient_visible_arc'
    else:
        covered=(arcs[:-1]<hi)&(arcs[1:]>lo)
        reliable=np.array(inside[:-1])&np.array(inside[1:])
        if not np.all(reliable[covered]):reason='local_curve_outside_control_hull'
        else:
            ends=np.stack([np.interp([lo,hi],arcs,x) for x in xy.T],axis=-1)
            chord=ends[1]-ends[0];magnitude=np.linalg.norm(chord)
            if magnitude>1e-6:
                return {**record,'reason':'supported'},chord/magnitude
            reason='degenerate_local_chord'
    return {**record,'reason':reason},None

def choose(history,paths,prob,lines,anchor_inside):
    # paths are model displacements from last observed position, not scoring data
    assert history.shape==(5,2) and paths.shape==(6,30,2) and prob.shape==(6,)
    velocity=(history[-1]-history[-2])/.1;speed=float(np.linalg.norm(velocity))
    main=int(np.argmax(prob));modes={k:main for k in RULES}
    record,t=frame(history[-1],lines,anchor_inside)
    record.update(modes=modes,speed_m_s=speed,angle_to_edge_deg=None,score_span_m=None)
    if t is None:return record
    if speed>=.20:
        record['angle_to_edge_deg']=float(np.degrees(np.arccos(np.clip(abs(velocity@t)/speed,0,1))))
    n=np.array([-t[1],t[0]])
    residual=paths[:,:5]-(np.arange(1,6)*.1)[:,None]*velocity
    scores={'normal':np.abs(residual@n).mean(1),'full_velocity':np.linalg.norm(residual,axis=-1).mean(1),
            'tangent':np.abs(residual@t).mean(1)}
    record['score_span_m']={k:float(np.ptp(v)) for k,v in scores.items()}
    for k,v in scores.items():modes[k]=int(np.lexsort((np.arange(6),-prob,v))[0])
    return record

def load_inputs(root):
    manifest=read(root/'inputs_manifest.json')
    assert sha(root/'PROTOCOL.md')==manifest['protocol']==PROTOCOL
    assert sha(root/'observed_and_candidates.npz')==manifest['observations_and_candidates']
    with np.load(root/'observed_and_candidates.npz',allow_pickle=False) as z: bank={k:z[k] for k in z.files}
    assert set(bank)=={'clip','person','t0','hist','paths','probs','maps','anchor_inside'}
    assert len(set(zip(bank['clip'],bank['person'],bank['t0'])))==len(bank['clip'])==59
    assert bank['hist'].shape==(59,5,2) and bank['paths'].shape==(59,6,30,2) and bank['anchor_inside'].shape==(59,)
    assert np.isfinite(bank['hist']).all() and np.isfinite(bank['paths']).all() and np.isfinite(bank['probs']).all()
    assert np.all(bank['probs']>=0) and np.allclose(bank['probs'].sum(1),1,atol=1e-6)
    return manifest,bank,json.loads(str(bank['maps']))

def collect(bank,maps):
    out=[]
    for i,c in enumerate(bank['clip']):
        rec={'key':[str(c),int(bank['person'][i]),int(bank['t0'][i])]}
        for r in READERS:rec[r]=choose(bank['hist'][i],bank['paths'][i],bank['probs'][i],maps[c][r],bank['anchor_inside'][i])
        out.append(rec)
    return out

def subgroup_selection(entries):
    a={}
    for r in READERS:
        rows=[x[r] for x in entries];support=[x for x in rows if x['reason']=='supported']
        a[r]={'windows':len(rows),'supported':len(support),'fallback':len(rows)-len(support),'reasons':dict(Counter(x['reason'] for x in rows)),
              'nearest_finite_edge_m':finite_stats([x['distance_m'] for x in rows]),'speed_m_s':finite_stats([x['speed_m_s'] for x in rows]),
              'supported_motion_angle_count':sum(x['angle_to_edge_deg'] is not None for x in support),
              'supported_unoriented_motion_angle_degrees':finite_stats([x['angle_to_edge_deg'] for x in support if x['angle_to_edge_deg'] is not None]),
              'supported_score_span_m':{k:finite_stats([x['score_span_m'][k] for x in support]) for k in RULES[1:]},
              'changed_from_probability':{k:sum(x['modes'][k]!=x['modes']['probability'] for x in support) for k in RULES[1:]}}
    both=[x for x in entries if all(x[r]['reason']=='supported' for r in READERS)]
    a['map_comparison']={'both_supported':len(both),'first_only':a['first']['supported']-len(both),'independent_only':a['independent']['supported']-len(both),
                         'neither':len(entries)-a['first']['supported']-a['independent']['supported']+len(both),
                         'same_normal_mode_if_both':sum(x['first']['modes']['normal']==x['independent']['modes']['normal'] for x in both)}
    return a

def support_report(entries):
    sites=sorted({row['key'][0] for row in entries})
    return {'scope':'Pre-outcome historical-support and frozen-mode audit. Distances are approximate ground metres; moving angle is unsigned from tangent. No future paths opened.',
            'protocol_sha256':PROTOCOL,'all':subgroup_selection(entries),'by_clip':{s:subgroup_selection([r for r in entries if r['key'][0]==s]) for s in sites}}

def transverse(path,lines,only_hull):
    def cross(u,v):return u[...,0]*v[...,1]-u[...,1]*v[...,0]
    start=path[:-1];end=path[1:];d=end-start
    for ln in lines:
        line=np.array(ln['xy']);left=line[:-1];right=line[1:]
        if only_hull:
            inside=np.array(ln['inside']);keep=inside[:-1]&inside[1:];left=left[keep];right=right[keep]
        if not len(left):continue
        v=right-left
        a=cross(d[:,None],left-start[:,None]);b=cross(d[:,None],right-start[:,None])
        c=cross(v,start[:,None]-left);e=cross(v,end[:,None]-left)
        if np.any((a*b<0)&(c*e<0)):return True
    return False

def metrics(errors,index):
    return None if not len(index) else dict(zip(('ade_m','fde_m'),errors[index].mean(0).tolist()))

def evaluate(bank,maps,entries,future):
    assert future.shape==(59,30,2) and np.isfinite(future).all()
    sites=sorted(maps);clips=bank['clip'];abs_paths=bank['paths'].astype(np.float64)+bank['hist'][:,-1:,None,:]
    d=np.linalg.norm(abs_paths-future[:,None],axis=-1)
    candidate_errors=np.stack([d.mean(2),d[:,:,-1]],axis=2)
    both=np.array([all(x[r]['reason']=='supported' for r in READERS) for x in entries])
    report={}
    for r in READERS:
        support=np.array([e[r]['reason']=='supported' for e in entries])
        all_i=np.arange(len(clips));modes={k:np.array([e[r]['modes'][k] for e in entries]) for k in RULES}
        errs={k:candidate_errors[all_i,m] for k,m in modes.items()};pair={k:errs['normal']-errs[k] for k in ('probability','full_velocity','tangent')}
        bins={}
        for gn,mask in [('all_including_fallback',np.ones(len(clips),bool)),('supported',support),('unsupported',~support),('both_maps_supported',both)]:
            measured={};mean={};delta={}
            for s in sites:
                idx=np.flatnonzero((clips==s)&mask)
                measured[s]={'windows':len(idx),'tracks':len(set(bank['person'][idx].tolist())),
                             'scores':{k:metrics(v,idx) for k,v in errs.items()},'normal_minus':{k:metrics(v,idx) for k,v in pair.items()}}
            valid=[x for x in measured.values() if x['windows']]
            def equal(field,arm):
                return {f:float(np.mean([v[field][arm][f] for v in valid])) for f in ('ade_m','fde_m')} if valid else None
            bins[gn]={'windows':int(mask.sum()),'contributing_clips':len(valid),'by_clip':measured,
                      'contributing_clip_equal':{k:equal('scores',k) for k in errs},
                      'normal_minus_contributing_clip_equal':{k:equal('normal_minus',k) for k in pair},
                      'pooled_normal_minus_by_window':{k:metrics(v,np.flatnonzero(mask)) for k,v in pair.items()}}
        crossings={}
        for restricted in (False,True):
            cases=[]
            for i,s in enumerate(clips):
                here=bank['hist'][i,-1:]
                truth=transverse(np.vstack((here,future[i])),maps[s][r],restricted)
                pred=np.array([transverse(np.vstack((here,p)),maps[s][r],restricted) for p in abs_paths[i]])
                cases.append({'truth':truth,'mixed_candidates':bool(pred.any() and not pred.all()),
                              **{k:bool(pred[m[i]]) for k,m in modes.items()}})
            def count(idx):
                return {'windows':len(idx),**{k:sum(int(cases[i][k]) for i in idx) for k in cases[0]}}
            key='segments_inside_calibration_hull' if restricted else 'all_drawn_segments'
            crossings[key]={'all':count(all_i),'supported':count(np.flatnonzero(support)),
                            'by_clip':{s:count(np.flatnonzero(clips==s)) for s in sites}}
        report[r]={'scores':bins,'strict_intersections':crossings}
    return report

def main(root,out,score):
    out.mkdir(exist_ok=True,parents=True);manifest,bank,maps=load_inputs(root)
    entries=collect(bank,maps);select=out/'private_selected_modes.json';save(select,entries);locked=sha(select)
    support=support_report(entries);save(out/'support.json',support)
    if not score:return
    if not Path('/kaggle/working').is_dir():raise ValueError('Full scores run on Kaggle only')
    assert sha(root/'sealed_targets.npz')==manifest['sealed_targets']
    with np.load(root/'sealed_targets.npz',allow_pickle=False) as z:
        assert json.loads(str(z['keys']))==[e['key'] for e in entries];future=z['future']
    assert sha(select)==locked
    summary={'protocol_sha256':PROTOCOL,'runner_sha256':sha(Path(__file__)),'input_manifest':manifest,'locked_choices_sha256':locked,'scope':'Four previously studied clips, one per site; no fitting, no sampling intervals. Unavailable geometry retains the original top choice. Strict geometric crossing is permitted and only descriptive.','historical_support':support,'readers':evaluate(bank,maps,read(select),future)}
    save(out/'visible_boundary_forecasts.json',summary)
    outputs=[out/'visible_boundary_forecasts.json',select,out/'support.json',root/'PROTOCOL.md',root/'inputs_manifest.json',Path(__file__)]
    with zipfile.ZipFile(out/'visible_boundary_forecasts.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in outputs:z.write(p,p.name)
        z.writestr('sha256.json',json.dumps({p.name:sha(p) for p in outputs},sort_keys=True))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('inputs',type=Path);p.add_argument('output',type=Path);p.add_argument('--score',action='store_true')
    args=p.parse_args();main(args.inputs,args.output,args.score)
