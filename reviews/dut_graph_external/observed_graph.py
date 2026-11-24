"""DUT single-camera history graph. The graph builder takes exactly five snapshots and five real frame IDs."""
import collections
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np

CLASSES=('Person','Car')
def sha(b):return hashlib.sha256(b).hexdigest()
def key_sha(keys):return sha(json.dumps(keys,separators=(',',':')).encode())
def array_sha(arrays):
    h=hashlib.sha256()
    for name,val in sorted(arrays.items()):
        arr=np.ascontiguousarray(val)
        h.update(name.encode()+b'\0'+arr.dtype.str.encode()+b'\0'+str(arr.shape).encode()+b'\0'+arr.tobytes())
    return h.hexdigest()

def read_scene(root,meta,manifest,retrieve):
    c=meta['clip']
    ratio_name=f'data/ratios/{c}_ratio_pixel2meter.txt'
    scale=float(retrieve(root,ratio_name,manifest[ratio_name]).read_text())
    assert scale==meta['pixels_per_metre'] and scale>0
    snaps=collections.defaultdict(dict);people=collections.defaultdict(list)
    for kind,suffix,tag,x,y in [('Person','ped','ped','x','y'),('Car','veh','veh','x_c','y_c')]:
        path=f'data/trajectories/{c}_traj_{suffix}.csv'
        with retrieve(root,path,manifest[path]).open() as f:
            for row in csv.DictReader(f):
                frame=int(row['frame']);ident=int(row['id']);key=kind,ident
                assert row['label']==tag and 1<=frame<=meta['frame_count'] and key not in snaps[frame]
                xy=(float(row[x])/scale,float(row[y])/scale);assert all(map(math.isfinite,xy))
                snaps[frame][key]=xy
                if kind=='Person':people[ident].append(frame)
    return snaps,people

def roster(people):
    # Eligibility is the pre-existing target-only continuous 35-point clock.
    # Object positions, neighbor presence and all errors are absent from this function.
    off=(np.arange(35,dtype=np.int64)*23976+4999)//10000
    for pid,fs in sorted(people.items()):
        for _,group in itertools.groupby(enumerate(sorted(fs)),lambda item:item[1]-item[0]):
            seg=[f for _,f in group]
            for f in range(seg[0],seg[-1]-int(off[-1])+1,int(off[-1])+1):
                yield pid,(f+off)

def edges(ego,history,frame_ids,reserve_car=False):
    times=np.asarray(frame_ids)
    if len(history)!=5 or times.shape!=(5,) or not np.all(np.diff(times)>0):raise ValueError('Expected ordered five observed frames')
    own=('Person',int(ego));now=history[-1];x0,y0=now[own]
    if not all(own in h for h in history):raise ValueError('Target history missing')
    def state(obj):
        present=[j for j in range(5) if obj in history[j]]
        if len(present)<2:return 0.,0.,0,0,len(present)/5
        prev=present[-2];step=int(times[-1]-times[prev]);dt=step*1000/23976
        x,y=now[obj];px,py=history[prev][obj]
        return (x-px)/dt,(y-py)/dt,4-prev,step,len(present)/5
    vx,vy,_,_,ownrho=state(own);pool=[];coincident=0
    for obj,(x,y) in now.items():
        if obj==own:continue
        dx=x-x0;dy=y-y0;d=math.hypot(dx,dy)
        if d<=1e-5:coincident+=1
        elif d<=10:pool.append((d,obj,dx,dy))
    pool.sort(key=lambda z:(z[0],z[1]))
    edge=np.zeros((12,9),np.float32);kind=np.full(12,-1,np.int8);ident=np.full(12,-1,np.int64)
    encoded=np.full(12,-1,np.int8);lag=np.zeros(12,np.int8);native_gap=np.zeros(12,np.int16)
    kept=pool[:12]
    if reserve_car and len(kept)==12 and all(r[1][0]=='Person' for r in kept):
        car=next((r for r in pool[12:] if r[1][0]=='Car'),None)
        if car is not None:kept=kept[:11]+[car]
    for i,(d,obj,dx,dy) in enumerate(kept):
        nx,ny,gap,span,rho=state(obj);good=gap!=0
        dvx,dvy=(nx-vx,ny-vy) if good else (0.,0.)
        edge[i]=[dx,dy,dvx,dvy,d,-(dx*dvx+dy*dvy)/(d+1e-6) if good else 0.,good,rho,ownrho]
        kind[i]=obj[0]=='Car';ident[i]=obj[1]
        encoded[i]=6 if kind[i] else 2 if good and math.hypot(nx,ny)<.20 else 0
        lag[i]=gap;native_gap[i]=span
    pre=[sum(r[1][0]=='Person' for r in pool),sum(r[1][0]=='Car' for r in pool),coincident]
    return edge,kind,ident,encoded,lag,native_gap,np.asarray(pre,np.int32)

def strata(sources):
    p=(sources==0).any(1);c=(sources==1).any(1)
    return {'empty':~p&~c,'person_only':p&~c,'car_only':~p&c,'mixed':p&c}

def summarize(g,ix):
    a={k:v[ix] for k,v in g.items()};src=a['source'];valid=a['lag']>0;pre=a['pre'];n=(src>=0).sum(1)
    out={'windows':len(ix),'strata':{s:int(m.sum()) for s,m in strata(src).items()},
        'precap_person_edges':int(pre[:,0].sum()),'precap_car_edges':int(pre[:,1].sum()),
        'precap_person_windows':int((pre[:,0]>0).sum()),'precap_car_windows':int((pre[:,1]>0).sum()),
        'near_coincident_excluded':int(pre[:,2].sum()),'truncated_windows':int(((pre[:,:2].sum(1))>12).sum()),
        'edges_lost_to_cap':int((pre[:,:2].sum(1)-n).sum()),'types':{str(t):int((a['encoded_type']==t).sum()) for t in (0,2,6)}}
    for j,k in enumerate(('person','car')):
        use=src==j
        out[k]={'kept_edges':int(use.sum()),'windows_with_any':int(use.any(1).sum()),'velocity_missing_edges':int((use&~valid).sum()),
                'distinct_lag_native_frames_and_edge_count':dict(sorted(collections.Counter(map(int,a['native_gap'][use&valid])).items()))}
    return out

def collect(root,coverage,retrieve):
    keys=[];frames=[];hist=[];parts=[];replacement=[]
    for m in sorted(coverage['clips'],key=lambda x:x['clip']):
        assert m['time_base']=='1/23976' and m['pts_step_ticks']==[1000]
        snaps,people=read_scene(root,m,coverage['members'],retrieve)
        for pid,f in roster(people):
            h=[snaps[int(t)] for t in f[:5]];own=('Person',pid);clip=m['clip']
            xy=np.asarray([s[own] for s in h]);hw=m['frame_size']
            assert np.all((xy*m['pixels_per_metre']>=0)&(xy*m['pixels_per_metre']<hw))
            keys.append((clip,pid,int(f[4])));frames.append(f);hist.append(xy);row=edges(pid,h,f[:5]);parts.append(row)
            needs_car=row[-1][1]>0 and not (row[1]==1).any()
            if needs_car:
                new=edges(pid,h,f[:5],True)
                assert row[1][-1]==0 and new[1][-1]==1
                for x,y in zip(row[:-1],new[:-1]):np.testing.assert_array_equal(x[:11],y[:11])
                replacement.append((len(keys)-1,new[0][-1],new[2][-1],new[4][-1],new[5][-1]))
    assert len(keys)==len(set(keys))==3066 and key_sha(keys)==coverage['key_manifest_sha256']
    g=dict(zip(('edges','source','neighbor_id','encoded_type','lag','native_gap','pre'),map(np.stack,zip(*parts))))
    names=np.array([k[0] for k in keys]);frame=np.stack(frames)
    report={'window_keys_sha256':key_sha(keys),'structural_sha256':array_sha({k:v for k,v in g.items() if k!='edges'}),
            'edge_float32_sha256':array_sha({'edges':g['edges']}),'all':summarize(g,np.arange(len(keys))),
            'by_clip':{s:summarize(g,np.flatnonzero(names==s)) for s in sorted(set(names))},
            'target_history_native_frame_gaps':dict(sorted(collections.Counter(map(int,np.diff(frame[:,:5],axis=1).ravel())).items())),
            'source_note':'Positions and typed identities from author track annotation. Original videos not redistributed. No future neighbor requirement.'}
    changed=np.zeros(len(keys),bool);redge=g['edges'][:,-1].copy();rid=g['neighbor_id'][:,-1].copy()
    for i,e,obj,_,_ in replacement:changed[i]=True;redge[i]=e;rid[i]=obj
    assert changed.sum()==810 and np.all(g['encoded_type'][changed,-1]!=6)
    with_car=(g['source']==1).any(1);assert np.sum(with_car|changed)==np.sum(g['pre'][:,1]>0)==1202
    g.update(reserve_changed=changed,reserve_last_edge=redge,reserve_last_id=rid)
    report['vehicle_slot']={'changed_windows':len(replacement),'original_car_support':int(with_car.sum()),
        'intervention_car_support':int((with_car|changed).sum()),'changed_by_clip':{s:int(np.sum((names==s)&changed)) for s in sorted(set(names))},
        'replacement_identity_sha256':array_sha({'changed':changed,'last_id':rid})}
    return keys,frame,np.stack(hist),g,report
