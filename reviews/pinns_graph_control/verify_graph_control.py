"""Independent source/weight replay. Never imports the training driver or fits a predictor."""
from collections import Counter,defaultdict
import argparse, hashlib, io, itertools, json
from pathlib import Path
import numpy as np
import torch

from src.models.predictor import UnstructuredShortPosePredictor

TRAIN_SHA='d63162e1ee8d1824436d0cc1a60e573b9108beb6982ea3998cfbb27543a6a9bd'
RAW='0b276b34ab1b01dae91f67c56bfaea5cc931511b05b4fc1c92686a891e6f6801'
POSE='b493a47536664ddb4fd66603c1bbf6b2c2e33f553520382027b3ae96b53447d0'
IDS='f5ee511a54ad818499d3dde6213f7f3b49332943ff2fc93d93aed87d316dbb91'
TREE='e61a2310793e900c6188588c750e47c44e23fbf41f429fb94ccf8b2ea88fd094'
CENSUS='8f3d6c78e5e4bc40091db485729774e9b53e5402e32f65a3091084c0226506b3'
LOCK='51c70ae3866ddd14b327d00bfa91aed665e0e6c703da8eba1a96d9e7dbd4f057'
MODELS=('M1','M2b_real','M2b_empty')
INTERVENTIONS=('real_edges_off','real_drop_car','real_drop_people')
METHODS=MODELS+INTERVENTIONS
SITES=('america_crossroad','italy_park','japan_crossroad','thailand_street')
N_SITE=(816,84,432,150)
TARGETS=('ade_m','fde_m')
CONTRASTS=(('M2b_real','M1'),('M2b_real','M2b_empty'),('M2b_empty','M1'))+tuple((a,'M2b_real') for a in INTERVENTIONS)

def h(b):return hashlib.sha256(b).hexdigest()
def jh(x):return h(json.dumps(x,separators=(',',':')).encode())
def ah(x):
    d=hashlib.sha256()
    for k in sorted(x):
        v=np.asarray(x[k]);head=[k.encode(),v.dtype.str.encode(),str(v.shape).encode()]
        d.update(b'\0'.join(head)+b'\0'+v.tobytes(order='C'))
    return d.hexdigest()
def near(a,b,tol=0,rtol=0):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_allclose(a,b,rtol=rtol,atol=tol)
    return float(np.max(np.abs(a.astype(float)-b.astype(float)),initial=0))

def flip_y(c):
    image=np.asarray(c['pts_2d'],dtype=float);ground=np.asarray(c['pts_3d'],dtype=float)
    A=[];rhs=[]
    for (u,v),(x,y) in zip(image,ground):
        A.extend([[u,v,1,0,0,0,-u*x,-v*x],[0,0,0,u,v,1,-u*y,-v*y]]);rhs.extend([x,y])
    H=np.r_[np.linalg.solve(A,rhs),1.].reshape(3,3)
    def transform(pt):
        p=H@np.r_[pt,1.];return p[:2]/p[2]
    uv=image.mean(0); origin=transform(uv)
    return np.linalg.det(np.stack([transform(uv+[2,0])-origin,transform(uv+[0,2])-origin]))>0

def raw_labels(root):
    census={};by_site=defaultdict(list);sha=hashlib.sha256();files=set()
    for path in sorted((root/'labels').glob('*.txt')):
        condition=path.stem.rsplit('_',1)[0];cal=root/'videos'/condition/'calib.json'
        c=json.loads(cal.read_text());site=c['location']+'_'+c['scene'];assert c['fps']==30.
        files.update((path,cal));flip=flip_y(c)
        objs=defaultdict(dict);people=defaultdict(list);kind_counts=Counter()
        for line in path.read_text().splitlines():
            cols=line.split()
            if not cols:continue
            assert len(cols)==12
            raw,kind,ident=int(cols[0]),cols[1],int(cols[2]);kind_counts[kind]+=1
            if (raw-1)%3 or raw<1:continue
            t=(raw-1)//3;key=kind,ident
            assert kind in ('Person','Car') and key not in objs[t]
            x,y=float(cols[8]),float(cols[9])
            pos=np.array([x,c['y_max']-(y-c['y_min']) if flip else y])
            assert np.isfinite(pos).all()
            objs[t][key]=(pos,int(float(cols[10])))
            if kind=='Person':people[ident].append(t)
        census[path.stem]=dict(kind_counts)
        for ident,times in people.items():
            for _,run in itertools.groupby(enumerate(sorted(times)),lambda z:z[1]-z[0]):
                run=[f for _,f in run]
                for j in range(len(run)//35):
                    ticks=run[j*35:(j+1)*35];t=ticks[4];key=('Person',ident)
                    xy=np.stack([objs[i][key][0] for i in ticks]);target=xy-xy[4]
                    by_site[site].append(((path.stem,ident,t),target,objs[t][key][1],edge_row(objs,ident,t)))
    for f in sorted(files):
        sha.update(f.relative_to(root).as_posix().encode()+b'\0'+hashlib.sha256(f.read_bytes()).digest())
    assert len(census)==47 and len(files)==69 and sha.hexdigest()==RAW
    assert Counter({k:sum(d.get(k,0) for d in census.values()) for k in set().union(*(v.keys() for v in census.values()))})=={'Person':180344,'Car':47555}
    assert tuple(sorted(by_site))==SITES and tuple(len(by_site[s]) for s in SITES)==N_SITE
    rows=[r for s in SITES for r in by_site[s]];keys,xy,fixed,graph=zip(*rows)
    assert len(set(keys))==len(keys)==1482 and jh(list(keys))==IDS
    gs={k:np.stack([e[k] for e in graph]) for k in graph[0]}
    distinct={n:gs[n] for n in ('neighbor_id','source_class','velocity_gap_ticks','candidate_counts','encoded_type')}
    assert ah(distinct)==CENSUS
    return list(keys),np.stack(xy),np.array(fixed),gs,census

def edge_row(frames,pid,t):
    # Assemble candidates through typed identities inside five time instants; future frames are never read.
    times=range(t-4,t+1); hist=defaultdict(dict)
    for tick in times:
        for k,(pos,_) in frames[tick].items():hist[k][tick]=pos
    ego=('Person',pid);origin=hist[ego][t];ev=(origin-hist[ego][t-1])/.1
    matched=[]
    for k,bytime in hist.items():
        if k==ego or t not in bytime:continue
        delta=bytime[t]-origin;d=float(np.sqrt(np.sum(delta**2)))
        matched.append((d,k,delta,bytime))
    nearby=sorted([x for x in matched if 1e-5<x[0]<=10],key=lambda z:(z[0],z[1][0],z[1][1]))
    edges=np.zeros((12,9),np.float32);onehot=np.zeros((12,8),np.float32)
    ids=np.full(12,-1,np.int64);src=np.full(12,-1,np.int8);gaps=np.full(12,-1,np.int8);ty=np.full(12,-1,np.int8)
    for i,(distance,k,dp,track) in enumerate(nearby[:12]):
        elapsed=t-max([u for u in track if u<t],default=t)
        has=int(elapsed>0)
        v=(track[t]-track[t-elapsed])/(elapsed*.1) if has else np.zeros(2)
        dv=(v-ev)*has
        source=int(k[0]=='Car'); category=6 if source else (2 if has and sum(v*v)<.2**2 else 0)
        ids[i]=k[1];src[i]=source;gaps[i]=elapsed;ty[i]=category;onehot[i,category]=1
        edges[i]=[*dp,*dv,distance,-sum(dp*dv)/(distance+1e-6),has,len(track)/5,1.]
    n=len(nearby)
    return {'nbr_edges':edges,'nbr_type_oh':onehot,'nbr_mask':(ids>=0).astype(np.float32),
            'neighbor_id':ids,'source_class':src,'velocity_gap_ticks':gaps,'candidate_counts':np.int64(n),'encoded_type':ty,
            'eligible_person_car_and_excluded_coincident_counts':np.array([sum(v[1][0]==c for v in nearby) for c in ('Person','Car')]+[sum(d<=1e-5 for d,*_ in matched)],np.int64)}

def model_inputs(path,keys,xy,fixed):
    assert h(path.read_bytes())==POSE
    with np.load(path,allow_pickle=False) as z:
        cache_keys=list(zip(z['clip'].tolist(),z['person_id'].tolist(),z['t0'].tolist()))
        order={k:i for i,k in enumerate(cache_keys)};assert set(keys)==set(order) and len(order)==len(cache_keys)
        ids=[order[k] for k in keys];kpt=z['hist_kpts'][ids];bb=z['hist_bbox'][ids];height=z['img_h'][ids]
        near(z['hist_pos_relative'][ids],xy[:,:5],1e-4);assert np.all(z['hist_mask'][ids]==1)
        cached_type=z['ego_type'][ids]
    n=len(keys);motion=np.zeros((n,5,9),np.float32)
    motion[:,:,:2]=xy[:,:5];v=np.diff(xy[:,:5],axis=1)/.1
    motion[:,1:,2:4]=v;motion[:,1:,4]=np.sqrt((v*v).sum(axis=-1));motion[:,:,5]=1;motion[:,1:,6]=1;motion[:,1:,7]=.1;motion[:,:,8]=1
    typ=np.where((fixed==1)|(motion[:,-1,4]<.20),2,0);np.testing.assert_array_equal(typ,cached_type)
    conf=np.clip(kpt[:,:,:,2],0,1);valid_box=bb[:,:,3]>1;valid=(conf>=.2)&valid_box[:,:,None]
    both=valid[:,:,11]&valid[:,:,12]
    root=np.where(both[:,:,None],(kpt[:,:,11,:2]+kpt[:,:,12,:2])*.5,bb[:,:,:2])
    den=np.where(valid_box,bb[:,:,3],1.)
    xp=np.zeros((n,5,17,6),np.float32)
    xp[:,:,:,:2]=np.clip((kpt[:,:,:,:2]-root[:,:,None,:])/den[:,:,None,None],-3.,3.)*valid[:,:,:,None]
    xp[:,:,:,2]=conf*valid;xp[:,:,:,3]=valid;xp[:,:,:,4]=both[:,:,None]
    box_h=bb[:,:,3].astype(float)/height[:,None]
    xp[:,:,:,5]=np.where(valid_box,np.log(np.maximum(box_h,.0001)),0.)[:,:,None]
    available=valid.sum(axis=(1,2));xp[available==0]=0.
    q=np.column_stack([available/85,(valid*conf).sum(axis=(1,2))/(available+1e-6),
       (np.minimum(box_h,1.)*valid_box).sum(1)/np.maximum(valid_box.sum(1),1),np.ones(n)]).astype(np.float32)
    return {k:torch.from_numpy(v) for k,v in {'x_m':motion,'x_p':xp,'q_vec':q,'ego_type_oh':np.eye(8,dtype=np.float32)[typ]}.items()}

def layer_eval(layer,current,neighbor,emb,present):
    batch,slots,D=neighbor.shape;heads=4;unit=D//heads
    sh=(batch,slots,heads,unit)
    add=layer.w_q(current).reshape(batch,1,heads,unit)+layer.w_k(neighbor).reshape(sh)+layer.w_e(emb).reshape(sh)
    score=(layer.leaky(add)*layer.attn_vec).sum(-1)
    logits=score.masked_fill(~present[:,:,None],-1e9)
    mass=torch.softmax(torch.cat((logits,layer.sink_logit.expand(batch,1,heads)),dim=1),dim=1)[:,:-1]
    values=layer.w_v(torch.cat((neighbor,emb),-1)).reshape(sh)
    pooled=(values*(mass*present[:,:,None])[:,:,:,None]).sum(1).reshape(batch,D)
    updated=layer.norm(current+pooled*present.any(1,keepdim=True))
    neigh=(neighbor+layer.nbr_ffn(neighbor))*present[:,:,None] if layer.update_nbr else neighbor
    return updated,neigh

@torch.inference_mode()
def forward(model,method,base,graph,inds):
    paths=[];probs=[]
    for i in range(0,len(inds),23):
        take=inds[i:i+23];x=base['x_m'][take];q=base['q_vec'][take]
        m=model.motion_enc(x);p,_=model.pose_enc(base['x_p'][take])
        gate=model.quality_gate
        available=q[:,0:1]>0;p=p*available
        learned=gate.gate_out(torch.cat((gate.mp_proj(torch.cat((m,p),1)),gate.q_mlp(q)),1))
        joint_gate=available*torch.sqrt((q[:,0:1]*q[:,1:2]).clamp(min=0))*(.4+.6*q[:,3:4])*torch.sigmoid(learned)
        fused=gate.out_norm(m+joint_gate*p);net=model.interaction
        vec=net.base_norm(fused+net.type_proj(base['ego_type_oh'][take]))
        if method!='M1':
            tags=graph['source_class'][take]
            keep=tags>=0
            if method in ('M2b_empty','real_edges_off'):keep[:]=False
            if method=='real_drop_car':keep&=(tags==0)
            if method=='real_drop_people':keep&=(tags==1)
            valid=torch.from_numpy(keep);mask=valid[:,:,None]
            edge=torch.from_numpy(graph['nbr_edges'][take])*mask
            types=torch.from_numpy(graph['nbr_type_oh'][take])*mask
            embedded=net.edge_mlp(edge)*mask
            h_neighbor=net.nbr_init_proj(torch.cat((edge,types),-1))*mask
            v1,h_neighbor=layer_eval(net.layer1,vec,h_neighbor,embedded,valid)
            v2,_=layer_eval(net.layer2,v1,h_neighbor,embedded,valid)
            change=v2-vec
            g=valid.any(1,keepdim=True)*torch.sigmoid(net.nbr_gate(torch.cat((vec,change),1)))
            assert torch.all(g[~valid.any(1)]==0)
            vec=net.out_norm(vec+g*change)
        y=model.decoder(vec,vec.new_zeros((len(take),64)),x[:,-1,2:4]*x[:,-1,6:7])
        paths.append(y['trajs'].numpy());probs.append(y['mode_probs'].numpy())
    return np.concatenate(paths),np.concatenate(probs)

def scores(paths,prob,target):
    near(prob.sum(1),np.ones(len(prob)),2e-6)
    assert np.all(prob>=0)
    winner=prob.argmax(1)
    step=np.linalg.norm(paths[np.arange(len(paths)),winner].astype(float)-target.astype(float),axis=-1)
    return winner,np.stack((step.mean(1),step[:,-1]),axis=1)

def summary(keys,groups,src,err,agg):
    def read(row):return np.array([row[c] for c in TARGETS])
    by={s:{m:err[m][g].mean(0) for m in METHODS} for s,g in groups.items()}
    for s,g in groups.items():
        for m in METHODS:near(by[s][m],read(agg['site_m'][s][m]),1e-12)
        a,b=(src[g]==0).any(1),(src[g]==1).any(1)
        for tag,mask in {'empty':~a&~b,'person_only':a&~b,'car_only':~a&b,'mixed':a&b,'with_any_person':a,'with_any_car':b}.items():
            cell=agg['strata_by_site'][s][tag];assert cell['windows']==mask.sum()
            if mask.any():near((err['M2b_real']-err['M1'])[g[mask]].mean(0),read(cell['paired_real_minus_m1_m']),1e-12)
            else:assert cell['paired_real_minus_m1_m'] is None
    for m in METHODS:near(np.mean([x[m] for x in by.values()],axis=0),read(agg['four_site_m'][m]),1e-12)
    cl=np.array([k[0] for k in keys]);rng=np.random.default_rng(2026);sim={}
    for s,g in groups.items():
        names=np.unique(cl[g]);G=len(names)
        ind=np.array([cl[g]==c for c in names]).astype(float)
        count=rng.multinomial(G,np.ones(G)/G,size=20000)
        denom=count@ind.sum(1)
        for a,b in CONTRASTS:
            # Direct paired differences per clip, independently of the runner's per-arm simulations.
            block=count@(ind@(err[a][g]-err[b][g]))/denom[:,None]
            sim.setdefault((a,b),[]).append(block)
    for (a,b),sims in sim.items():
        row=agg['contrasts'][a+'_minus_'+b]
        near(np.mean([v[a]-v[b] for v in by.values()],axis=0),read(row['four_site_m']),1e-12)
        ci=np.percentile(np.mean(sims,axis=0),[2.5,97.5],axis=0).T
        near(ci,read(row['paired_clip_95_m']),1e-11)

def validate_support(clips,idx,ed,row):
    src=ed['source_class'][idx];p,c=(src==0).any(1),(src==1).any(1)
    typ=ed['encoded_type'][idx];possible=ed['eligible_person_car_and_excluded_coincident_counts'][idx]
    expect={'windows':len(idx),'clips':len(set(clips[idx])),'edges':int((src>=0).sum()),
            'velocity_valid_edges':int((ed['velocity_gap_ticks'][idx]>0).sum()),
            'truncated_windows':int((possible[:,:2].sum(1)>12).sum()),'edges_lost_to_cap':int(np.maximum(possible[:,:2].sum(1)-12,0).sum()),
            'within_radius_before_cap_person_windows':int((possible[:,0]>0).sum()),'within_radius_before_cap_car_windows':int((possible[:,1]>0).sum()),
            'near_coincident_excluded_rows':int(possible[:,2].sum()),
            'observed_types':{'pedestrian_or_unclassified_speed':int((typ==0).sum()),'history_slow_pedestrian':int((typ==2).sum()),'car':int((typ==6).sum())},
            'strata':{'empty':int((~p&~c).sum()),'person_only':int((p&~c).sum()),'car_only':int((~p&c).sum()),'mixed':int((p&c).sum()),
                      'with_any_person':int(p.sum()),'with_any_car':int(c.sum())}}
    assert expect==row

def batch_digest(inds):
    d=hashlib.sha256();r=torch.Generator().manual_seed(49)
    for _ in range(28):
        d.update(inds[torch.randperm(len(inds),generator=r).numpy()].astype('<i8').tobytes())
    return d.hexdigest()

def state_code(model):
    return ah({k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})

def verify(path,root,pose,repo,out,expected_sha):
    import zipfile
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    bundle=h(path.read_bytes());assert bundle==expected_sha
    core=[repo/'scripts/train_pinns_heldout_sites.py',repo/'scripts/benchmark_heldout_sites.py',*sorted((repo/'src').rglob('*.py'))]
    assert h(json.dumps([(p.relative_to(repo).as_posix(),h(p.read_bytes())) for p in core]).encode())==TREE
    keys,xy,fixed,graph,census=raw_labels(root);base=model_inputs(pose,keys,xy,fixed)
    clips=np.array([k[0] for k in keys]);sidx={s:np.flatnonzero(np.char.startswith(clips,s+'_')) for s in SITES}
    pairs=0;direct_paths=0;argmax=0;states=set();batch_checks={};max_path=max_p=max_err=0.;max_edge=0.
    with zipfile.ZipFile(path) as z:
        members={s+'_'+a+'.pt' for s in SITES for a in MODELS}|{'private.npz','aggregate.json','run_graph_control.py'}
        manifests=json.loads(z.read('sha256.json'))
        assert z.testzip() is None and len(z.namelist())==16 and set(z.namelist())==members|{'sha256.json'} and set(manifests)==members
        for n in manifests:assert h(z.read(n))==manifests[n]
        A=json.loads(z.read('aggregate.json'));identity=A['source']
        assert A['runner_sha256']==manifests['run_graph_control.py']==TRAIN_SHA and A['protocol_sha256']==LOCK
        assert A['protocol']=={'seed':42,'epochs':28,'batch_size':64,'learning_rate':.002,'min_lr_ratio':.05,'weight_decay':.0001,'gradient_clip':2.}
        assert (identity['pose_sha256'],identity['labels_calibration_sha256'],identity['ordered_keys_sha256'],identity['frozen_tree_sha256'],
                identity['typed_slot_structure_sha256'])==(POSE,RAW,IDS,TREE,CENSUS)
        b=A['backend'];assert b['device']=='cpu' and 1<=b['threads']<=4 and b['interop']==1 and b['determinism'] is True
        validate_support(clips,np.arange(len(keys)),graph,identity['all'])
        for s,g in sidx.items():validate_support(clips,g,graph,identity['by_site'][s])
        with np.load(io.BytesIO(z.read('private.npz')),allow_pickle=False) as q:
            rows={k:q[k] for k in q.files}
        assert set(rows)=={'clip','person_id','t0','fut_xy',*graph}|{m+f for m in METHODS for f in ('_paths','_probs','_errors')}
        assert keys==list(zip(rows['clip'].tolist(),rows['person_id'].tolist(),rows['t0'].tolist()))
        target=xy[:,5:].astype(np.float32);near(target,rows['fut_xy'])
        for k,v in graph.items():
            if k=='nbr_edges':max_edge=near(v,rows[k],1e-6,1e-6)
            else:np.testing.assert_array_equal(v,rows[k])
        assert ah({k:rows[k] for k in ('nbr_edges','nbr_mask','nbr_type_oh')})==identity['runtime_graph_sha256']
        err={m:rows[m+'_errors'] for m in METHODS}
        for s,g in sidx.items():
            train=np.concatenate([v for k,v in sidx.items() if k!=s]);assert not set(clips[train])&set(clips[g])
            # One identical minibatch permutation and one identical initialization across each paired arm's shared parameters.
            batch_checks[s]=bh=batch_digest(train)
            torch.manual_seed(42);prototype=UnstructuredShortPosePredictor('M1');common=state_code(prototype)
            torch.manual_seed(42);q=UnstructuredShortPosePredictor('M2b');q.load_state_dict({**q.state_dict(),**prototype.state_dict()});full=state_code(q)
            for arm in MODELS:
                model=UnstructuredShortPosePredictor('M1' if arm=='M1' else 'M2b')
                name=s+'_'+arm+'.pt';ck=torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=True)
                meta=A['folds'][s][arm]
                assert ck['arm']==arm and ck['site']==s and meta['weights']==name
                assert ck['batch_order_sha256']==meta['batch_order_sha256']==bh
                assert ck['shared_initial_sha256']==meta['shared_initial_sha256']==common
                assert ck['full_initial_sha256']==meta['full_initial_sha256']==(common if arm=='M1' else full)
                np.testing.assert_array_equal(ck['train_rows'].numpy(),train);np.testing.assert_array_equal(ck['test_rows'].numpy(),g)
                loss=ck['loss_per_batch'].numpy();assert np.isfinite(loss).all() and (loss>=0).all()
                assert len(train)==meta['train_windows'] and len(g)==meta['test_windows'] and len(loss)==28*((len(train)+63)//64)==meta['optimizer_steps']
                model.load_state_dict(ck['model_state'],strict=True);model.eval();state=state_code(model);assert state not in states;states.add(state)
                for m in ((arm,*INTERVENTIONS) if arm=='M2b_real' else (arm,)):
                    paths,prob=forward(model,m,base,graph,g)
                    sp,sq=rows[m+'_paths'][g],rows[m+'_probs'][g]
                    max_path=max(max_path,near(paths,sp,2e-4,2e-5));max_p=max(max_p,near(prob,sq,2e-5,2e-5))
                    w,e=scores(paths,prob,target[g]);ws,es=scores(sp,sq,target[g]);np.testing.assert_array_equal(w,ws)
                    near(es,err[m][g],1e-10);max_err=max(max_err,near(e,err[m][g],2e-4,2e-5));argmax+=len(g);pairs+=e.size;direct_paths+=len(g)*6
        assert len(states)==12 and argmax==1482*6
        for control,mask in [('real_edges_off',graph['source_class']>=0),('real_drop_people',graph['source_class']==0),('real_drop_car',graph['source_class']==1)]:
            u=~mask.any(1)
            for f in ('_paths','_probs','_errors'):np.testing.assert_array_equal(rows[control+f][u],rows['M2b_real'+f][u])
        summary(keys,sidx,graph['source_class'],err,A)
    result={'bundle_sha256':bundle,'frozen_runner_sha256':TRAIN_SHA,'auditor_sha256':h(Path(__file__).read_bytes()),'verified_fits':len(states),
            'windows':len(keys),'candidate_paths_compared_including_inference_controls':direct_paths,'top1_decisions_compared':argmax,
            'ADE_FDE_scalars_compared':pairs,'max_graph_channel_abs_diff':max_edge,'max_candidate_coordinate_abs_diff_m':max_path,
            'max_candidate_probability_abs_diff':max_p,'max_top1_ADE_FDE_abs_diff_m':max_err,'four_site_m':A['four_site_m'],'contrasts':A['contrasts'],
            'limits':'No refitting. The code checks reproducible recorded batch order, source and initial checksums, loss counts, resulting weights/predictions, not every historical optimizer operation.'}
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,dict)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('archive','metadata','pose','repo','output'):p.add_argument(k,type=Path)
    p.add_argument('--expected-sha256',required=True)
    o=p.parse_args();verify(o.archive,o.metadata,o.pose,o.repo,o.output,o.expected_sha256)
