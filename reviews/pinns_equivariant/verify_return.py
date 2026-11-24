"""Independent CPU inference from returned tensors, with an explicit GRU recurrence and metric reconstruction."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scripts.benchmark_heldout_sites import load_site_windows
from scripts.train_pinns_heldout_sites import tensorize
from scripts.verify_heldout_bundle import parameter_digest,IDS_SHA,LABELS_SHA
from scripts.run_heldout_seed_sensitivity import json_digest

ARMS=('global_motion','global_pose','local_motion','local_pose')
SCORES=('ade_m','fde_m','minade6_m','minfde6_m','mode0_selected','distinct_mean_pair_m')
def sha(b):return hashlib.sha256(b).hexdigest()
def near(x,y,tol=1e-9):
    a,b=np.asarray(x),np.asarray(y);assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_allclose(a,b,atol=tol,rtol=0)
    return float(np.max(np.abs(a-b),initial=0))
def gru(sequence,state,name):
    v=sequence.new_zeros(len(sequence),state[name+'.weight_hh_l0'].shape[1])
    wi,wh,bi,bh=[state[name+'.'+k] for k in ('weight_ih_l0','weight_hh_l0','bias_ih_l0','bias_hh_l0')]
    gx=F.linear(sequence,wi,bi)
    for step in range(sequence.shape[1]):
        ir,iz,inn=gx[:,step].chunk(3,1);hr,hz,hn=F.linear(v,wh,bh).chunk(3,1)
        r=(ir+hr).sigmoid();z=(iz+hz).sigmoid();n=(inn+r*hn).tanh();v=(1-z)*n+z*v
    return v
def linear(data,state,prefix):return F.linear(data,state[prefix+'.weight'],state[prefix+'.bias'])

@torch.inference_mode()
def independent(arm,state,xy,xp,q):
    n=len(xy);basis=np.zeros((n,2,2));ok=np.zeros(n,bool);rel=xy-xy[:,-1:]
    for i in range(n):
        for j in (3,2,1,0):
            v=-rel[i,j];length=np.hypot(*v)
            if length>=.05-1e-12:
                ok[i]=True
                if arm.startswith('local'):
                    x,y=v/length;basis[i]=[[x,-y],[y,x]]
                else:basis[i]=np.eye(2)
                break
    z=rel@basis;vel=np.diff(z,axis=1,prepend=z[:,:1])/.1
    motion=torch.from_numpy(np.concatenate((z,vel,np.hypot(vel[:,:,0],vel[:,:,1])[:,:,None]),axis=2).astype(np.float32))
    hm=gru(motion,state,'motion')
    if arm.endswith('pose'):
        feature=torch.from_numpy(xp);g=gru(F.silu(linear(feature.flatten(2),state,'pose_input')),state,'pose')
        pose_out=linear(torch.cat((g,torch.from_numpy(q)),1),state,'pose_out')
        hm+=pose_out*(feature[:,:,:,3]>.5).any(2).any(1)[:,None]
    c=F.layer_norm(hm,(48,),state['context.0.weight'],state['context.0.bias'])
    c=F.silu(linear(c,state,'context.1'));logits=linear(c,state,'scores')
    w=logits.softmax(1).numpy();w[~ok]=np.float32(1/6)
    moving=linear(c,state,'residual').reshape(n,5,3,2).numpy().astype(float)
    knot=np.concatenate((np.zeros((n,5,1,2)),moving),2)
    times=np.arange(1,31);cv=xy[:,-1:]+np.diff(xy[:,-2:],axis=1)*times[None,:,None]
    out=np.repeat(cv[:,None],6,axis=1)
    for t in range(30):
        # Integer tick interpolation on [0,10], [10,20], [20,30].
        k=t//10;lam=(t+1-10*k)/10.
        loc=(1-lam)*knot[:,:,k]+lam*knot[:,:,k+1]
        out[:,1:,t]+=loc@basis.transpose(0,2,1)
    return out,w,ok

def scores(p,q,y):
    ds=np.hypot(*(p-y[:,None]).transpose(3,0,1,2));a=ds.mean(2);f=ds[:,:,-1]
    ii=np.arange(len(p));k=q.argmax(1)
    pairs=[np.hypot(*(p[:,i]-p[:,j]).transpose(2,0,1)).mean(1) for i in range(6) for j in range(i)]
    return np.stack([a[ii,k],f[ii,k],np.min(a,1),np.min(f,1),k==0,np.mean(pairs,0)],1)

def verify(bundle,root,pose,registered):
    torch.set_num_threads(2)
    plan=json.loads((registered/'plan.json').read_text())
    assert sha(pose.read_bytes())==plan['pose_sha256']
    groups,_,data_sha=load_site_windows(root);assert data_sha==LABELS_SHA
    raw,sites,keys,_,_,_=tensorize(groups,pose);assert json_digest(keys)==IDS_SHA
    xy=np.stack([w.xy for s in sorted(groups) for w in groups[s]])
    xp,q=raw['x_p'].numpy(),raw['q_vec'].numpy()
    result={a:np.full((len(keys),6),np.nan) for a in (*ARMS,'last_two_cv')}
    gaps={'candidate_m':0.,'probability':0.,'reported_row_metric':0.};rot_gaps={};valid=np.zeros(len(keys),bool)
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        digests=json.loads(z.read('sha256.json'))
        assert set(digests)==set(z.namelist())-{'sha256.json'} and all(sha(z.read(k))==v for k,v in digests.items())
        for n in ('equivariant_model.py','run_equivariant.py','test_equivariance.py','PROTOCOL.md','IMPLEMENTATION_CHECKS.md','initial_state.npz','plan.json'):
            assert (registered/n).read_bytes()==z.read(n)
        r=json.loads(z.read('equivariant_results.json'));assert r['plan']==plan and r['fits']==16
        folds=json.loads(z.read('folds.json'));assert json_digest(folds)==plan['folds_sha256']
        for site,idx in sites.items():
            others=sorted(set(sites)-{site});fold=folds[site]
            assert list(map(tuple,fold['test_keys']))==[keys[i] for i in idx] and fold['train_sites']==others
            assert list(map(tuple,fold['train_keys']))==[keys[i] for s in others for i in sites[s]]
            h=xy[idx,:5];y=xy[idx,5:];orders=None
            with np.load(io.BytesIO(z.read(site+'/targets.npz')),allow_pickle=False) as trg:
                assert list(zip(trg['clip'],trg['person_id'],trg['t0']))==[keys[i] for i in idx]
                np.testing.assert_equal(trg['truth'],y);np.testing.assert_equal(trg['history'],h);valid[idx]=trg['valid']
            cv=h[:,-1:]+np.diff(h[:,-2:],axis=1)*np.arange(1,31)[None,:,None]
            result['last_two_cv'][idx]=scores(np.repeat(cv[:,None],6,1),np.full((len(h),6),1/6),y)
            for arm in ARMS:
                stem=site+'/'+arm
                ck=torch.load(io.BytesIO(z.read(stem+'.pt')),weights_only=True,map_location='cpu');state=ck['model_state']
                assert ck['site']==site and ck['arm']==arm and ck['seed']==42 and ck['initial_sha256']==plan['same_initial_state_sha256']
                assert ck['fold_sha256']==json_digest(fold) and ck['parameter_sha256']==parameter_digest(state)
                log=ck['log'];assert [a['epoch'] for a in log]==list(range(1,29))
                ordered=[a['ordered_indices_sha256'] for a in log]
                if orders is not None:assert orders==ordered
                orders=ordered
                p,prob,ok=independent(arm,state,h,xp[idx],q[idx]);assert np.array_equal(ok,valid[idx])
                with np.load(io.BytesIO(z.read(stem+'.npz')),allow_pickle=False) as archive:
                    gaps['candidate_m']=max(gaps['candidate_m'],near(p,archive['trajs'],2e-4))
                    gaps['probability']=max(gaps['probability'],near(prob,archive['probs'],2e-5))
                    assert np.array_equal(prob.argmax(1),archive['probs'].argmax(1))
                    near(p[:,0],cv,1e-12)
                    stored=scores(archive['trajs'],archive['probs'],y)
                    gaps['reported_row_metric']=max(gaps['reported_row_metric'],near(stored,archive['errors'],1e-10))
                    result[arm][idx]=stored
                local_gaps=[]
                for angle in (37,90,173):
                    a=np.deg2rad(angle);c,s=np.cos(a),np.sin(a);mat=np.array([[c,-s],[s,c]]);translation=np.array([13.25,-9.5])
                    pp,qq,oo=independent(arm,state,h@mat.T+translation,xp[idx],q[idx])
                    assert np.array_equal(ok,oo)
                    cp=float(np.max(np.abs(pp-(p@mat.T+translation))));ap=float(np.max(np.abs(qq-prob)));sw=int(np.sum(qq.argmax(1)!=prob.argmax(1)))
                    cloud=r['ground_coordinate_tests'][site][arm][str(angle)]
                    near(cp,cloud['max_mode_indexed_coordinate_error_m'],4e-4);near(ap,cloud['max_probability_error'],4e-5)
                    if arm.startswith('local'):assert cp<1e-5 and ap<2e-6 and sw==cloud['top_mode_changes']==0
                    local_gaps.append([cp,ap,sw])
                rot_gaps.setdefault(site,{})[arm]=np.max(local_gaps,0).tolist()
    for label,mask in [('all',np.ones(len(keys),bool)),('identifiable',valid),('degenerate',~valid)]:
        g=r['models'][label]
        for arm,e in result.items():
            means=[]
            for s,idx in sites.items():
                selected=idx[mask[idx]];m=g['by_site'][s]
                assert len(selected)==m['windows']
                if len(selected):
                    v=e[selected].mean(0);near(v,[m['metrics'][arm][k] for k in SCORES]);means.append(v)
            if means:near(np.mean(means,0),[g['site_equal'][arm][k] for k in SCORES])
    rng=np.random.default_rng(2026);base=[]
    for s,idx in sites.items():
        names,inv=np.unique([keys[i][0] for i in idx],return_inverse=True);n=len(names)
        sample=rng.multinomial(n,np.ones(n)/n,size=20000)
        base.append((idx,inv,sample,sample@np.bincount(inv)))
    for contrast,v in r['paired'].items():
        a,b=contrast.split('_minus_'); diff=result[a][:,:2]-result[b][:,:2];average=[];rep=np.zeros((20000,2))
        for idx,inv,w,denom in base:
            accum=np.zeros((w.shape[1],2));np.add.at(accum,inv,diff[idx]);average.append(diff[idx].mean(0));rep+=w@accum/denom[:,None]/4
        near(np.mean(average,0),[v['mean'][k] for k in SCORES[:2]])
        near(np.percentile(rep,[2.5,97.5],axis=0).T,[v['paired_clip_95'][k] for k in SCORES[:2]])
    return {'archive_sha256':sha(bundle.read_bytes()),'private_window_keys_sha256':IDS_SHA,'fits':16,'model_windows_verified':1482*4,'mode_paths_verified':1482*4*6,'rotation_tests_per_fit':3,'max_local_vs_returned':gaps,'orbit_by_site_arm_max_path_prob_top_switches':rot_gaps,'primary':r['paired']['local_pose_minus_local_motion']}
if __name__=='__main__':
    ps=argparse.ArgumentParser(description=__doc__)
    for k in ('bundle','data','pose','registered','output'):ps.add_argument(k,type=Path)
    a=ps.parse_args();v=verify(a.bundle,a.data,a.pose,a.registered);a.output.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v['primary']))
