"""Offline checks of the real history identities and a fully synthetic stored-prediction scoring path."""
import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from motion_cloud import fetch,SITES
from observed_graph import collect,key_sha,sha
from run_transfer import MATRICES,ARMS,RESERVE,score,load_weights
from verify_return import paired_check

D=Path(__file__).resolve().parent

def historical(root):
    keys,frames,h,g,a=collect(root,json.loads((D/'coverage.json').read_bytes()),fetch)
    alternate=json.loads((D/'independent_history.json').read_bytes())
    one=[];two=[];swap=[]
    for i,key in enumerate(keys):
        aa=[];bb=[]
        for j,kind in enumerate(g['source'][i]):
            if kind<0:break
            typed=['Car' if kind else 'Person',int(g['neighbor_id'][i,j])]
            aa.append(typed);bb.append(typed+[round(5*float(g['edges'][i,j,7])),int(g['lag'][i,j]>0),int(g['encoded_type'][i,j])])
        one.append([key,aa]);two.append([key,bb])
        if g['reserve_changed'][i]:swap.append([*key,int(g['reserve_last_id'][i]),int(g['neighbor_id'][i,-1])])
    computed=[key_sha(one),key_sha(two),key_sha(swap)]
    ref=alternate
    expected=[ref['ordered_memberships_sha256'],ref['ordered_members_and_history_sha256'],'eece02bf94166402fdcbd76bc1b0f6110015701c0601b2876ce06b40c09e371e']
    assert computed==expected and expected[-1] in json.dumps(alternate)
    assert json.loads(json.dumps(a))==json.loads((D/'historical_denominators.json').read_bytes())
    return computed

def synthetic():
    rng=np.random.default_rng(17);n=9
    keys=[('clip'+str(i%3),i,20) for i in range(n)]
    frames=np.tile(np.arange(35),(n,1))+16;ticks=frames*1000
    xy=rng.normal(size=(n,35,2));h=xy[:,:5];mask=np.ones((n,5),bool);mask[:2]=False
    g={'source':np.repeat([[-1, -1]],n,axis=0),'pre':np.zeros((n,3),int),'reserve_changed':np.arange(n)<2}
    g['source'][0]=[0,0];g['source'][1]=[0,0];g['source'][2]=[0,1];g['pre'][:3,1]=1
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        for angle in MATRICES:
            for s in SITES:
                arms=ARMS+(RESERVE,) if angle=='0' else ARMS;out={}
                for arm in arms:
                    p=rng.normal(size=(n,6,30,2)).astype(np.float32);q=rng.uniform(size=(n,6));q/=q.sum(1,keepdims=True)
                    if arm==RESERVE:
                        unchanged=~g['reserve_changed'];p[unchanged]=out['M2b_real_p'][unchanged];q[unchanged]=out['M2b_real_q'][unchanged]
                    out.update({arm+'_p':p,arm+'_q':q,arm+'_top':q.argmax(1)})
                np.savez(td/f'paths_{angle}_{s}.npz',**out)
        with patch('run_transfer.build',return_value=(keys,frames,ticks,xy)):agg=score(td,keys,frames,h,g,mask,td)
        assert agg['native_vehicle_slot']['affected']['windows']==2 and agg['native_vehicle_slot']['precap_any_car']['windows']==3
        with np.load(td/'errors_and_targets.npz') as saved:
            delta=(saved['0_'+RESERVE]-saved['0_M2b_real']).mean(0)
            paired_check(delta,g['reserve_changed'],np.array([k[0] for k in keys]),agg['native_vehicle_slot']['affected'])
        assert len(agg['all_axes'])==5 and len(agg['native_primary_and_secondary'])==6

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('weight_file',type=Path);arg=p.parse_args()
    print('Agreed historical and cell-replacement IDs',historical(arg.data))
    models,ids=load_weights(arg.weight_file);assert len(models)==12
    synthetic();print('12 real states loaded without an external forward. Synthetic end-to-end scoring and CI checks passed.')
