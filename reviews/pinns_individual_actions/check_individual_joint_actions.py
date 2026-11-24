"""Synthetic inputs only: common initial state, forward and gradient isolation, paired summaries."""
from pathlib import Path
import tempfile
import numpy as np
import torch
import run_individual_joint_actions as j
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
torch.manual_seed(19381)
n=12; dims=(n,5,17,6)
x={'x_m':torch.randn(n,5,9),'x_p':torch.randn(dims),'q_vec':torch.rand(n,4),'ego_type_oh':torch.eye(8)[torch.arange(n)%8]}
x['x_m'][:,:,5:7]=1
x['x_p'][:,:,:,3]=1
a={k:j.model_for(k) for k in j.ARMS};states={k:m.state_dict() for k,m in a.items()}
for i,st in states.items():
    for j2,st2 in states.items():
        for k in st.keys()&st2.keys(): torch.testing.assert_close(st[k],st2[k],rtol=0,atol=0)
assert not hasattr(a['no_head'].decoder,'act_head')
assert 'cond_mlp.0.weight' not in a['auxiliary'].decoder.state_dict()
assert set(states['end_to_end'])==set(states['detached'])
for name,m in a.items():
    assert not m.use_map and not m.use_graph and m.fusion_mode=='M1'
for training in (False,True):
    result={}
    for name,m in a.items():
        m.train(training)
        torch.manual_seed(342)
        result[name]=m(x)
    for name in ('trajs','mode_logits','mode_probs'):
        torch.testing.assert_close(result['no_head'][name],result['auxiliary'][name],rtol=0,atol=0)
    for key in ('trajs','act_probs','act_logits','mode_probs'):
        torch.testing.assert_close(result['end_to_end'][key],result['detached'][key],rtol=0,atol=0)
for name,m in a.items():
    m.eval();m.zero_grad(set_to_none=True)
    p=m(x)
    objective=p['trajs'].square().mean()+p['mode_logits'].square().mean()
    objective.backward()
    if name!='no_head':
        grad=m.decoder.act_head[0].weight.grad
        assert (grad is not None and grad.abs().sum().item()>0) == (name=='end_to_end')
    if name in ('detached','end_to_end'):
        assert m.decoder.cond_mlp[0].weight.grad.abs().sum()>0
    assert m.motion_enc.gru.weight_ih.grad.abs().sum()>0 if hasattr(m.motion_enc,'gru') and hasattr(m.motion_enc.gru,'weight_ih') else any(p.grad is not None and p.grad.abs().sum()>0 for p in m.motion_enc.parameters())
    if name!='no_head':
        m.zero_grad(set_to_none=True);pred=m(x);torch.nn.functional.binary_cross_entropy_with_logits(pred['act_logits'],torch.rand(n,5)).backward()
        assert m.decoder.act_head[0].weight.grad.abs().sum()>0
truth=torch.randn(n,30,2);data=dict(x,fut_xy=truth,fut_mask=torch.ones(n,30),y_act=(torch.rand(n,5)>.5).float(),m_act=torch.ones(n))
idx={s:np.arange(i*3,(i+1)*3) for i,s in enumerate(sorted(j.SITE_COUNTS))}
keys=[(f'c{i//3}',i,30) for i in range(n)]
with tempfile.TemporaryDirectory() as t:
    out=Path(t);plan=j.design(data,idx,keys);j.save_json(out/'preregistration.json',plan)
    (out/'driver.py').write_bytes(Path(j.__file__).read_bytes())
    evaluated={}
    for site in idx:
        folder=out/site;folder.mkdir()
        for name in j.ARMS:
            identity={'registration_sha256':j.json_digest(plan),'heldout':site,'arm':name,'mode':j.ARMS[name],'training_backend':{'device':'cpu'},
                      **{k:plan['folds'][site][k] for k in ('train_ids_sha256','test_ids_sha256','training_positive_weights')},
                      'initial_state_sha256':j.parameter_digest(states[name])}
            torch.save({'identity':identity,'model_state':states[name]},folder/(name+'.pt'))
        evaluated[site]=j.run_fold(out,plan,{'device':'cpu'},data,idx,keys,site,False)
        j.run_fold(out,plan,{'device':'cpu'},data,idx,keys,site,True)
    summary=j.aggregate(evaluated);assert set(summary['four_site_macro'])==set(j.ARMS)
    assert summary['primary_paired']['ade_m']['paired_clip_95pct']==[0,0]
    assert summary['primary_paired']['brier']['detached_minus_e2e']==0
    j.bundle(out,sorted(idx),plan,summary)
    import zipfile
    with zipfile.ZipFile(out/'joint_actions_all.zip') as z:
        assert z.testzip() is None and len([n for n in z.namelist() if n.endswith('.pt')])==16
        for site in idx:
            assert site+'/targets.npz' in z.namelist()
print('PASS: synthetic initial parameters; paired dropout and forwards; regression/head/bridge gradients; all 16 mocked states, 4 folds, archive and independent-batch replay. No optimization.')
