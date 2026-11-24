"""Independent synthetic CPU checks for associativity and the two published FLOP accounting rules."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.utils.flop_counter import FlopCounterMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

class Census(TorchDispatchMode):
    def __init__(self):self.table=defaultdict(lambda:{'calls':0,'float_output_elements':0,'reduce_float_adds_lower_bound':0})
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        res=func(*args,**(kwargs or {}));row=self.table[str(func)]
        row['calls']+=1
        row['float_output_elements']+=sum(x.numel() for x in tree_leaves(res) if isinstance(x,torch.Tensor) and x.is_floating_point())
        if str(func).startswith('aten.sum.') and isinstance(args[0],torch.Tensor) and args[0].is_floating_point() and res.is_floating_point():
            row['reduce_float_adds_lower_bound']+=max(0,args[0].numel()-res.numel())
        return res

def run(repo):
    sys.path.insert(0,str(repo))
    from src.models.predictor import UnstructuredShortPosePredictor
    from src.models.pose_inference import prepare_pose_inference
    from tests.test_invariants import TestMathematicalInvariants
    from scripts.count_forward_ops import count_float_elements
    torch.set_num_threads(2);torch.manual_seed(23)
    ref=UnstructuredShortPosePredictor('Ours_Full').eval()
    opt=prepare_pose_inference(ref)
    make=TestMathematicalInvariants()._dummy_batch
    weights={name:format(t.data_ptr(),'x') for name,t in ref.named_parameters()}
    assert dict(ref.named_parameters()).keys()==dict(opt.named_parameters()).keys()
    assert all(t.data_ptr()!=int(weights[n],16) and torch.equal(t,dict(ref.named_parameters())[n]) for n,t in opt.named_parameters())
    layers=[(b.nbr_lin.in_features,b.nbr_lin.out_features) for b in ref.pose_enc.children() if hasattr(b,'nbr_lin')];assert layers==[(6,32),(32,64),(64,128)]
    out={'weights':'Random synthetic weights; no real case read or evaluated.','torch':str(torch.__version__),'platform':sys.platform,'channels':layers,'parameters':sum(v.numel() for v in ref.parameters()),'torch_default_target':'CPU, dense, float32 weights; floating-point multiply-add convention 2','durations':{},'source_sha256':{n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in ('src/models/pose_inference.py','src/models/pose_stgcn.py','src/models/predictor.py','scripts/count_forward_ops.py')}}
    for t in (5,10,20):
        data=make(bsz=12,seq_t=t);measured={};per=[2*12*t*17*17*(o-i) for i,o in layers]
        for name,model in [('original',ref),('optimized',opt)]:
            measured[name]={}
            for kind,mapper in [('default',{}),('extended',dict.fromkeys((torch.ops.aten.add,torch.ops.aten.sub,torch.ops.aten.mul),count_float_elements))]:
                with FlopCounterMode(display=False,custom_mapping=mapper) as mode: outputs=model(data)
                measured[name][kind]={'total':mode.get_total_flops(),'operators':{str(k):n for k,n in mode.flop_counts['Global'].items()}}
            if t==20:
                census=Census()
                with census:model(data)
                measured[name]['aten_forward']=dict(census.table)
        delta=sum(per)
        for k in ('default','extended'):assert measured['original'][k]['total']-measured['optimized'][k]['total']==delta
        out['durations'][str(t)]={'synthetic_counts':measured,'spatial_reduction_by_layer':per,'spatial_reduction':delta}
    try:prepare_pose_inference(UnstructuredShortPosePredictor('M0').eval())
    except Exception as exc:out['motion_only_transform']=type(exc).__name__+': '+str(exc)
    else:out['motion_only_transform']='accepted'
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    r=run(a.repo);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print([(t,[(v['synthetic_counts'][n]['default']['total'],v['synthetic_counts'][n]['extended']['total']) for n in ('original','optimized')]) for t,v in r['durations'].items()])
