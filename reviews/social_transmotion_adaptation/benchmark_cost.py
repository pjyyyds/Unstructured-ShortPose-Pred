"""Estimate the fixed fit budget on synthetic data only, on the current CPU."""
import json
import os
from pathlib import Path
import platform
import statistics
import time

import torch
from check_protocol import ROOT, model_class

torch.set_num_threads(min(4,os.cpu_count() or 1))
torch.set_num_interop_threads(1)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
cls,_=model_class(ROOT/'source/model_jrdb.py',True)
model=cls(tok_dim=40,nhid=128,nhead=4,dim_feedfwd=1024,nlayers_local=6,nlayers_global=3,
          dropout=.1,output_scale=1,obs_and_pred=35,num_tokens=2,device='cpu').train()
opt=torch.optim.Adam(model.parameters(),lr=1e-4)
x=torch.randn(16,5,2,4)
truth=torch.randn(16,30,2)
pad=torch.zeros(16,1,dtype=torch.bool)
cost=[]
for _ in range(6):
    started=time.perf_counter()
    opt.zero_grad(set_to_none=True)
    path=model(x,pad)[:,5:,0]
    loss=(100*torch.linalg.vector_norm(path-truth,dim=-1).mean())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1)
    opt.step()
    cost.append(time.perf_counter()-started)
seconds=statistics.median(cost[1:])
updates=2*28*sum((n+15)//16 for n in (666,1398,1050,1332))
result=dict(hardware=platform.machine(),torch=str(torch.__version__),threads=torch.get_num_threads(),
            synthetic_batch=16,updates_in_benchmark=6,skipped_warmup_steps=1,seconds_per_batch=cost,
            post_warmup_median_seconds=seconds,updates_planned=updates,
            same_machine_training_projection_hours=seconds*updates/3600,
            interpretation='Synthetic single-machine extrapolation. Kaggle benchmarks or guarantees unavailable.')
(ROOT/'synthetic_cost.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
