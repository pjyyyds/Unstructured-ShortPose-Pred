"""Independent timestamp-only replay of all four external demonstration sequences."""
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

root=Path(__file__).resolve().parent
expected=json.loads((root/'external_motion_lock.json').read_text())['cohort']
master,people={},{}
with tarfile.open(root/'imptc_samples.tar.gz','r|gz') as z:
    for f in z:
        fields=Path(f.name).parts
        if fields[1:]==('context','master_timestamp_sync.json'):
            master[fields[0]]=np.array(sorted(int(t) for t in json.load(z.extractfile(f))),dtype=np.int64)
        if len(fields)==4 and fields[1]=='vrus' and fields[-1]=='track.json':
            j=json.load(z.extractfile(f))
            if j['overview']['class_id']==0 and j['overview']['class_name'].lower() in ('person','pedestrian'):
                people[fields[:1]+fields[2:3]]=np.array([int(r['ts']) for r in j['track_data'].values() if r['status']==2])
keys=[]; stamps=[]; counts={s:0 for s in master}
for (seq,pid),accepted in sorted(people.items()):
    grid=master[seq]
    mask=np.isin(grid,accepted)
    # Sliding convolution finds every 86-tick uninterrupted accepted span.
    support=np.convolve(mask.astype(int),np.ones(86,dtype=int),'valid')
    end=-1
    for i in np.flatnonzero(support==86):
        if i%5 or i<=end: continue
        anchor=grid[i+10]
        target=anchor+np.arange(-4,31)*100000
        # Global absolute-distance minimization; argmin selects the earliest time on ties.
        chosen=np.argmin(np.abs(grid[:,None]-target[None,:]),axis=0)
        if (len(np.unique(chosen))!=35 or (chosen[[0,4,34]]!=[i,i+10,i+85]).any()
                or np.max(np.abs(grid[chosen]-target))>20000):continue
        keys.append([seq,pid,int(anchor)]); stamps.append(grid[chosen].tolist()); counts[seq]+=1
        end=i+85
def sha(obj):return hashlib.sha256(json.dumps(obj,separators=(',',':'),sort_keys=True).encode()).hexdigest()
assert sha(stamps)==expected['all_selected_timestamps_sha256'] and sha(keys)==expected['ordered_window_keys_sha256']
assert counts=={s:v['windows'] for s,v in expected['per_sequence'].items()} and len(keys)==564
assert len(people)==61
print('Independent UTC matrix matched every timestamp of 564 windows. No coordinates or prediction inspected.')
