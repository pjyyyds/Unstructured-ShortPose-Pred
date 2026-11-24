"""Create a private one-cell Kaggle CPU job from sealed local artifacts (no internet, images or fitting)."""
import base64,hashlib,json
from pathlib import Path
d=Path(__file__).resolve().parent
def mode_digest(entries):
    core=[[e['key'],*[{'reason':e[r]['reason'],'modes':e[r]['modes']} for r in ('first','independent')]] for e in entries]
    return hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':')).encode()).hexdigest()
values={}
for name in ('evaluate_visible_boundaries.py','PROTOCOL.md','inputs_manifest.json','observed_and_candidates.npz','sealed_targets.npz'):
    raw=(d/name).read_bytes();values[name]={'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
cell='''# PRIVATE Kaggle CPU cell. Embedded PINNS observations, forecasts and released coordinates must remain private.
import base64,hashlib,json,subprocess,sys
from pathlib import Path
values=PAYLOAD
expected_modes=DIGEST
base=Path('/kaggle/working/visible_boundary_trial'); inp=base/'inputs';out=base/'replay';inp.mkdir(exist_ok=True,parents=True)
def stage(name):
    v=values[name];raw=base64.b64decode(v['base64'],validate=True)
    assert hashlib.sha256(raw).hexdigest()==v['sha256']
    (inp/name).write_bytes(raw)
for name in ('evaluate_visible_boundaries.py','PROTOCOL.md','inputs_manifest.json','observed_and_candidates.npz'):stage(name)
cmd=[sys.executable,str(inp/'evaluate_visible_boundaries.py'),str(inp),str(out)]
subprocess.run(cmd,check=True)
entries=json.loads((out/'private_selected_modes.json').read_text())
core=[[e['key'],*[{'reason':e[r]['reason'],'modes':e[r]['modes']} for r in ('first','independent')]] for e in entries]
actual=hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert expected_modes==actual, (actual,expected_modes)
print('Historical support and all frozen choices verified before reading target arrays',actual,flush=True)
stage('sealed_targets.npz')
subprocess.run(cmd+['--score'],check=True)
dest=Path('/kaggle/working/visible_boundary_forecasts.zip')
dest.write_bytes((out/dest.name).read_bytes())
print('Finished PINNS visible-boundary trial:',dest.stat().st_size,hashlib.sha256(dest.read_bytes()).hexdigest(),flush=True)
'''
entries=json.loads((d/'blind/private_selected_modes.json').read_text())
cell=cell.replace('PAYLOAD',repr(values)).replace('DIGEST',repr(mode_digest(entries)))
p=d/'kaggle_cell.py';p.write_text(cell);compile(cell,str(p),'exec')
print('private cell',p.stat().st_size,'choice digest',mode_digest(entries),'source sha',hashlib.sha256(p.read_bytes()).hexdigest())
