"""Build the private Kaggle CPU launcher from the registered implementation and shared initial weights."""
from pathlib import Path
import base64,hashlib,zlib
p=Path(__file__).resolve().parent
cell='''# PRIVATE Kaggle CPU. Attach the verified PINNS pose cache or original five-seed held-out ZIP.
from pathlib import Path
import base64,hashlib,os,subprocess,sys,zipfile,zlib
rev='66fb39d2de3df280ac831a4b5a0c0f095f1a0779'
repo=Path('/tmp/shortpose_equivariant_'+rev[:12])
subprocess.run(['git','init','-q',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'fetch','-q','--depth=1','https://github.com/pjyyyds/Unstructured-ShortPose-Pred.git',rev],check=True)
subprocess.run(['git','-C',str(repo),'checkout','--detach','-q',rev],check=True)
sys.path.insert(0,str(repo))
from scripts.stage_pinns_metadata import stage_metadata
data=stage_metadata(Path('/tmp/pinns_equivariant_metadata'))
digest=lambda b:hashlib.sha256(b).hexdigest()
pose_sha='b493a47536664ddb4fd66603c1bbf6b2c2e33f553520382027b3ae96b53447d0'
mount=Path('/kaggle/input')
pose=next((p for p in sorted(mount.rglob('pose.npz')) if digest(p.read_bytes())==pose_sha),None)
if pose is None:
    old=next((p for p in sorted(mount.rglob('shortpose_heldout_five_seeds.zip')) if p.stat().st_size==42019120 and digest(p.read_bytes())=='0bf2745ab2d5ddcf5f5fbb5d330e6c6cc800625cec301f4c7105d031bdb49d2d'),None)
    if old is None:raise ValueError('Missing registered PINNS pose cache or original private five-seed ZIP')
    with zipfile.ZipFile(old) as z:raw=z.read('pose.npz')
    assert digest(raw)==pose_sha
    pose=Path('/tmp/pinns_equivariant_pose.npz');pose.write_bytes(raw)
dst=Path('/tmp/pinns_equivariant_driver');dst.mkdir(exist_ok=True)
def restore(name,h,data):
    b=zlib.decompress(base64.b64decode(data));assert digest(b)==h
    (dst/name).write_bytes(b)
'''
for name in ('PROTOCOL.md','IMPLEMENTATION_CHECKS.md','equivariant_model.py','run_equivariant.py','test_equivariance.py','plan.json','initial_state.npz'):
    b=(p/name).read_bytes()
    cell+=f'restore({name!r},{hashlib.sha256(b).hexdigest()!r},{base64.b64encode(zlib.compress(b,9)).decode()!r})\n'
cell+='''env={**os.environ,'PYTHONPATH':str(repo),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2','OMP_NUM_THREADS':'2'}
subprocess.run([sys.executable,str(dst/'test_equivariance.py')],env=env,check=True)
out=Path('/kaggle/working/pinns_equivariant')
subprocess.run([sys.executable,'-u',str(dst/'run_equivariant.py'),str(data),str(pose),'--mode','fit','--plan',str(dst/'plan.json'),'--out',str(out)],env=env,check=True)
result=Path('/kaggle/working/equivariant_forecasts.zip')
result.write_bytes((out/result.name).read_bytes())
print('Verified all 16 training cells / received archive:',result.stat().st_size,digest(result.read_bytes()),flush=True)
'''
compile(cell,'kaggle_cell.py','exec')
out=p/'kaggle_cell.py';out.write_text(cell);print(out.stat().st_size,hashlib.sha256(out.read_bytes()).hexdigest())
