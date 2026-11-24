import base64,hashlib,zlib
from pathlib import Path
d=Path(__file__).resolve().parent
src=['motion_cloud.py','shared_space_freeze.json','motion_and_detector_freeze.json','coverage.json']
enc={n:(hashlib.sha256((d/n).read_bytes()).hexdigest(),base64.b85encode(zlib.compress((d/n).read_bytes(),9)).decode()) for n in src}
template='''# Kaggle CPU with internet. Attach the published-to-this-private-task PINNS seed42 or five-seed weights archive.
# Rebuild the 3066 windows from the official source CSV and the checked presentation-time lattice of eleven original MP4s.
import base64,hashlib,json,os,subprocess,sys,tempfile,zipfile,zlib
from pathlib import Path
SOURCES=EMBED
REV='032800e94bb0828c45d3f5195ac5cc17aa93bc4c'
def sha(data):return hashlib.sha256(data).hexdigest()
def run():
 root=Path('/kaggle/input');out=Path('/kaggle/working/dut_shared_space_motion.zip')
 if not root.is_dir() or not out.parent.is_dir():raise RuntimeError('Use Kaggle CPU')
 def verify():
  with zipfile.ZipFile(out) as z:
   assert z.testzip() is None;m=json.loads(z.read('sha256.json'));assert set(m)==set(z.namelist())-{'sha256.json'} and all(sha(z.read(k))==v for k,v in m.items())
   a=json.loads(z.read('aggregate.json'));assert a['scorer_sha256']==SOURCES['motion_cloud.py'][0] and a['protocol_sha256']==SOURCES['motion_and_detector_freeze.json'][0] and a['axis_and_timing']['0']['all']['M0']['windows']==3066
 if out.exists():verify();print('Verified existing result',out,sha(out.read_bytes()));return
 with tempfile.TemporaryDirectory(prefix='dut_motion_') as tmp:
  tmp=Path(tmp);repo=tmp/'repo';code=tmp/'code';code.mkdir()
  for k,(h,b) in SOURCES.items():
   x=zlib.decompress(base64.b85decode(b));assert sha(x)==h;(code/k).write_bytes(x)
  subprocess.run(['git','init','-q',str(repo)],check=True)
  subprocess.run(['git','-C',str(repo),'fetch','-q','--depth=1','https://github.com/pjyyyds/Unstructured-ShortPose-Pred.git',REV],check=True)
  subprocess.run(['git','-C',str(repo),'checkout','-q','--detach','FETCH_HEAD'],check=True)
  assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==REV
  weights=root
  for name,digest in [('shortpose_heldout.zip','507c0a5decf79a4a48b982987e89c37b8c99e456f5c7f11cc3d49f34bd5741ec'),('shortpose_heldout_five_seeds.zip','0bf2745ab2d5ddcf5f5fbb5d330e6c6cc800625cec301f4c7105d031bdb49d2d')]:
   found=[p for p in root.rglob(name) if sha(p.read_bytes())==digest]
   if found:weights=found[0];break
  env={**os.environ,'PYTHONPATH':str(repo),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'}
  cmd=[sys.executable,'-u',str(code/'motion_cloud.py'),str(weights),str(out)]
  subprocess.run(cmd+['--check-only'],cwd=code,env=env,check=True);subprocess.run(cmd,cwd=code,env=env,check=True)
 verify();print('Verified external result',out,out.stat().st_size,sha(out.read_bytes()))
if __name__=='__main__':run()
'''
p=d/'kaggle_dut_shared_space_motion.py';p.write_text(template.replace('EMBED',repr(enc)));compile(p.read_text(),str(p),'exec');print(len(p.read_bytes()),hashlib.sha256(p.read_bytes()).hexdigest(),enc['motion_cloud.py'][0])
