import base64,hashlib,zlib
from pathlib import Path
d=Path(__file__).resolve().parent
names=['motion_cloud.py','detector_pilot.py','shared_space_freeze.json','motion_and_detector_freeze.json','coverage.json','author_video_catalog.json']
enc={k:(hashlib.sha256((d/k).read_bytes()).hexdigest(),base64.b85encode(zlib.compress((d/k).read_bytes(),9)).decode()) for k in names}
template="""# Kaggle with Internet. Full source and all runtime inputs obtained from the authors. CPU pilot or GPU/T4 for full 1656-frame census.
# FULL=False uses the 11 preregistered pilot windows, without trajectory scoring.
FULL=FULL_FLAG
import base64,hashlib,importlib.util,json,os,subprocess,sys,tempfile,zipfile,zlib
from pathlib import Path
SOURCES=EMBED
REV='032800e94bb0828c45d3f5195ac5cc17aa93bc4c'
def sha(b):return hashlib.sha256(b).hexdigest()
def run():
 root=Path('/kaggle/input');out=Path('/kaggle/working')/('dut_shared_space_pose_cache.zip' if FULL else 'dut_shared_space_detector_pilot.zip')
 if not root.is_dir() or not out.parent.is_dir():raise RuntimeError('Launch on Kaggle')
 def verify():
  with zipfile.ZipFile(out) as z:
   m=json.loads(z.read('sha256.json'));assert z.testzip() is None and set(m)==set(z.namelist())-{'sha256.json'} and all(sha(z.read(k))==v for k,v in m.items())
   a=json.loads(z.read('aggregate.json'));assert a['script_sha256']==SOURCES['detector_pilot.py'][0] and a['windows']==(3066 if FULL else 11)
 if out.exists():verify();print('Reusing verified',out,sha(out.read_bytes()));return
 with tempfile.TemporaryDirectory(prefix='dut_pose_') as tmp:
  tmp=Path(tmp);repo=tmp/'repo';code=tmp/'code';code.mkdir()
  for name,(h,s) in SOURCES.items():
   b=zlib.decompress(base64.b85decode(s));assert sha(b)==h;(code/name).write_bytes(b)
  subprocess.run(['git','init','-q',str(repo)],check=True);subprocess.run(['git','-C',str(repo),'fetch','-q','--depth=1','https://github.com/pjyyyds/Unstructured-ShortPose-Pred.git',REV],check=True);subprocess.run(['git','-C',str(repo),'checkout','-q','--detach','FETCH_HEAD'],check=True);assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==REV
  sys.path.insert(0,str(repo))
  from scripts.run_heldout_cloud import MODEL_URL,MODEL_SHA,download_zip
  pose=next((p for p in root.rglob('*.onnx') if p.stat().st_size==54330655 and sha(p.read_bytes())==MODEL_SHA),None)
  if pose is None:
   packed=tmp/'official_pose.zip';download_zip(MODEL_URL,packed,50799818)
   with zipfile.ZipFile(packed) as z:
    candidates=[n for n in z.namelist() if Path(n).name=='end2end.onnx' and '__MACOSX' not in Path(n).parts];assert len(candidates)==1;b=z.read(candidates[0]);assert sha(b)==MODEL_SHA;pose=tmp/'rtm.onnx';pose.write_bytes(b)
  if importlib.util.find_spec('onnxruntime') is None:subprocess.run([sys.executable,'-m','pip','-q','install','onnxruntime'],check=True)
  env={**os.environ,'PYTHONPATH':str(repo),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'}
  subprocess.run([sys.executable,'-u',str(code/'detector_pilot.py'),str(pose),str(out)]+(['--full'] if FULL else []),cwd=code,env=env,check=True)
 verify();print('Verified',out,out.stat().st_size,sha(out.read_bytes()))
if __name__=='__main__':run()
"""
for full,suffix in [(False,'detector_pilot'),(True,'pose_cache')]:
 p=d/('kaggle_dut_'+suffix+'.py');p.write_text(template.replace('EMBED',repr(enc)).replace('FULL_FLAG',str(full)));compile(p.read_text(),str(p),'exec');print(p.name,len(p.read_bytes()),hashlib.sha256(p.read_bytes()).hexdigest())
