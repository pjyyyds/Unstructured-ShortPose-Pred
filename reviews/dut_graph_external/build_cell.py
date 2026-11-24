"""Private Kaggle CPU graph transfer launcher. No external trajectories, video, or model weights are embedded."""
import base64
import hashlib
from pathlib import Path
import zlib

D=Path(__file__).resolve().parent
names=('run_transfer.py','observed_graph.py','test_transfer.py','motion_cloud.py','fusion_cloud.py','coverage.json','shared_space_freeze.json','motion_and_detector_freeze.json','historical_denominators.json','PROTOCOL.md','HANDEDNESS_ADDENDUM.md','VEHICLE_SLOT_ADDENDUM.md')
sources={n:(hashlib.sha256((D/n).read_bytes()).hexdigest(),base64.b85encode(zlib.compress((D/n).read_bytes(),9)).decode()) for n in names}
launcher='''# Private Kaggle notebook, Internet on; prediction subprocess always uses CPU.
import base64,hashlib,json,os,subprocess,sys,tempfile,zipfile,zlib
from pathlib import Path
SOURCES=__SOURCE_CONTENT__
REV='032800e94bb0828c45d3f5195ac5cc17aa93bc4c'
WEIGHTS_SHA='0b3e590b8d7cb2bf99670fd27d3bdbf2a40fa0e77bf29f7ac0423b73535589f8'
def sha(b):return hashlib.sha256(b).hexdigest()
def run():
    inputs=Path('/kaggle/input');work=Path('/kaggle/working')
    if not inputs.is_dir() or not work.is_dir():raise RuntimeError('Use a private Kaggle notebook')
    def artifact(name,expected=None):
        local=work/name
        hits=([local] if local.is_file() else [])+sorted(inputs.rglob(name))
        if not hits:raise FileNotFoundError('Attach the original private archive '+name)
        digests={sha(p.read_bytes()) for p in hits}
        if len(digests)!=1 or (expected is not None and digests!={expected}):raise ValueError('Ambiguous/incorrect archive '+name)
        return hits[0],digests.pop()
    weights,_=artifact('pinns_graph_control.zip',WEIGHTS_SHA);cache,cache_hash=artifact('dut_shared_space_pose_cache.zip')
    output=work/'dut_shared_space_graph_transfer.zip'
    def checked():
        with zipfile.ZipFile(output) as z:
            assert z.testzip() is None and len(set(z.namelist()))==len(z.namelist())
            manifest=json.loads(z.read('sha256.json'));assert set(manifest)==set(z.namelist())-{'sha256.json'}
            for n in manifest:assert sha(z.read(n))==manifest[n]
            r=json.loads(z.read('aggregate.json'))
            assert r['source_training_archive_sha256']==WEIGHTS_SHA and r['pose_cache_sha256']==cache_hash
            for n,h in r['runner_and_protocol'].items():assert SOURCES[n][0]==h==manifest['sources/'+n]
            assert r['conditions']==['0','90','180','270','mirror_y'] and r['native_vehicle_slot']['affected']['windows']==810
    if output.exists():
        checked();print('Reused verified output',output,sha(output.read_bytes()),flush=True);return
    with tempfile.TemporaryDirectory(prefix='dut_graph_') as temp:
        temp=Path(temp);code=temp/'runner';code.mkdir();repo=temp/'repo'
        for name,(h,encoded) in SOURCES.items():
            raw=zlib.decompress(base64.b85decode(encoded));assert sha(raw)==h;(code/name).write_bytes(raw)
        subprocess.run(['git','init','-q',str(repo)],check=True)
        subprocess.run(['git','-C',str(repo),'fetch','-q','--depth=1','https://github.com/pjyyyds/Unstructured-ShortPose-Pred.git',REV],check=True)
        subprocess.run(['git','-C',str(repo),'checkout','-q','--detach','FETCH_HEAD'],check=True)
        assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==REV
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(repo),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'}
        subprocess.run([sys.executable,'-u',str(code/'test_transfer.py')],cwd=code,env=env,check=True)
        cmd=[sys.executable,'-u',str(code/'run_transfer.py'),'--data',str(temp/'author_files'),'--weights',str(weights),'--cache',str(cache),'--output',str(output)]
        subprocess.run(cmd,cwd=code,env=env,check=True)
    checked();print('Verified',output,output.stat().st_size,sha(output.read_bytes()),flush=True)
if __name__=='__main__':run()
'''
out=D/'kaggle_cell.py';text=launcher.replace('__SOURCE_CONTENT__',repr(sources));compile(text,str(out),'exec');out.write_text(text)
print(out.stat().st_size,hashlib.sha256(out.read_bytes()).hexdigest())
