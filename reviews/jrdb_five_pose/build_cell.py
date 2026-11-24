"""Build one private standalone notebook cell with all frozen experimental code, no person records."""
import base64
import hashlib
from pathlib import Path
import zlib
D=Path(__file__).resolve().parent
files=('PROTOCOL.md','input_support.json','check_intersection.py','run_five_source_poses.py','prior_native_residual.py','test_five_source.py','author_members.py','evaluate_outdoor_motion.py','check_outdoor_projection.py','check_training_coordinates.py')
older = D.parent / "jrdb_development"
shared = {name: older / name for name in ("evaluate_outdoor_motion.py", "check_outdoor_projection.py", "check_training_coordinates.py")}
shared["prior_native_residual.py"] = older / "native_pose_residual/run_jrdb_pose_residual.py"
sources = {}
for name in files:
    payload = shared.get(name, D / name).read_bytes()
    sources[name] = hashlib.sha256(payload).hexdigest(), base64.b85encode(zlib.compress(payload, 9)).decode()

src="""# Private Kaggle CPU notebook. Internet enabled. No raw JRDB images or account needed for these author-released derivatives.
import base64,hashlib,json,os,subprocess,sys,tempfile,zipfile,zlib
from pathlib import Path
SOURCES=__SOURCES__
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    work=Path('/kaggle/working');out=work/'jrdb_five_source_poses.zip'
    if not work.is_dir():raise RuntimeError('Use Kaggle CPU')
    def validate():
        with zipfile.ZipFile(out) as z:
            assert z.testzip() is None
            m=json.loads(z.read('sha256.json'));assert len(z.namelist())==len(set(z.namelist())) and set(m)==set(z.namelist())-{'sha256.json'}
            for k,h in m.items():assert sha(z.read(k))==h
            for k in ('PROTOCOL.md','input_support.json','check_intersection.py','run_five_source_poses.py','prior_native_residual.py'):assert m['sources/'+k]==SOURCES[k][0]
            r=json.loads(z.read('aggregate.json'));assert r['protocol_sha256']==SOURCES['PROTOCOL.md'][0] and r['alpha']==1
            assert sorted(x['test_windows'] for x in r['folds'].values())==[6,68,123]
            assert r['support']['ordered_keys_sha256']=='6539eaf4eb4615493a2bad7dd453d8121fc7345097ed9742f544e37fe2c2f7d6'
    if out.exists():validate();print('Reused',out,sha(out.read_bytes()));return
    with tempfile.TemporaryDirectory(prefix='jrdb_five_source_') as t:
        root=Path(t);code=root/'code';code.mkdir()
        for n,(digest,enc) in SOURCES.items():
            raw=zlib.decompress(base64.b85decode(enc));assert sha(raw)==digest;(code/n).write_bytes(raw)
        sys.path.insert(0,str(code))
        from author_members import get_members
        data=get_members(root/'author_training_records',code)
        env={**os.environ,'PYTHONPATH':str(code),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        subprocess.run([sys.executable,'-u',str(code/'test_five_source.py')],env=env,check=True)
        cmd=[sys.executable,'-u',str(code/'run_five_source_poses.py'),str(data)]
        subprocess.run(cmd+['--check-only'],env=env,check=True)
        candidate=root/'trial.zip'
        subprocess.run(cmd+['--output',str(candidate)],env=env,check=True)
        out.write_bytes(candidate.read_bytes())
    validate();print('Verified',out,out.stat().st_size,sha(out.read_bytes()),flush=True)
if __name__=='__main__':main()
"""
src=src.replace('__SOURCES__',repr(sources));out=D/'kaggle_cell.py';compile(src,str(out),'exec');out.write_text(src)
print('Bytes and SHA-256',out.stat().st_size,hashlib.sha256(out.read_bytes()).hexdigest())
