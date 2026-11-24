"""Pack the registered decision rule and synthetic tests into a private Kaggle notebook cell."""
import base64
import hashlib
from pathlib import Path
import zlib

base=Path(__file__).resolve().parent
files=('PROTOCOL.md','selector.py','run_rerank.py','test_selector.py')
parts={f:(hashlib.sha256((base/f).read_bytes()).hexdigest(),base64.b85encode(zlib.compress((base/f).read_bytes(),9)).decode()) for f in files}
template='''# Attach the existing individual joint-actions Kaggle output containing joint_actions_all.zip.
import base64, hashlib, os, subprocess, sys, tempfile, zlib
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix="pinns_action_rerank_"))
repo=root/"source"; repo.mkdir()
rev="6dd78f7a4f77fb1d81603f53c67b8c205053e9f9"
subprocess.run(["git","-C",str(repo),"init","-q"],check=True)
subprocess.run(["git","-C",str(repo),"fetch","-q","--depth=1","https://github.com/pjyyyds/Unstructured-ShortPose-Pred.git",rev],check=True)
subprocess.run(["git","-C",str(repo),"checkout","-q","--detach",rev],check=True)
assert subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()==rev
folder=root/"audit"; folder.mkdir()
payload=PAYLOAD
for name,(checksum,s) in payload.items():
    data=zlib.decompress(base64.b85decode(s))
    assert hashlib.sha256(data).hexdigest()==checksum
    (folder/name).write_bytes(data)
sys.path.insert(0,str(repo))
search=Path("/kaggle/input")
def source(name,expected):
    return next((p for p in sorted(search.rglob(name)) if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==expected),None)
joint=source("joint_actions_all.zip","b84eaaadcaab4518ded15c83e9d746030e03693e89dfe9e23b15ef2f1fc8587e")
if joint is None:raise FileNotFoundError("Attach the original validated joint_actions_all.zip as Kaggle input")
hist=source("pinns_matched_histories.npz","caa8f0959f28f2d69a6f16cc9df115ab6222d0b6eb3297edd575615fef81b94c")
if hist is None:
    from scripts.stage_pinns_metadata import stage_metadata
    info=["--metadata",str(stage_metadata(Path("/tmp/shortpose_action_consistency_metadata")))]
else:info=["--census",str(hist)]
env={**os.environ,"PYTHONPATH":str(repo),"OPENBLAS_NUM_THREADS":"1"}
out=Path("/kaggle/working/action_rerank")
cmd=[sys.executable,str(folder/"run_rerank.py"),str(repo),str(joint),str(out),*info]
subprocess.run([sys.executable,str(folder/"test_selector.py")],cwd=repo,env=env,check=True)
subprocess.run([*cmd,"--check-only"],cwd=repo,env=env,check=True)
subprocess.run(cmd,cwd=repo,env=env,check=True)
print(out/"action_consistency_rerank.zip")
'''
(base/'kaggle_cell.py').write_text(template.replace('PAYLOAD',repr(parts)))
