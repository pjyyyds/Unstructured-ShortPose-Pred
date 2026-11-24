# Kaggle CPU with Internet and the completed history-based mode-ranking notebook output.
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

revision = 'e661d82c8a6f03951f568617e978eb2904c4c364'
repo = Path('/tmp/shortpose_subhalf_metre')
subprocess.run(['git','clone','--quiet','https://github.com/pjyyyds/Unstructured-ShortPose-Pred.git', str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','-q','--detach',revision],check=True)
sys.path.insert(0,str(repo))
from scripts.stage_pinns_metadata import stage_metadata
from scripts.evaluate_pinns_short_displacement import evaluate

sources=list(Path('/kaggle/input').rglob('reranked_modes.zip'))
if len(sources)!=1:
    raise RuntimeError(f'Expected exactly one original reranking archive, found {len(sources)}')
data=stage_metadata(Path('/tmp/pinns_subhalf_labels'))
out=Path('/kaggle/working')
scores=out/'subhalf_scores.json'
evaluate(data,scores,sources[0])
result=json.loads(scores.read_text())
design=json.loads((repo/'reviews/pinns_short_displacement/history_support.json').read_text())
assert result['design']==design['design'] and result['protocol_sha256']==design['protocol_sha256']
target=out/'five_frame_subhalf_results.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    z.write(scores,scores.name)
    z.write(repo/'reviews/pinns_short_displacement/history_support.json','history_support.json')
print('Verified historical assignments, original site metrics and paired five-seed results:',hashlib.sha256(target.read_bytes()).hexdigest(), flush=True)
