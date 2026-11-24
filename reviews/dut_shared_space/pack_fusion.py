from pathlib import Path
import hashlib,zlib,base64
from pack_motion import template
d=Path(__file__).resolve().parent
names=['motion_cloud.py','detector_pilot.py','fusion_cloud.py','shared_space_freeze.json','motion_and_detector_freeze.json','coverage.json']
enc={k:(hashlib.sha256((d/k).read_bytes()).hexdigest(),base64.b85encode(zlib.compress((d/k).read_bytes(),9)).decode()) for k in names}
src=template.replace("dut_shared_space_motion.zip","dut_shared_space_fusion.zip").replace("SOURCES['motion_cloud.py'][0]","SOURCES['fusion_cloud.py'][0]").replace("a['axis_and_timing']['0']['all']['M0']['windows']","a['by_axis_and_availability']['0']['all']['M0']['windows']").replace("cmd=[sys.executable,'-u',str(code/'motion_cloud.py'),str(weights),str(out)]","""cache=find_cache()
  cmd=[sys.executable,'-u',str(code/'fusion_cloud.py'),str(weights),str(cache),str(out)]""").replace("subprocess.run(cmd+['--check-only'],cwd=code,env=env,check=True);","").replace('EMBED',repr(enc))
source_anchor=" def verify():"
insert=""" def find_cache():
  here=out.parent/'dut_shared_space_pose_cache.zip'
  copies=([here] if here.is_file() else [])+sorted(root.rglob('dut_shared_space_pose_cache.zip'))
  if not copies:raise FileNotFoundError('Attach the complete pose cache or run its Kaggle cell in this session')
  if len({sha(p.read_bytes()) for p in copies})!=1:raise ValueError('Distinct full pose caches found; mount exactly the intended one')
  return copies[0]
 def verify():"""
src=src.replace(source_anchor,insert).replace("a=json.loads(z.read('aggregate.json'));assert","a=json.loads(z.read('aggregate.json'));assert a['pose_cache_sha256']==sha(find_cache().read_bytes()),'Existing fusion result and current full pose cache differ';assert")
p=d/'kaggle_dut_shared_space_fusion.py';p.write_text(src);compile(src,str(p),'exec');print(p.name,len(p.read_bytes()),hashlib.sha256(p.read_bytes()).hexdigest())
