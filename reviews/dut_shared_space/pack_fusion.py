from pathlib import Path
import hashlib,zlib,base64
from pack_motion import template
d=Path(__file__).resolve().parent
names=['motion_cloud.py','detector_pilot.py','fusion_cloud.py','shared_space_freeze.json','motion_and_detector_freeze.json','coverage.json']
enc={k:(hashlib.sha256((d/k).read_bytes()).hexdigest(),base64.b85encode(zlib.compress((d/k).read_bytes(),9)).decode()) for k in names}
src=template.replace("dut_shared_space_motion.zip","dut_shared_space_fusion.zip").replace("SOURCES['motion_cloud.py'][0]","SOURCES['fusion_cloud.py'][0]").replace("a['axis_and_timing']['0']['all']['M0']['windows']","a['by_axis_and_availability']['0']['all']['M0']['windows']").replace("cmd=[sys.executable,'-u',str(code/'motion_cloud.py'),str(weights),str(out)]","""from_work=out.parent/'dut_shared_space_pose_cache.zip';copies=([from_work] if from_work.is_file() else [])+sorted(root.rglob('dut_shared_space_pose_cache.zip'))
  if not copies:raise FileNotFoundError('Attach the full DUT pose cache or run the full detector/cache cell in this Kaggle session')
  if len({sha(x.read_bytes()) for x in copies})>1:raise ValueError('Ambiguous distinct full pose caches')
  cmd=[sys.executable,'-u',str(code/'fusion_cloud.py'),str(weights),str(copies[0]),str(out)]""").replace("subprocess.run(cmd+['--check-only'],cwd=code,env=env,check=True);","").replace('EMBED',repr(enc))
p=d/'kaggle_dut_shared_space_fusion.py';p.write_text(src);compile(src,str(p),'exec');print(p.name,len(p.read_bytes()),hashlib.sha256(p.read_bytes()).hexdigest())
