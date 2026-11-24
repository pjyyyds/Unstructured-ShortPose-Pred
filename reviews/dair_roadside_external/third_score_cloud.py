"""Forward-only, frozen four-checkpoint PINNS evaluation on the author's measured DAIR roadside example.
Use --check-only for source and model identity; actual pose/forecast inference is Kaggle only.
"""
import argparse,hashlib,io,itertools,json,os,sys,tempfile,zipfile
from pathlib import Path
import numpy as np

from full_range import BASE as ROOT
STATES={'folds/america_crossroad/M0.pt': 'b58e909e67ce3b94f34cf6c5933d1c75d39184207107cba17da79291bff6a743', 'folds/america_crossroad/M1.pt': '8ce6d093d1dc34dda0f57da008464f1ca66c9ce7b4638347bfa33111f020026b', 'folds/italy_park/M0.pt': '39bdc069331a19d1480e00ce519ec7d8076ade70ddda75225ff914e75a200e84', 'folds/italy_park/M1.pt': 'd73d488a7b37b98d5b2e786a402ca61c3d14c7c35fd00cba3dfd2e57796a832d', 'folds/japan_crossroad/M0.pt': '328c15620a68c7070fe82d0b742f7dfcb01fb22c9011ee022ae832ed249db4f4', 'folds/japan_crossroad/M1.pt': '0a2734ddbc1f5a1208552d0784fba6bc6ff0b56a40875d473295e0fd91fc46a1', 'folds/thailand_street/M0.pt': '028204da29c49d1c4efd4fa3d071ce40218e77fdc4a4d228c28dd9956721e542', 'folds/thailand_street/M1.pt': 'aeaf2b9a8ce966c65f22013604a14c051f3116a18ada7c97715782f41ae1452b'}
MODELS=('M0','M1')
ANGLES=(0,90,180,270)
METHODS=(*MODELS,'cv_measured','cv_nominal')
FOLDS=tuple(sorted({k.split('/')[1] for k in STATES}))
def sha(b):return hashlib.sha256(b).hexdigest()
def jh(o):return sha(json.dumps(o,sort_keys=True,separators=(',',':')).encode())
def rawsha(p):return sha(p.read_bytes())
def verify_weights(path):
 ret={}
 for rel,digest in STATES.items():
  _,site,model_pt=rel.split('/')
  if path.is_file():
   with zipfile.ZipFile(path) as z:
    names=[k for k in z.namelist() if k in (rel,'seed_42/'+site+'/'+model_pt)]
    if len(names)!=1:raise ValueError('Wrong model archive '+rel)
    blob=z.read(names[0])
  else:
   names=[p for p in path.rglob(model_pt) if p.parent.name==site and rawsha(p)==digest]
   if not names:raise ValueError('Missing registered '+rel)
   blob=names[0].read_bytes()
  if sha(blob)!=digest:raise ValueError('Unexpected frozen weight '+rel)
  ret[site,model_pt[:-3]]=blob
 return ret

def matrix(degrees):
 a={0:(1,0),90:(0,1),180:(-1,0),270:(0,-1)}[degrees];c,s=a
 return np.array([[c,-s],[s,c]],float)

def inputs(history,poses,boxes,image_h,angle):
 import torch
 from src.data.features import build_motion_features,build_pose_and_quality_features
 assert history.shape==(5,2) and poses.shape==(5,17,3) and boxes.shape==(5,4)
 rel=(history-history[-1])@matrix(angle).T
 motion=build_motion_features(rel,np.ones(5),dt=.1)
 pose,q,_,_=build_pose_and_quality_features(poses,boxes,np.ones(5),float(image_h))
 ego=np.zeros(8,np.float32);ego[2 if motion[-1,4]<.2 else 0]=1
 return {k:torch.from_numpy(v).unsqueeze(0) for k,v in {'x_m':motion,'x_p':pose,'q_vec':q,'ego_type_oh':ego}.items()}

def reduce_scores(v,valid,people):
 ids=np.flatnonzero(valid)
 if not len(ids):return {'windows':0,'people':0,'window_equal':None,'person_equal':None}
 unique=sorted(set(people[ids]))
 return {'windows':len(ids),'people':len(unique),'window_equal':dict(zip(('ade_m','fde_m'),v[ids].mean(0).tolist())),
         'person_equal':dict(zip(('ade_m','fde_m'),np.mean([v[ids[people[ids]==p]].mean(0) for p in unique],axis=0).tolist()))}

def main(weights,onnx,out,check):
 from scripts.run_heldout_cloud import MODEL_SHA
 data=json.loads((ROOT/'private_third_cohort.json').read_text());coverage=json.loads((ROOT/'third_coverage.json').read_text())
 enrolled_path=ROOT/'third_enrollment.json'
 assert rawsha(enrolled_path)=='e576d72a3883724f6719dc2cdb3d1cd3554c37a8bd00e0c071bbf11f47e5aa37'
 enrolled=json.loads(enrolled_path.read_text())
 rec=data['records'];actual_keys=[r['key'] for r in rec];assert len(rec)==54 and len({tuple(k) for k in actual_keys})==54
 assert sha(json.dumps(actual_keys,separators=(',',':')).encode())==enrolled['keys_sha256']
 for name,expected_digest in data['members'].items():
  dirname='images' if name.endswith('.jpg') else 'metadata'
  assert rawsha(ROOT/'private_expansion'/dirname/name)==expected_digest
 assert jh(data['members'])==coverage['source_members_manifest_sha256']
 assert rawsha(onnx)==MODEL_SHA
 states=verify_weights(weights)
 if check:print('Accepted all 54 preregistered windows and 8 model identities',flush=True);return
 if not Path('/kaggle/working').is_dir():raise RuntimeError('Run the registered evaluation on Kaggle CPU')
 import cv2,torch,onnxruntime
 from src.models.predictor import UnstructuredShortPosePredictor
 from src.data.pinns_extractor import RTMPoseONNXExtractor
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 models={}
 for (site,n),blob in states.items():
  ck=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
  assert ck['identity']['test_site']==site and ck['identity']['model']==n and ck['identity']['protocol']['seed']==42
  net=UnstructuredShortPosePredictor(n).cpu().eval();net.load_state_dict(ck['model_state'],strict=True);models[site,n]=net
 pose=RTMPoseONNXExtractor(onnx_path=onnx,batch_size=1)
 assert pose.sess is not None and pose.sess.get_providers()==['CPUExecutionProvider']
 points=[];probs=[];joint_cache=[];qualities=[];nominal=[];measured=[]
 for r in rec:
  # Only historical positions, clocks and measured past person crops enter model inference.
  hxy=np.array(r['ground_xy_m'][:5]);boxes=np.array(r['boxes_cxcywh'],np.float32);times=np.array(r['timestamps_us'],np.int64)
  joint=[]
  for image,box,wh in zip(r['image_paths'],boxes,r['images_wh']):
   img=cv2.imread(str(ROOT/'private_expansion'/'images'/image),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
   assert img is not None and (img.shape[1],img.shape[0])==tuple(wh)
   joint.append(pose.extract_crops(img,box[None])[0])
  joint=np.stack(joint);assert joint.shape==(5,17,3) and np.isfinite(joint).all()
  joint_cache.append(joint);case=[];pq=[]
  elapsed=(times[5:]-times[4])/(times[4]-times[3])
  cv1=elapsed[:,None]*(hxy[-1]-hxy[-2])[None,:];cv_nom=np.arange(1,31)[:,None]*(hxy[-1]-hxy[-2])[None,:]
  measured.append(cv1);nominal.append(cv_nom)
  for a in ANGLES:
   m=matrix(a);x=inputs(hxy,joint,boxes,r['images_wh'][0][1],a)
   if not a:qualities.append(x['q_vec'][0].numpy())
   np.testing.assert_allclose((elapsed[:,None]*((hxy[-1]-hxy[-2])@m.T)[None,:])@m,cv1,rtol=0,atol=1e-12)
   yy=[];qq=[]
   with torch.inference_mode():
    for s in FOLDS:
     pairs=[];pairq=[]
     for n in MODELS:
      f=models[s,n](x)
      pred=f['trajs'][0].numpy().astype(float)@m
      q=f['mode_probs'][0].numpy();assert np.isfinite(pred).all() and np.isfinite(q).all() and abs(q.sum()-1)<1e-5
      pairs.append(pred);pairq.append(q)
     yy.append(pairs);qq.append(pairq)
   case.append(yy);pq.append(qq)
  points.append(case);probs.append(pq)
 # Reveal the future ground coordinates to the scoring code only after every model's predictions are frozen in memory.
 paths=np.asarray(points);pr=np.asarray(probs);truth=np.array([np.asarray(r['ground_xy_m'])[5:]-r['ground_xy_m'][4] for r in rec])
 selected=np.take_along_axis(paths,pr.argmax(4)[:,:,:,:,None,None,None],axis=4).squeeze(4)
 delta=np.linalg.norm(selected-truth[:,None,None,None,:,:],axis=-1);errors=np.stack([delta.mean(-1),delta[:,:,:,:,-1]],axis=-1)
 def cv_error(pred):
  d=np.linalg.norm(np.asarray(pred)-truth,axis=-1);return np.stack([d.mean(-1),d[:,-1]],axis=-1)
 cvm,cvn=cv_error(measured),cv_error(nominal)
 people=np.array([str(r['key'][0])+':'+str(r['key'][1]) for r in rec]);times=np.array([r['timestamps_us'] for r in rec]);relative=times-times[:,4:5]
 grid=np.arange(-4,31)*100000
 masks={'all_nominal_nearest':np.max(abs(relative-grid),axis=1)<=50000,'history_spacing_within_1ms':np.max(abs(np.diff(times[:,:5])-100000),axis=1)<=1000,
        'all_35_grid_within_1ms':np.max(abs(relative-grid),axis=1)<=1000,'endpoint_within_1ms':abs(relative[:,-1]-3000000)<=1000}
 assert masks['all_nominal_nearest'].all()
 summaries={}
 for j,a in enumerate(ANGLES):
  means={n:errors[:,j,:,i,:].mean(1) for i,n in enumerate(MODELS)};means.update(cv_measured=cvm,cv_nominal=cvn)
  means['M1_minus_M0']=means['M1']-means['M0']
  for n in MODELS:means[n+'_minus_cv_measured']=means[n]-cvm
  summaries[str(a)]={k:{n:reduce_scores(v,mask,people) for n,v in means.items()} for k,mask in masks.items()}
  if not a:
   for grouping,labels,available in [('by_sequence',[r['key'][0] for r in rec],enrolled['seqs']),
                                    ('by_intersection',[r['site'] for r in rec],sorted({v['intersection'] for v in enrolled['seqs'].values()}))]:
    summaries[grouping]={s:{n:reduce_scores(v,np.asarray(labels)==s,people) for n,v in means.items()} for s in available}
 report={'study':'All remaining 36 official training sequences in the preregistered third external cohort. Never used to fit the PINNS checkpoints. 54 windows from 29 sequence-local people; descriptive.',
  'source':'https://github.com/AIR-THU/DAIR-V2X-Seq/tree/main/dataset/v2x-seq-spd','source_member_manifest_sha256':coverage['source_members_manifest_sha256'],'weights_sha256':STATES,
  'model_sha256':MODEL_SHA,'script_sha256':rawsha(Path(__file__)),'protocol_sha256':rawsha(ROOT/'third_cohort_freeze.json'),'expansion_freeze_sha256':rawsha(ROOT/'expansion_freeze.txt'),
  'runtime':{'torch':str(torch.__version__),'onnxruntime':onnxruntime.__version__,'providers':pose.sess.get_providers(),'opencv':cv2.__version__},
  'coverage':enrolled,'input':'Official camera-time offline tracked 2D crops and metric 3D centres. A single t0 virtual-LiDAR coordinate frame via each per-frame calibration. No gesture ground truth; 17-image keypoints estimated by RTMPose. No adaptation.',
  'estimand':'Four fixed PINNS checkpoint errors equally weighted first, then pooled across 54 windows and 29 sequence-local identities. Intersection and sequence summaries include zero-support groups. No inference to additional people/sites. Primary zero-degree M1 minus M0 ADE. All rotations prespecified, no selection. Network uses fixed 0.1s discretization; measured clock separately reported for every case.',
  'rotations_deg':list(ANGLES),'summary_by_rotation':summaries,'quality':np.asarray(qualities).tolist()}
 out.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(dir=out.parent) as temp:
  w=Path(temp);np.savez_compressed(w/'private.npz',paths_relative_native_axes=paths,probs=pr,top1_errors_m=errors,gt_relative=truth,
       measured_cv=np.array(measured),nominal_cv=np.array(nominal),rtmpose_kpts=np.array(joint_cache),quality=np.asarray(qualities),timestamps_us=times,actors=people,
       pinnings_fold_order=np.array(FOLDS),model_order=np.array(MODELS),keys_json=np.array(json.dumps(actual_keys)))
  members={'aggregate.json':(json.dumps(report,indent=2,allow_nan=False)+'\n').encode(),'private.npz':(w/'private.npz').read_bytes(),
    'source_members.json':json.dumps(data['members'],sort_keys=True).encode(),'cloud_runner.py':Path(__file__).read_bytes()}
  members['sha256.json']=json.dumps({k:sha(v) for k,v in members.items()},sort_keys=True).encode()
  zpath=w/'result.zip'
  with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
   for k,v in members.items():z.writestr(k,v)
  with zipfile.ZipFile(zpath) as z:assert z.testzip() is None
  zpath.replace(out)
 print('Verified external predictions archived',out,rawsha(out),flush=True)

if __name__=='__main__':
 c=argparse.ArgumentParser()
 for k in ('weights','onnx','output'):c.add_argument(k,type=Path)
 c.add_argument('--check-only',action='store_true');a=c.parse_args();main(a.weights,a.onnx,a.output,a.check_only)
