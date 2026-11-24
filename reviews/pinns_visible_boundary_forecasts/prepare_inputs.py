"""Pack already-seen PINNS development inputs; keep future labels separate from historical geometry and frozen predictions."""
import argparse,hashlib,json
from pathlib import Path
import cv2
import numpy as np
import torch
from src.data.features import build_motion_features,build_pose_and_quality_features
from src.data.pinns_extractor import _check_right_handed_homography
from src.models.predictor import UnstructuredShortPosePredictor

WEIGHTS={'america_crossroad':'8ce6d093d1dc34dda0f57da008464f1ca66c9ce7b4638347bfa33111f020026b','italy_park':'d73d488a7b37b98d5b2e786a402ca61c3d14c7c35fd00cba3dfd2e57796a832d','japan_crossroad':'0a2734ddbc1f5a1208552d0784fba6bc6ff0b56a40875d473295e0fd91fc46a1','thailand_street':'aeaf2b9a8ce966c65f22013604a14c051f3116a18ada7c97715782f41ae1452b'}
POSE='b493a47536664ddb4fd66603c1bbf6b2c2e33f553520382027b3ae96b53447d0'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def get_rows(path,requested,flip,ysum):
    rows={}
    for line in path.read_text().splitlines():
        f=line.split()
        if len(f)<11 or f[1]!='Person':continue
        frame,id_=int(f[0]),int(f[2])
        if frame<1 or (frame-1)%3:continue
        k=(id_,(frame-1)//3)
        if k not in requested:continue
        assert k not in rows
        uv=np.array([float(f[3]),float(f[4])]);xy=np.array([float(f[8]),float(f[9])])
        if flip:xy[1]=ysum-xy[1]
        rows[k]=(uv,xy)
    assert rows.keys()==requested
    return rows

@torch.inference_mode()
def prepare(meta,artifacts,maps,dest):
    torch.set_num_threads(2)
    paths=[maps/'pilot_polylines.json',maps/'independent_visual_boundary.json',maps/'boundary_support.json']
    assert [sha(p) for p in paths]==['dc8a6c57727c4c58d3598c1f1551f8fd4e1fbfda47a3edf624599c64b20846bd','67f09740fd247105970779b18665e5cbe30bdeabd10f1734f61d2a1bd5c60aa1','fe04dbf6eba310736a2260d5a3536a5eecd848162e60be6d4f1d25d44769e5fb']
    first,second,report=map(read,paths); cache=artifacts/'pose.npz';assert sha(cache)==POSE
    with np.load(cache,allow_pickle=False) as z:pose={k:z[k] for k in z.files}
    keys=[];histories=[];candidates=[];probs=[];inside=[];mapsout={};labels={}
    for clip,items in sorted(first['clips'].items()):
        site=next(s for s in WEIGHTS if clip.startswith(s+'_'))
        cond=clip.rsplit('_',1)[0];camfile=meta/'videos'/cond/'calib.json';lab=meta/'labels'/(clip+'.txt')
        ca=read(camfile);flip=_check_right_handed_homography(ca);ysum=ca['y_min']+ca['y_max']
        source=report['clips'][clip]
        assert sha(camfile)==source['calibration_sha256'] and sha(lab)==source['labels_sha256']
        hull=cv2.convexHull(np.asarray(ca['pts_2d'],np.float32))
        in_hull=lambda p:cv2.pointPolygonTest(hull,tuple(map(float,p)),False)>=0
        h,_=cv2.findHomography(np.asarray(ca['pts_2d'],float),np.asarray(ca['pts_3d'],float),method=0)
        def transform(line):
            uv=np.array(line,float);hom=np.c_[uv,np.ones(len(uv))]@h.T
            assert np.all(np.abs(hom[:,2])>1e-8)
            xy=hom[:,:2]/hom[:,2:3]
            if not flip:xy[:,1]=ysum-xy[:,1]
            return {'xy':xy.tolist(),'inside':[in_hull(p) for p in uv]}
        line2=second['annotations'][site];assert line2['source_png']==clip+'_0.png' and line2['source_frame']==0
        mapsout[clip]={'first':[transform(v['points']) for v in items],'independent':[transform(v) for v in line2['polylines_xy']]}
        indices=np.flatnonzero(pose['clip']==clip);subset=[(int(pose['person_id'][i]),int(pose['t0'][i])) for i in indices]
        assert len(indices)==source['windows'] and all(t>=4 for _,t in subset)
        need={(p,t-d) for p,t in subset for d in range(5)};observed=get_rows(lab,need,flip,ysum)
        model_file=artifacts/'folds'/site/'M1.pt';assert sha(model_file)==WEIGHTS[site]
        ck=torch.load(model_file,map_location='cpu',weights_only=True);assert ck['identity']['test_site']==site and ck['identity']['model']=='M1'
        model=UnstructuredShortPosePredictor('M1').eval();model.load_state_dict(ck['model_state'],strict=True)
        for i,(person,time) in zip(indices,subset):
            history=np.stack([observed[person,t][1] for t in range(time-4,time+1)])
            rel=history-history[-1]
            np.testing.assert_allclose(rel,pose['hist_pos_relative'][i],atol=1e-6,rtol=0)
            mask=np.ones(5,np.float32);motion=build_motion_features(rel,mask,dt=.1)
            pp,qq,_,_=build_pose_and_quality_features(pose['hist_kpts'][i],pose['hist_bbox'][i],mask,float(pose['img_h'][i]))
            d={'x_m':motion,'x_p':pp,'q_vec':qq,'ego_type_oh':np.eye(8,dtype=np.float32)[pose['ego_type'][i]]}
            out=model({k:torch.from_numpy(a[None]) for k,a in d.items()})
            keys.append([clip,person,time]);histories.append(history);inside.append(in_hull(observed[person,time][0]))
            candidates.append(out['trajs'][0].numpy());probs.append(out['mode_probs'][0].numpy())
        labels[clip]=(lab,flip,ysum,subset)
    assert len(keys)==59
    pack=dest/'observed_and_candidates.npz'
    np.savez_compressed(pack,clip=[k[0] for k in keys],person=[k[1] for k in keys],t0=[k[2] for k in keys],hist=histories,paths=candidates,probs=probs,anchor_inside=inside,maps=np.array(json.dumps(mapsout)))
    # Seal targets separately after the blinded input bank has been serialized. No future statistic or score here.
    future=[]
    for clip,(lab,flip,ysum,subset) in labels.items():
        requested={(p,t+j) for p,t in subset for j in range(1,31)};rows=get_rows(lab,requested,flip,ysum)
        future.extend(np.stack([rows[p,t+j][1] for j in range(1,31)]) for p,t in subset)
    target=dest/'sealed_targets.npz'
    np.savez_compressed(target,keys=np.array(json.dumps(keys)),future=np.stack(future))
    write(dest/'inputs_manifest.json',{'protocol':sha(dest/'PROTOCOL.md'),'observations_and_candidates':sha(pack),'sealed_targets':sha(target),'frozen_pose':POSE,'frozen_M1_weights_by_held_site':WEIGHTS,'first_map':sha(paths[0]),'independent_map':sha(paths[1]),'registered_source':sha(paths[2]),'labels_and_calibration':{clip:{field:v[field] for field in ('calibration_sha256','labels_sha256')} for clip,v in report['clips'].items()},'window_count':len(keys)})
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    for k in ('metadata','artifacts','maps','output'):a.add_argument(k,type=Path)
    p=a.parse_args();prepare(p.metadata,p.artifacts,p.maps,p.output)
