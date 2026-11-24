"""Materialize the predefined author-example cohort, only required past RGB and calibration; no forecasts."""
from collections import Counter,defaultdict
import hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
from stage_example import ROOT,catalog,stage_ranges
BASE='V2X-Seq-SPD-Example/infrastructure-side/'
def h(v):return hashlib.sha256(v).hexdigest()
def read(path):return json.loads(path.read_text())
def prepare():
 source=ROOT/'private_source'/BASE
 d=catalog()
 def stage_rel(names):
  missing=[BASE+s for s in sorted(set(names)) if not (source/s).is_file()]
  if missing:stage_ranges(missing)
 stage_rel(['data_info.json']);info=read(source/'data_info.json')
 info.sort(key=lambda f:(f['sequence_id'],int(f['frame_id'])))
 stage_rel([f['label_camera_std_path'] for f in info])
 rows=defaultdict(dict);timing={};types=Counter();used=[]
 for row in info:
  seq=row['sequence_id'];f=int(row['frame_id']);timing[seq,f]=row
  for obj in read(source/row['label_camera_std_path']):
   types[obj['type']]+=1
   if obj['type']!='Pedestrian':continue
   pid=int(obj['track_id']);c=obj['3d_location'];b=obj['2d_box'];pt=[c[k] for k in 'xyz'];box=[b[k] for k in ('xmin','ymin','xmax','ymax')]
   if not np.isfinite(pt+box).all() or box[2]<=box[0] or box[3]<=box[1]:continue
   assert f not in rows[seq,pid]
   rows[seq,pid][f]=(pt,box)
 for (seq,pid),rec in sorted(rows.items()):
  for _,part in itertools.groupby(enumerate(sorted(rec)),key=lambda a:a[1]-a[0]):
   ticks=[f for _,f in part]
   for start in range(0,len(ticks)-34,35):
    frames=ticks[start:start+35];used.append((seq,pid,frames))
 if not used:raise ValueError('No eligible 35-frame person tracklet in the example')
 names=[];hist=set()
 for s,p,frames in used:
  for t in frames:names.append(timing[s,t]['calib_virtuallidar_to_world_path'])
  for t in frames[:5]:
   row=timing[s,t]; hist.add(row['image_path'])
   names.extend([row['image_path'],row['calib_camera_intrinsic_path'],row['calib_virtuallidar_to_camera_path']])
 stage_rel(names)
 import cv2
 records=[];private=[];pose_pixels=[];timestamps=[];ground=[]
 for s,p,frames in used:
  fr=np.array(frames);rows_i=[timing[s,t] for t in frames];ts=np.array([int(r['image_timestamp']) for r in rows_i]);lid=np.array([int(r['pointcloud_timestamp']) for r in rows_i]);assert (np.diff(ts)>0).all()
  points=[];boxes=[];images=[];Hs=[];rig=[];bounds=[];projection=[]
  for j,r in enumerate(rows_i):
   pt,box=rows[s,p][frames[j]];transform=read(source/r['calib_virtuallidar_to_world_path'])
   rot=np.array(transform['rotation']);trans=np.asarray(transform['translation']).reshape(3)
   np.testing.assert_allclose(rot.T@rot,np.eye(3),atol=.012);assert .99<np.linalg.det(rot)<1.01
   rig.append(np.column_stack((rot,trans)));points.append(rot@pt+trans)
   if j<5:
    b=np.asarray(box);x1,y1,x2,y2=b;path=source/r['image_path'];img=cv2.imread(str(path))
    assert img is not None and img.shape[0]>y1 and img.shape[1]>x1 and x2>0 and y2>0
    K=np.array(read(source/r['calib_camera_intrinsic_path'])['cam_K']).reshape(3,3)
    ext=read(source/r['calib_virtuallidar_to_camera_path']);rot2=np.array(ext['rotation']);trans2=np.array(ext['translation']).reshape(3)
    cam=rot2@pt+trans2;pixel=K@cam;assert cam[2]>0 and np.isfinite(pixel).all()
    projection.append((pixel[:2]/pixel[2]).tolist());bounds.append([img.shape[1],img.shape[0]])
    boxes.append([(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1]);images.append(r['image_path'])
  rig=np.array(rig)
  # Express every observed/target position in the same decision-frame virtual LiDAR axes using each released world transform.
  xyz=(np.array(points)-rig[4,:,3])@np.linalg.inv(rig[4,:,:3]).T
  original=np.array([rows[s,p][t][0] for t in frames])
  np.testing.assert_allclose(xyz[4],original[4],atol=1e-7)
  reconstruction=xyz@rig[4,:,:3].T+rig[4,:,3]
  np.testing.assert_allclose(reconstruction,points,atol=1e-8,rtol=0)
  dt=(ts-ts[4])*1e-6
  strict=np.max(np.abs(dt-np.arange(-4,31)*.1))<=.001000001
  entry={'sequence':s,'actor':p,'first':int(fr[0]),'t0':int(fr[4]),'last':int(fr[-1]),'rig_translation_change_m_max':float(np.linalg.norm(rig[:,:,3]-rig[4,:,3],axis=1).max()),'rig_rotation_element_change_max':float(abs(rig[:,:,:3]-rig[4,:,:3]).max()),'history_span_s':float(-dt[0]),'endpoint_s':float(dt[-1]),
         'history_max_gap_error_ms':float(np.max(np.abs(np.diff(ts[:5])/1000-100))),
         'max_abs_grid_error_ms':float(np.max(np.abs(dt-np.arange(-4,31)*.1))*1000),'strict_1ms_grid':bool(strict),
         'nearest_grid_index_verified':bool(np.max(np.abs(dt-np.arange(-4,31)*.1))<=.05),'camera_minus_lidar_ms_range':((ts-lid)[[np.argmin(ts-lid),np.argmax(ts-lid)]]/1000).tolist()}
  records.append(entry);private.append({'key':[s,p,int(fr[4])],'frame_ids':frames,'timestamps_us':ts.tolist(),'pointcloud_timestamps_us':lid.tolist(),
      'ground_xy_m':xyz[:,:2].tolist(),'boxes_cxcywh':boxes,'image_paths':images,'images_wh':bounds,'projected_3D_centres_px':projection})
 alltimes=np.array([int(r['image_timestamp']) for r in info]);support={'example_scene_ids':sorted({i['sequence_id'] for i in info}),'intersections':sorted({i['intersection_loc'] for i in info}),
 'camera_ids':sorted({i['camera_id'] for i in info}),'camera_frames':len(info),'all_source_classes':dict(types),'all_pedestrian_tracks':len(rows),
 'complete_disjoint_five_plus_thirty_windows':len(records),'distinct_admitted_people':len(set((s,p) for s,p,_ in used)),'unique_observed_images':len(hist),
 'camera_gap_ms_min_median_max':np.quantile(np.diff(alltimes)/1000,[0,.5,1]).tolist(),'window_timing':records}
 manifest={n:h((source/n).read_bytes()) for n in sorted(set(['data_info.json']+names+[r['label_camera_std_path'] for r in info]))}
 (ROOT/'coverage.json').write_text(json.dumps(support,indent=2)+'\n')
 (ROOT/'private_cohort.json').write_text(json.dumps({'records':private,'source_members_sha256':manifest},separators=(',',':'))+'\n')
 print(json.dumps(support,indent=1))
if __name__=='__main__':prepare()
