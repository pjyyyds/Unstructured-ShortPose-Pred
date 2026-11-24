"""Enrollment only. No external model, errors, predictions, or pose estimation."""
import collections, hashlib, itertools, json
from pathlib import Path
import numpy as np
from full_range import BASE, directory, ranges
D='V2X-Seq-SPD/infrastructure-side/'
I='V2X-Seq-SPD-infrastructure-side-image/'
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads(p.read_text())
def run(download_past_images=False):
    split_path=BASE/'full_split.json';assert sha(split_path.read_bytes())=='0a59332221f3618639f34fb70678d03c5221ab80fd69f2cf41acffca0d8859fd'
    freeze=BASE/'expansion_freeze.txt';assert sha(freeze.read_bytes())=='7ed8a594190dd416edae7811e7b83e82ab12f29c323459d855656721470f2e9d'
    split=load(split_path);third=BASE/'third_cohort_freeze.json'
    assert sha(third.read_bytes())=='f4e7a5d5e8627a63dc10bf9979f413dbbf6a3766d7cbc91865046bfce9d8aacb'
    seqs=load(third)['remaining_train_ids']
    assert len(seqs)==36 and set(seqs)==set(split['batch_split']['train'])-set(load(third)['prior_ten'])
    m=directory('metadata');images=directory('images')
    paths=ranges('metadata',m,[D+'data_info.json']);info=load(paths[D+'data_info.json'])
    selected=sorted((r for r in info if r['sequence_id'] in seqs),key=lambda r:(r['sequence_id'],int(r['frame_id'])))
    train_frames=set(split['infrastructure_split']['train']);assert all(r['frame_id'] in train_frames for r in selected)
    lab=ranges('metadata',m,[D+r['label_camera_std_path'] for r in selected]); print('Selected labels',len(lab),flush=True)
    records=collections.defaultdict(dict);fr={};ped_count=collections.Counter();site={}
    for r in selected:
        s=r['sequence_id'];f=int(r['frame_id']);fr[s,f]=r;site[s]=r['intersection_loc']
        for o in load(lab[D+r['label_camera_std_path']]):
            if o['type']!='Pedestrian':continue
            pid=int(o['track_id']);c=o['3d_location'];box=o['2d_box'];pt=[float(c[k]) for k in 'xyz'];b=[float(box[k]) for k in ('xmin','ymin','xmax','ymax')]
            ped_count[s]+=1
            if not np.isfinite(pt+b).all() or b[2]<=b[0] or b[3]<=b[1]:continue
            if f in records[s,pid]:raise ValueError(f'duplicate {s}/{pid}/{f}')
            records[s,pid][f]=(pt,b)
    cases=[];rejected=[];seen={};member_names=[];image_names=[]
    for (s,pid),fs in sorted(records.items()):
        for _,g in itertools.groupby(enumerate(sorted(fs)),lambda y:y[1]-y[0]):
            run=[f for _,f in g]
            for i in range(0,len(run)-34,35):
                ticks=run[i:i+35];r=[fr[s,t] for t in ticks];clock=np.array([int(x['image_timestamp']) for x in r]);d=clock-clock[4];grid=np.arange(-4,31)*100000
                physical=tuple((x['intersection_loc'],x['camera_ip'],pid,int(x['image_timestamp'])) for x in r)
                if any(x in seen for x in physical):rejected.append([s,pid,ticks[4],'physical actor/time overlap']);continue
                if np.any(np.diff(clock)<=0) or np.max(abs(d-grid))>50000:rejected.append([s,pid,ticks[4],'measured timestamps outside nearest nominal grid']);continue
                seen.update(dict.fromkeys(physical,s))
                for x in r:member_names.append(D+x['calib_virtuallidar_to_world_path'])
                for x in r[:5]:
                    member_names.extend(D+x[k] for k in ('calib_camera_intrinsic_path','calib_virtuallidar_to_camera_path'));image_names.append(I+Path(x['image_path']).name)
                cases.append((s,pid,ticks,clock))
    print('Enrolled candidate windows',len(cases),'rejected',rejected,'unique past images',len(set(image_names)),flush=True)
    keys=[[s,p,f[4]] for s,p,f,ts in cases]
    labs={k:sha(p.read_bytes()) for k,p in {**paths,**lab}.items()}
    enrolled={'source_commit':load(third)['official_source_commit'],'protocol_sha256':sha(third.read_bytes()),
              'source_metadata_and_labels_sha256':sha(json.dumps(labs,sort_keys=True,separators=(',',':')).encode()),
              'source_frames':len(selected),'source_unique_camera_clocks':len({(r['intersection_loc'],r['camera_ip'],r['image_timestamp']) for r in selected}),
              'pedestrian_rows':sum(ped_count.values()),'source_people':len(records),'windows':len(cases),'distinct_sequence_people':len({(s,p) for s,p,f,t in cases}),
              'unique_past_images':len(set(image_names)),'past_jpeg_compressed_bytes':sum(images[k][1] for k in set(image_names)),
              'required_calibrations_compressed_bytes':sum(m[k][1] for k in set(member_names)), 'rejected':rejected,'seqs':{},
              'keys_sha256':sha(json.dumps(keys,separators=(',',':')).encode())}
    for s in seqs:
        my=[(p,f,ts) for ss,p,f,ts in cases if s==ss];enrolled['seqs'][s]={'intersection':site[s],'frames':sum(r['sequence_id']==s for r in selected),'pedestrian_rows':ped_count[s],'pedestrian_ids':sum(ss==s for ss,p in records),'enrolled_ids':len({p for p,f,ts in my}),'windows':len(my)}
    ticks=np.array([ts for s,p,f,ts in cases],dtype=np.int64).reshape(-1,35)
    if len(ticks):
        dt=ticks-ticks[:,4:5];enrolled['timing']={'endpoint_s_min_median_max':np.quantile(dt[:,-1]/1e6,(0,.5,1)).tolist(),'history_s_min_median_max':np.quantile(-dt[:,0]/1e6,(0,.5,1)).tolist(),'all_35_grid_1ms':int((abs(dt-np.arange(-4,31)*100000).max(1)<=1000).sum()),'history_intervals_1ms':int((abs(np.diff(ticks[:,:5],axis=1)-100000).max(1)<=1000).sum()),'endpoint_1ms':int((abs(dt[:,-1]-3000000)<=1000).sum())}
    (BASE/'third_enrollment.json').write_text(json.dumps(enrolled,indent=2)+'\n')
    (BASE/'third_enrollment_keys.json').write_text(json.dumps(keys,separators=(',',':'))+'\n')
    if not download_past_images:return
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Run third-cohort images and pose on Kaggle')
    mr=ranges('metadata',m,member_names);im=ranges('images',images,image_names)
    import cv2
    out=[];timings=[]; 
    for s,pid,ticks,clock in cases:
        rows=[fr[s,t] for t in ticks];world=[];rig=[];boxes=[];imagepaths=[];dims=[];checks=[]
        for j,(t,r) in enumerate(zip(ticks,rows)):
            pos,box=records[s,pid][t]
            cal=load(mr[D+r['calib_virtuallidar_to_world_path']]);R=np.array(cal['rotation']);T=np.array(cal['translation']).reshape(3)
            assert .99<np.linalg.det(R)<1.01;np.testing.assert_allclose(R.T@R,np.eye(3),atol=.012)
            world.append(R@pos+T);rig.append((R,T))
            if j>=5:continue
            name=I+Path(r['image_path']).name;path=im[name];img=cv2.imread(str(path),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            assert img is not None;h,w=img.shape[:2];l,u,rr,b=box;assert l<w and u<h and rr>0 and b>0
            ci=load(mr[D+r['calib_camera_intrinsic_path']]);ce=load(mr[D+r['calib_virtuallidar_to_camera_path']]);cam=np.array(ce['rotation'])@pos+np.array(ce['translation']).reshape(3);pix=np.asarray(ci['cam_K']).reshape(3,3)@cam
            assert cam[2]>0 and np.isfinite(pix).all();checks.append((pix[:2]/pix[2]).tolist())
            boxes.append([(l+rr)/2,(u+b)/2,rr-l,b-u]);imagepaths.append(name);dims.append([w,h])
        R,T=rig[4];ground=(np.array(world)-T)@np.linalg.inv(R).T
        np.testing.assert_allclose(ground[4],records[s,pid][ticks[4]][0],atol=1e-7);np.testing.assert_allclose(ground@R.T+T,world,rtol=0,atol=1e-8)
        dt=clock-clock[4];cal={'history_span_s':float(-dt[0]/1e6),'endpoint_s':float(dt[-1]/1e6),'history_1ms':bool(np.max(abs(np.diff(clock[:5])-100000))<=1000),'endpoint_1ms':bool(abs(dt[-1]-3000000)<=1000),'grid_1ms':bool(np.max(abs(dt-np.arange(-4,31)*100000))<=1000)}
        timings.append(cal);out.append({'key':[s,pid,ticks[4]],'site':site[s],'camera':rows[4]['camera_id'],'frame_ids':ticks,'timestamps_us':clock.tolist(),'ground_xy_m':ground[:,:2].tolist(),'boxes_cxcywh':boxes,'image_paths':imagepaths,'images_wh':dims,'projected_3D_centres_px':checks})
    defs={'zip_metadata_bytes':234358463,'zip_images_bytes':3560631820,'metadata_central_sha256':sha((BASE/'full_metadata_central.bin').read_bytes()),'image_central_sha256':sha((BASE/'full_images_central.bin').read_bytes()),'training_split_sha256':sha(split_path.read_bytes()),'protocol_sha256':sha(freeze.read_bytes()),'git_source_revision':'c885c54af0c34bc515fa9ca8b5e8fda76a15462c','public_split_url':'https://github.com/AIR-THU/DAIR-V2X/blob/c885c54af0c34bc515fa9ca8b5e8fda76a15462c/data/split_datas/cooperative-split-data-spd.json','public_full_folder':'https://drive.google.com/drive/folders/1AomWNJjy5KMmjrb0BHHUfcelhEN5zuWn'}
    sup={'sources':defs,'official_training_sequences':seqs,'source_frames':len(selected),'unique_source_sensor_clocks':len({(r['intersection_loc'],r['camera_ip'],r['image_timestamp']) for r in selected}),'unique_past_images':len(set(image_names)),'num_windows':len(out),'unique_sequence_local_people':len({(s,p) for s,p,_,_ in cases}),'rejected_timing_or_overlap':rejected,'per_sequence':{},'measurement':{}}
    for s in seqs:
        os=[o for o in out if o['key'][0]==s];sup['per_sequence'][s]={'site':site[s],'frames':sum(x['sequence_id']==s for x in selected),'pedestrian_rows':ped_count[s],'pedestrian_ids':sum(ss==s for ss,p in records),'windows':len(os),'enrolled_people':len({o['key'][1] for o in os})}
    for key in ('history_span_s','endpoint_s'):
        vals=[t[key] for t in timings];sup['measurement'][key+'_min_median_max']=np.quantile(vals,(0,.5,1)).tolist() if vals else None
    for key in ('history_1ms','endpoint_1ms','grid_1ms'):sup['measurement'][key]=sum(t[key] for t in timings)
    members={k:sha(p.read_bytes()) for k,p in {**paths,**lab,**mr,**im}.items()}
    sup['third_protocol_sha256']=sha(third.read_bytes());sup['source_members_manifest_sha256']=sha(json.dumps(members,sort_keys=True,separators=(',',':')).encode());sup['source_named_members']=len(members)
    (BASE/'third_coverage.json').write_text(json.dumps(sup,indent=2)+'\n')
    (BASE/'private_third_cohort.json').write_text(json.dumps({'records':out,'timing':timings,'members':members},separators=(',',':'))+'\n')
    print(json.dumps(sup,indent=1),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stage-past',action='store_true');run(p.parse_args().stage_past)
