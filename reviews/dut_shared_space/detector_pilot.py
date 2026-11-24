"""Frozen DUT real-person detection and bidirectional source-point association, without evaluating trajectory errors."""
import argparse,concurrent.futures, csv, ctypes as ct,ctypes.util, hashlib, io,json,os,tempfile,urllib.request,zipfile,zlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from motion_cloud import D,build,sha,read
def axis_tiles(length):
    if length<=384:return [0]
    return sorted(set(range(0,length-384+1,288))|{length-384})
def overlap(a,b):
    xy0=np.maximum(a[:2],b[:2]);xy1=np.minimum(a[2:],b[2:]);size=np.maximum(xy1-xy0,0)
    v=size.prod();union=(a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-v
    return v/union if union>0 else 0.
def dedup(boxes):
    candidates=sorted(boxes,key=lambda x:(-x[4],*x[:4]));keep=[]
    for row in candidates:
        if all(overlap(row[:4],k[:4])<=.5 for k in keep):keep.append(row)
    return np.array(keep,dtype=np.float32).reshape(-1,5)
def associate(boxes,points):
    matches={};membership=defaultdict(list)
    for i,box in enumerate(boxes):
        members=[p for p,(x,y) in points.items() if box[0]<=x<=box[2] and box[1]<=y<=box[3]]
        for p in members:membership[p].append((i,len(members)))
    for p in points:
        hits=membership[p]
        if not hits:matches[p]=('missing',-1)
        elif len(hits)>1 and any(count>1 for _,count in hits):matches[p]=('multiple_boxes_and_people',-1)
        elif len(hits)>1:matches[p]=('multiple_boxes',-1)
        elif hits[0][1]>1:matches[p]=('multiple_people_in_box',-1)
        else:matches[p]=('unique',hits[0][0])
    return matches
def source_videos(root):
    catalog=read(D/'author_video_catalog.json');c=read(D/'coverage.json');names=['vci-raw-dut/'+s['clip']+'_roi.mp4' for s in c['clips']]
    if all((root/k).is_file() and sha((root/k).read_bytes())==c['members'][k] for k in names):return
    url='https://drive.usercontent.google.com/download?id=1J0PW4NoL2mi_eaN_8_SgkWkc0rk7MJZC&confirm=t';size=104243676
    def span(bounds):
        a,b=bounds
        for j in range(4):
            try:
                req=urllib.request.Request(url,headers={'Range':f'bytes={a}-{b}','Accept-Encoding':'identity'})
                with urllib.request.urlopen(req,timeout=50) as r:
                    assert r.status==206 and r.headers['Content-Range']==f'bytes {a}-{b}/{size}';v=r.read(b-a+2);assert len(v)==b-a+1;return v
            except (OSError,AssertionError):
                if j==3:raise
    first=catalog[names[0]]['offset'];last=catalog[names[-1]];end=last['data_offset']+last['compressed']
    chunks=[(0,min(v['offset'] for v in catalog.values())-1)]+[(i,min(end,i+3_000_000)-1) for i in range(first,end,3_000_000)]+[(end,size-1)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:rar=b''.join(pool.map(span,chunks))
    # Use standard system libarchive RAR5 decoder, never execute code from the source archive.
    lib=ct.CDLL(ctypes.util.find_library('archive') or 'libarchive.so.13')
    def func(name,args,res):
        f=getattr(lib,name);f.argtypes=args;f.restype=res;return f
    vp=ct.c_void_p;char=ct.c_char_p;nsize=ct.c_size_t
    new=func('archive_read_new',[],vp);formats=func('archive_read_support_format_all',[vp],ct.c_int);filters=func('archive_read_support_filter_all',[vp],ct.c_int)
    op=func('archive_read_open_memory',[vp,vp,nsize],ct.c_int);next_entry=func('archive_read_next_header',[vp,ct.POINTER(vp)],ct.c_int);pathname=func('archive_entry_pathname',[vp],char)
    readbuf=func('archive_read_data',[vp,vp,nsize],ct.c_ssize_t);free=func('archive_read_free',[vp],ct.c_int)
    handle=new();raw=ct.create_string_buffer(rar);expected=set(names);got=set();entry=vp()
    try:
        assert formats(handle)>=0 and filters(handle)>=-20 and op(handle,raw,len(rar))==0
        while True:
            status=next_entry(handle,ct.byref(entry))
            if status==1:break
            assert status==0,status
            name=pathname(entry).decode().replace('\\','/')
            if name not in expected:continue
            assert name not in got;dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();temp=dest.with_suffix('.part')
            with temp.open('wb') as f:
                buf=ct.create_string_buffer(262144)
                while True:
                    n=readbuf(handle,buf,len(buf));assert n>=0
                    if n==0:break
                    b=buf.raw[:n];f.write(b);h.update(b)
            assert h.hexdigest()==c['members'][name];temp.replace(dest);got.add(name)
        assert got==expected
    finally:free(handle)
def collect(root,onnx,out,full):
    if not Path('/kaggle/working').is_dir():raise RuntimeError('External detector and pose verification run only on Kaggle')
    import torch, torchvision, cv2,onnxruntime
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
    from src.data.pinns_extractor import RTMPoseONNXExtractor
    from scripts.run_heldout_cloud import MODEL_SHA
    assert sha(onnx.read_bytes())==MODEL_SHA;coverage=read(D/'coverage.json');keys,fr,ticks,xy=build(root)
    source_videos(root);sites=[r['clip'] for r in coverage['clips']]
    chosen=list(range(len(keys))) if full else [next(i for i,k in enumerate(keys) if k[0]==s) for s in sites]
    need=defaultdict(lambda:defaultdict(set))
    for i in chosen:
        s,p,t=keys[i]
        for f in fr[i,:5]:need[s][int(f)].add(p)
    weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1;assert weights.url=='https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'
    net=fasterrcnn_resnet50_fpn_v2(weights=weights).eval()
    dst=Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/',1)[-1];weightsha=sha(dst.read_bytes());assert weightsha.startswith('dd69338a')
    requested=os.environ.get('DUT_DETECTOR_DEVICE','auto').strip().lower()
    if requested not in {'auto','cpu','cuda'}:raise ValueError('DUT_DETECTOR_DEVICE must be auto, cpu or cuda')
    if requested=='cuda' and not torch.cuda.is_available():raise RuntimeError('DUT_DETECTOR_DEVICE=cuda but this Kaggle worker has no CUDA device')
    dev=torch.device('cuda' if requested!='cpu' and torch.cuda.is_available() else 'cpu');net.to(dev);torch.set_num_threads(2)
    runtime={'requested_detector_device':requested,'actual_detector_parameter_device':str(next(net.parameters()).device),'cuda_visible':torch.cuda.is_available(),'torch_cuda_version':torch.version.cuda,'gpu_model':torch.cuda.get_device_name(dev) if dev.type=='cuda' else None,'gpu_capability':list(torch.cuda.get_device_capability(dev)) if dev.type=='cuda' else None}
    pose=RTMPoseONNXExtractor(onnx_path=onnx,batch_size=16);assert pose.sess is not None
    # Explicitly label provider choice; identical graph/model SHA and source pixels; no fitting.
    stored={};cnt=Counter();all_box_count=0;checked_frames=0;tags=[]
    for s in sites:
        point=defaultdict(dict)
        with (root/('data/trajectories/'+s+'_traj_ped.csv')).open() as f:
            for row in csv.DictReader(f):
                t=int(row['frame'])
                if t in need[s]:point[t][int(row['id'])]=(float(row['x']),float(row['y']))
        cap=cv2.VideoCapture(str(root/('vci-raw-dut/'+s+'_roi.mp4')));assert cap.isOpened()
        end=max(need[s]);scene=next(r for r in coverage['clips'] if r['clip']==s);w,h=scene['frame_size'];assert [int(cap.get(3)),int(cap.get(4))]==[w,h] and int(cap.get(7))==scene['frame_count']
        for t in range(1,end+1):
            assert cap.grab()
            if t not in need[s]:continue
            ok,img=cap.retrieve();assert ok and img.shape[:2]==(h,w)
            crops=[(x,y) for y in axis_tiles(h) for x in axis_tiles(w)];candidates=[]
            for start in range(0,len(crops),2):
                subset=crops[start:start+2];data=[torch.from_numpy(cv2.cvtColor(img[y:y+384,x:x+384],cv2.COLOR_BGR2RGB)).permute(2,0,1).to(dev,torch.float32)/255. for x,y in subset]
                with torch.inference_mode():detections=net(data)
                for (x,y),o in zip(subset,detections):
                    good=(o['labels']==1)&(o['scores']>=.5);bb=o['boxes'][good].detach().cpu().numpy()+np.array([x,y,x,y]);sc=o['scores'][good].detach().cpu().numpy()
                    candidates.extend(np.column_stack((bb,sc)))
            boxes=dedup(candidates);all_box_count+=len(boxes);checked_frames+=1
            matches=associate(boxes,point[t]);selected=sorted(need[s][t]);unique=[p for p in selected if matches[p][0]=='unique']
            for p in selected:cnt[matches[p][0]]+=1
            if unique:
                xyxy=np.array([boxes[matches[p][1],:4] for p in unique]);bb=np.column_stack(((xyxy[:,:2]+xyxy[:,2:])/2,xyxy[:,2:]-xyxy[:,:2])).astype('f');kpts=pose.extract_crops(img,bb)
                for p,b,k in zip(unique,bb,kpts):stored[s,p,t]=(b,k)
            tags.extend([[s,p,t,matches[p][0]] for p in selected])
        cap.release();print(s,'complete, unique selected person-frames',cnt['unique'],flush=True)
    final_keys=[keys[i] for i in chosen];fids=fr[chosen,:5];b=np.zeros((len(chosen),5,4),np.float32);j=np.zeros((len(chosen),5,17,3),np.float32);matched=np.zeros((len(chosen),5),bool)
    for i,((s,p,t),seq) in enumerate(zip(final_keys,fids)):
        for l,f in enumerate(seq):
            if (s,p,int(f)) in stored:
                b[i,l],j[i,l]=stored[s,p,int(f)];matched[i,l]=True
    assert np.isfinite(j).all();hist=np.bincount(matched.sum(1),minlength=6).tolist()
    valid=(j[:,:,:,2]>=.2) & matched[:,:,None]
    report={'mode':'all_windows' if full else 'prespecified_first_window_each_clip','windows':len(chosen),'source_historical_images':checked_frames,'requested_historical_person_observations':len(chosen)*5,'unique_person_frame_labels':dict(cnt),'detected_person_boxes_all_processed_frames':all_box_count,'matched_observations':int(matched.sum()),'matched_frames_per_window_0_to_5':hist,'frames_with_one_joint_at_or_above_0_2':int(valid.any(-1).sum()),'frames_with_9_joints_at_or_above_0_2':int((valid.sum(-1)>=9).sum()),'all_five_frames_at_least_9_joints':int((valid.sum(axis=-1)>=9).all(axis=1).sum()),'full_cohort_keys_sha256':coverage['key_manifest_sha256'],'score_protocol_sha256':sha((D/'motion_and_detector_freeze.json').read_bytes()),'detector_sha256':weightsha,'detector_url':weights.url,'device':str(dev),'backend_identity':{**runtime,'rtmpose_selected_providers':pose.active_providers,'rtmpose_session_providers':pose.sess.get_providers()},'torchvision':torchvision.__version__,'torch':str(torch.__version__),'onnxruntime':onnxruntime.__version__,'ort_providers':pose.sess.get_providers(),'onnx_sha256':MODEL_SHA,'script_sha256':sha(Path(__file__).read_bytes()),'scope':'No forecasting, no accepted person-frame has multiple matching released points or person detection boxes. A uniquely matched source point verifies geometric association only; 2D joints are estimates from top-down imagery.'}
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        tmp=Path(temp);np.savez_compressed(tmp/'private.npz',keys_json=np.array(json.dumps(final_keys)),source_frame_1based=fids,matched=matched,bbox=b,kpts=j,association_json=np.array(json.dumps(tags,separators=(',',':'))))
        m={'aggregate.json':json.dumps(report,indent=2).encode(),'private.npz':(tmp/'private.npz').read_bytes(),'source.py':Path(__file__).read_bytes()};m['sha256.json']=json.dumps({k:sha(v) for k,v in m.items()}).encode()
        zpath=tmp/'bundle.zip'
        with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
            for k,v in m.items():z.writestr(k,v)
        with zipfile.ZipFile(zpath) as z:assert z.testzip() is None
        zpath.replace(out)
    print(json.dumps(report,indent=1))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('onnx',type=Path);p.add_argument('out',type=Path);p.add_argument('--full',action='store_true');p.add_argument('--source-root',type=Path,default=D/'private_source')
    a=p.parse_args();collect(a.source_root,a.onnx,a.out,a.full)
