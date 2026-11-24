from pathlib import Path
import collections,csv,hashlib,json,math
import cv2,numpy as np
D=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def sha(b):return hashlib.sha256(b).hexdigest()
def frame(cap,f):
    cap.set(cv2.CAP_PROP_POS_FRAMES,f-1);ok,b=cap.read();assert ok;return b
def q(x):return list(map(float,np.quantile(x,(0,.5,.9,1))))
def main():
    c=read(D/'coverage.json');tr={r['path']:r for r in read(D/'author_tree.json')['tree']};G=D/'private_reference';S=D/'private_source';out=[];orb=cv2.ORB_create(1500);bf=cv2.BFMatcher(cv2.NORM_HAMMING);cv2.setNumThreads(1)
    def orb_bin(img):
        return orb.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),None)
    for r in c['clips']:
        s=r['clip'];w,h=r['frame_size'];n=r['frame_count'];bg=cv2.imread(str(G/('data/backgrounds/'+s+'_background_GSOC.png')))
        kp0,des0=orb_bin(bg);cp=cv2.VideoCapture(str(S/('vci-raw-dut/'+s+'_roi.mp4')));samples=[]
        for f in (1,(n+1)//2,n):
            img=frame(cp,f);kp,des=orb_bin(img);pairs=bf.knnMatch(des0,des,k=2);matches=[m for m,q in pairs if m.distance<.75*q.distance]
            uv=np.array([[kp[p.trainIdx].pt[i]-kp0[p.queryIdx].pt[i] for i in (0,1)] for p in matches]);dist=np.linalg.norm(uv,axis=1)
            samples.append({'frame_1based':f,'raw_ratio_test_matches':len(matches),'median_delta_xy_px':np.median(uv,axis=0).tolist(),'median_distance_px':float(np.median(dist)),'p90_distance_px':float(np.quantile(dist,.9)),'fraction_within_2px_of_zero_delta':float(np.mean(dist<=2))})
        cp.release();out.append({'clip':s,'background_wh':list(bg.shape[1::-1]),'decoded_roi_wh':[w,h],'source_full_frame_roi_xywh':r['roi_xywh'],'samples':samples})
    s='roundabout_04';r=next(x for x in c['clips'] if x['clip']==s);w,h=r['frame_size'];ox,oy,ow,oh=r['roi_xywh'];n=r['frame_count']
    clean=S/('vci-raw-dut/'+s+'_roi.mp4');cb=clean.read_bytes();assert hashlib.sha1(b'blob '+str(len(cb)).encode()+b'\0'+cb).hexdigest()==tr['legacy/clips/roi_clips/roundabout_04_roi.mp4']['sha']
    full=G/('legacy/clips/stabilized/stabilized_'+s+'.mp4');marked=G/('legacy/clips/roi_clips/'+s+'_roi_box.mp4')
    caps=[cv2.VideoCapture(str(p)) for p in (clean,full,marked)];assert all(int(cp.get(7))==n for cp in caps)
    histories={}
    for name,p in [('raw',S/('data/trajectories/'+s+'_traj_ped.csv')),('filtered',G/('legacy/clips/trajectories_filtered/'+s+'_traj_ped_filtered.csv'))]:
        data=collections.defaultdict(dict)
        with p.open() as f:
            for rr in csv.DictReader(f):
                pid=int(float(rr['id']));t=int(float(rr['frame']));xy=[float(rr[k]) for k in ('x','y')] if name=='raw' else [float(rr[k]) for k in ('x_est','y_est')]
                data[t][pid]=xy
        histories[name]=data
    tmp=np.zeros((17,17),np.uint8);cv2.circle(tmp,(8,8),5,255,2);ys,xs=np.nonzero(tmp);xo=xs-8;yo=ys-8
    samples=[];ring=collections.defaultdict(list);center_diff=[]
    for f in sorted(set(range(1,n+1,10))|{n}):
        plain=frame(caps[0],f);labels=frame(caps[2],f)
        src=frame(caps[1],f);src=src[oy:oy+h,ox:ox+w].astype(float)
        base=plain.astype(float);mae=float(abs(base-src).mean());near={}
        for dt in (-1,1):
            if 1<=f+dt<=n:near[str(dt)]=float(abs(base-frame(caps[1],f+dt)[oy:oy+h,ox:ox+w]).mean())
        # blue-vs-(green/red) score, report fixed threshold >20 levels above clean frame and red/green margins.
        def blue(img):
            v=img.astype(float);return v[:,:,0]-np.maximum(v[:,:,1],v[:,:,2])
        marked_blue=blue(labels);delta=marked_blue-blue(plain);flags=(marked_blue>20)&(delta>20)
        for typ,data in histories.items():
            for dt in (-1,0,1):
                if 1<=f+dt<=n:
                    for pid,(x,y) in data[f+dt].items():
                        xx=int(x)+xo;yy=int(y)+yo;good=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
                        if good.any():ring[typ,dt].append(float(flags[yy[good],xx[good]].mean()))
        for pid,pt in histories['raw'][f].items():
            if pid in histories['filtered'][f]:center_diff.append(float(np.linalg.norm(np.array(pt)-histories['filtered'][f][pid])))
        samples.append({'frame':f,'fixed_crop_mae_same':mae,'fixed_crop_mae_adjacent':near})
    for cap in caps:cap.release()
    good=[s for s in samples if len(s['fixed_crop_mae_adjacent'])==2]
    reg={'provenance':'Official author background image + 11 ROI MP4s; author full stabilized and rendered labels exist for roundabout_04. ROI movie is byte-identical to the Git-tracked author copy.','plan_sha256':sha((D/'registration_plan.txt').read_bytes()),'eleven_clips':out,'clip04':{'public_clean_movie_git_blob':tr['legacy/clips/roi_clips/roundabout_04_roi.mp4']['sha'],'author_roi_xywh':r['roi_xywh'],'encoded_roi_wh':[w,h],'same_frame_count':n,'full_source_crop_frames_checked':len(samples),'matching_frame_mae_range_median90_max':q([r['fixed_crop_mae_same'] for r in samples]),'previous_mae_quants':q([s['fixed_crop_mae_adjacent']['-1'] for s in good]),'next_mae_quants':q([s['fixed_crop_mae_adjacent']['1'] for s in good]),'best_same_vs_two_neighbors':sum(s['fixed_crop_mae_same']<min(s['fixed_crop_mae_adjacent'].values()) for s in good),'two_comparator_frames':len(good),'ring_fraction_summary':{f'{k}[{dt:+}]':{'n':len(vals),'mean_fraction':float(np.mean(vals)),'median_fraction':float(np.median(vals))} for (k,dt),vals in ring.items()},'raw_to_marked_filtered_position_difference_px_min_median_p90_max':q(center_diff),'samples':samples},'interpretation':'Identity-coordinate spatial correspondence is empirically testable for all clips; the author full-video temporal/crop and label-overlay control is present only for the published clip04 example. Consecutive frames move slowly and overlap in demos; a colored marker validates tracker registration, not joint locations, ground truth bounding boxes, or physical feet. No fitted geometric correction or information from future backgrounds was used for evaluation.'}
    (D/'registration.json').write_text(json.dumps(reg,indent=2)+'\n');allrows=[x for a in out for x in a['samples']];print('frames',len(allrows),'matches minmax',min(s['raw_ratio_test_matches'] for s in allrows),max(s['raw_ratio_test_matches'] for s in allrows),'median distance minmax',min(s['median_distance_px'] for s in allrows),max(s['median_distance_px'] for s in allrows));print({k:v for k,v in reg['clip04'].items() if k!='samples'})
if __name__=='__main__':main()
