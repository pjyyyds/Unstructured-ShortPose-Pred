from register_roi import D,read,frame,sha
import collections,csv,cv2,numpy as np,json
def main():
    cv2.setNumThreads(1);c=read(D/'coverage.json');S=D/'private_source';G=D/'private_reference';summary=[]
    for r in c['clips']:
        s=r['clip'];w,h=r['frame_size'];n=r['frame_count'];data=collections.defaultdict(dict)
        with (S/('data/trajectories/'+s+'_traj_ped.csv')).open() as f:
            for row in csv.DictReader(f):data[int(row['frame'])][int(row['id'])]=np.array([float(row['x']),float(row['y'])],np.float32)
        legacy=G/('legacy/clips/trajectories_filtered/'+s+'_traj_ped_filtered.csv');nrows=0;mx=0.
        with legacy.open() as f:
            for row in csv.DictReader(f):
                t=int(float(row['frame']));p=int(float(row['id']));src=[float(row[k]) for k in ('x','y')];mx=max(mx,float(abs(np.array(src)-data[t][p]).max()));nrows+=1
        assert nrows==sum(map(len,data.values()))
        cp=cv2.VideoCapture(str(S/('vci-raw-dut/'+s+'_roi.mp4')));tally=collections.defaultdict(list)
        for f in np.linspace(1,n-1,20,dtype=int):
            a=cv2.cvtColor(frame(cp,int(f)),cv2.COLOR_BGR2GRAY);b=cv2.cvtColor(frame(cp,int(f+1)),cv2.COLOR_BGR2GRAY);both=sorted(set(data[f])&set(data[f+1]))
            if not both:continue
            pts=np.array([data[f][p] for p in both]);diff=np.array([data[f+1][p]-data[f][p] for p in both])
            for lbl,inp in [('raw_roi',pts),('x_plus_40',pts+np.array([40.,0.],np.float32)),('horizontal_mirror',np.column_stack((w-1-pts[:,0],pts[:,1])).astype(np.float32))]:
                xy,ok,rerr=cv2.calcOpticalFlowPyrLK(a,b,inp.reshape(-1,1,2),None,winSize=(15,15),maxLevel=3)
                good=(ok.ravel()!=0)&(inp[:,0]>=0)&(inp[:,0]<w)&(inp[:,1]>=0)&(inp[:,1]<h);z=xy.reshape(-1,2)-inp
                for v,gt in zip(z[good],diff[good]):tally[lbl].append((float(np.linalg.norm(v-gt)),float(np.linalg.norm(gt)),float(v@gt/(np.linalg.norm(v)*np.linalg.norm(gt)+1e-12))))
        cp.release();stats={}
        for k,val in tally.items():
            v=np.array(val);move=v[:,1]>=1;stats[k]={'valid':len(v),'median_displacement_error_px':float(np.median(v[:,0])),'p90_error_px':float(np.quantile(v[:,0],.9)),'source_displacement_at_least_one_px_n':int(move.sum()),'median_cosine_moving':float(np.median(v[move,2])) if move.any() else None}
        summary.append({'clip':s,'legacy_unfiltered_xy_rows_compared':nrows,'max_legacy_to_current_raw_xy_abs_px':mx,'image_flow':stats});print(s,[(k,round(v['median_displacement_error_px'],3),round(v['median_cosine_moving'] or 0.,3)) for k,v in stats.items()],flush=True)
    (D/'flow_registration.json').write_text(json.dumps({'source':'https://github.com/dongfang-steven-yang/vci-dataset-dut/tree/80b8c746833664cd1e5244fccad79c7f2a7cbe31','plan_sha256':sha((D/'flow_registration_plan.txt').read_bytes()),'per_clip':summary,'scope':'Source-motion concordance between positions and delivered videos, no fitted offsets or geometric corrections. Both the released CSRT points and Lucas-Kanade estimate motion from images and can share scene-level biases. Only clip 04 has an independent author-rendered demonstration.'},indent=2)+'\n')
if __name__=='__main__':main()
