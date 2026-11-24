"""Outcome-blind raw point/clock coverage from all eleven source-labelled shared-space clips."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,itertools,json,pickletools
import av,numpy as np
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def roi(p):
    ops=list(pickletools.genops(p.read_bytes()));assert set(o.name for o,v,i in ops)<={'PROTO','MARK','BININT1','BININT2','BININT','TUPLE','BINPUT','STOP'}
    q=[int(v) for o,v,i in ops if o.name in ('BININT1','BININT2','BININT')];assert len(q)==4;return q
def run():
    protocol=json.loads((D/'shared_space_freeze.json').read_text());root=D/'private_source';results=[];all_keys=[];manifest={}
    for clip in protocol['clips']:
        p=root/'vci-raw-dut'/(clip+'_roi.mp4');points=root/'data/trajectories'/(clip+'_traj_ped.csv');vehicles=root/'data/trajectories'/(clip+'_traj_veh.csv');ratio=root/'data/ratios'/(clip+'_ratio_pixel2meter.txt');rp=root/'legacy/clips/rois'/(clip+'_roi')
        manifest.update({str(f.relative_to(root)):sha(f) for f in (p,points,vehicles,ratio,rp)})
        with av.open(str(p)) as cont:
            stream=cont.streams.video[0];tb=stream.time_base;fps=stream.average_rate;size=[stream.width,stream.height];decl=stream.frames;video_codec=stream.codec_context.name
            ticks=sorted(pkt.pts for pkt in cont.demux(stream) if pkt.pts is not None)
        times=np.array(ticks,dtype=np.int64)*float(tb);assert len(times)==decl and np.all(np.diff(times)>0)
        with av.open(str(p)) as cont:first=next(cont.decode(video=0));assert first.pts==ticks[0] and [first.width,first.height]==size
        anns=defaultdict(dict);center_oob=0
        for r in csv.DictReader(points.open()):
            assert r['label']=='ped';pid=int(r['id']);frame=int(r['frame']);xy=tuple(float(r[a]) for a in ('x','y'));assert np.isfinite(xy).all() and frame not in anns[pid] and 1<=frame<=decl
            if not (0<=xy[0]<size[0] and 0<=xy[1]<size[1]):center_oob+=1
            anns[pid][frame]=xy
        keys=[];recorded=[]
        for pid,rows in sorted(anns.items()):
            for _,run in itertools.groupby(enumerate(sorted(rows)),lambda z:z[1]-z[0]):
                frames=np.array([f for _,f in run]);pts=times[frames-1];i=0
                while i<len(frames) and pts[-1]>=pts[i]+3.4-(np.diff(times).max()/2)-1e-10:
                    grid=pts[i]+np.arange(35)/10;right=np.clip(np.searchsorted(pts,grid),i,len(pts)-1);left=np.maximum(i,right-1);indices=np.where(abs(pts[left]-grid)<=abs(pts[right]-grid)+1e-12,left,right)
                    ff=frames[indices];tt=pts[indices];assert len(set(ff))==35
                    if np.max(abs(tt-tt[4]-np.arange(-4,31)/10))>0.05+1e-10:i+=1;continue
                    hist_xy=[rows[int(f)] for f in ff[:5]]
                    if any(not (0<=x<size[0] and 0<=y<size[1]) for x,y in hist_xy):
                        i=int(indices[-1])+1;continue
                    keys.append((clip,pid,int(ff[4])));recorded.append((ff.tolist(),tt.tolist()));i=int(indices[-1])+1
        rt=np.array([t for f,t in recorded]);hist=rt[:,4]-rt[:,0] if len(rt) else np.array([]);endpts=rt[:,-1]-rt[:,4] if len(rt) else np.array([])
        assert all(len(set(fs))==35 for fs,t in recorded)
        vc=set();vrows=0
        for r in csv.DictReader(vehicles.open()):vc.add(r['id']);vrows+=1
        original_roi=roi(rp)
        result={'clip':clip,'video_sha256':sha(p),'presentation_clock':'MP4 stream presentation time; no independent exposure timestamps in source','frame_size':size,'roi_xywh':original_roi,'codec':video_codec,'frame_rate':str(fps),'time_base':str(tb),'frame_count':decl,'pts_step_ticks':sorted(set(np.diff(ticks).tolist())),'ped_point_rows':sum(map(len,anns.values())),'pedestrian_ids':len(anns),'vehicle_rows':vrows,'vehicle_ids':len(vc),'tracked_points_outside_released_pixels':center_oob,'pixels_per_metre':float(ratio.read_text()),'windows':len(keys),'eligible_people':len(set(k[1] for k in keys)),'history_span_range_s': [float(hist.min()),float(hist.max())] if len(hist) else None,'endpoint_range_s':[float(endpts.min()),float(endpts.max())] if len(endpts) else None,'history_intervals_within_1ms':int((np.max(abs(np.diff(rt[:,:5])-0.1),axis=1)<=.001+1e-10).sum()) if len(rt) else 0,'endpoint_within_1ms':int(np.sum(abs(endpts-3)<=.001+1e-10)),'grid_35_within_1ms':int((np.max(abs(rt-rt[:,4,None]-np.arange(-4,31)/10),axis=1)<=.001+1e-10).sum()) if len(rt) else 0}
        results.append(result);all_keys.extend([{'key':k,'frames_1based':f,'time_s':t} for k,(f,t) in zip(keys,recorded)]);print(clip,decl,size,str(fps),len(keys),flush=True)
    (D/'private_cohort.json').write_text(json.dumps(all_keys,separators=(',',':'))+'\n');out={'protocol_sha256':sha(D/'shared_space_freeze.json'),'members':manifest,'clips':results,'windows':len(all_keys),'clip_local_people':sum(r['eligible_people'] for r in results),'source_people':sum(r['pedestrian_ids'] for r in results),'ped_rows':sum(r['ped_point_rows'] for r in results),'total_video_frames':sum(r['frame_count'] for r in results),'vehicles':sum(r['vehicle_ids'] for r in results),'cameras':1,'key_manifest_sha256':hashlib.sha256(json.dumps([x['key'] for x in all_keys],separators=(',',':')).encode()).hexdigest()};(D/'coverage.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':run()
