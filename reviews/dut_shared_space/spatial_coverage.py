"""Decision-time coverage of per-clip map patches; uses no forecasts or future coordinates."""
import collections,csv,hashlib,json,math
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    c=read(D/'coverage.json');cohort=read(D/'private_cohort.json');keys=[v['key'] for v in cohort]
    assert hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest()==c['key_manifest_sha256']
    # Only the already-enrolled (clip,ID,t0) keys are consumed; all selected position sources below have source-frame <=t0.
    by=collections.defaultdict(list)
    for clip,actor,t in keys:by[clip].append((actor,t))
    summary={};allcounts=collections.Counter();total_cells=0
    for meta in c['clips']:
        s=meta['clip'];scale=meta['pixels_per_metre'];w,h=meta['frame_size'];max_t=max(t for a,t in by[s])
        cell_pts=collections.defaultdict(lambda:collections.defaultdict(list));positions={}
        p=D/'private_source'/'data/trajectories'/(s+'_traj_ped.csv')
        with p.open() as f:
            for row in csv.DictReader(f):
                frame=int(row['frame']);actor=int(row['id'])
                if frame>max_t:continue
                x,y=float(row['x']),float(row['y']);cell=(math.floor(x/(5*scale)),math.floor(y/(5*scale)))
                cell_pts[cell][actor].append(frame);positions[actor,frame]=(x,y,cell)
        for people in cell_pts.values():
            for actor,times in people.items():people[actor]=sorted(times)
        bins=collections.defaultdict(list);quadrants=collections.Counter();prior=recent=early=0
        for actor,t in by[s]:
            x,y,cell=positions[actor,t];bins[cell].append((actor,t));quadrants[min(3,int(x*4/w)),min(3,int(y*4/h))]+=1
            # Count only earlier frame numbers of other actors. 23-frame look-back corresponds to <=1.0 s at 23.976 Hz.
            earlier=any(who!=actor and ts[0]<t for who,ts in cell_pts[cell].items());prior+=earlier
            last_second=False
            for who,ts in cell_pts[cell].items():
                if who==actor:continue
                k=np.searchsorted(ts,t,side='left')
                if k and t-ts[k-1]<=23:last_second=True;break
            recent+=last_second;early+=t-1<24
        tile_counts=sorted((len(v) for v in bins.values()),reverse=True);patch_count=math.ceil(w/(5*scale))*math.ceil(h/(5*scale))
        vals={'windows':len(by[s]),'people':len({a for a,t in by[s]}),'roi_width_height_m':[w/scale,h/scale],'possible_5m_grid_cells':patch_count,
              'occupied_t0_cells':len(bins),'patches_with_one_window':sum(len(v)==1 for v in bins.values()),'patches_with_at_least_10_windows':sum(len(v)>=10 for v in bins.values()),
              'patches_with_at_least_3_different_people':sum(len({a for a,t in v})>=3 for v in bins.values()),'windows_in_top_1_patch':tile_counts[0],'windows_in_top_3_patches':sum(tile_counts[:3]),
              'supported_normalized_4x4_bins':len(quadrants),'all_past_other_id_same_cell':int(prior),'previous_1s_other_id_same_cell':int(recent),'decision_before_full_1s_prior_video':int(early)}
        summary[s]=vals;total_cells+=patch_count
        for key,v in vals.items():
            if isinstance(v,int):allcounts[key]+=v
    out={'source':'https://github.com/dongfang-steven-yang/vci-dataset-dut/tree/80b8c746833664cd1e5244fccad79c7f2a7cbe31','cohort_keys_sha256':c['key_manifest_sha256'],'analysis':'Fixed 5 m square patches anchored at the upper left of each released stabilized ROI. A cell at a cropped boundary is counted even if partially visible. Occupancy uses only each enrolled pedestrian’s current unfiltered point. To audit causal context, count same-cell author-tracked points of other pedestrians at strictly earlier frame times, and within the preceding 23 source frames (<=1 s); target ID always excluded. No future coordinate, forecast, score or fitted model is used. The original 3066 IDs were selected using future track completeness, and this remains a selection limitation.',
         'interpretation':'One shared-space site with eleven separately stabilized camera references, per-clip pixel/metric ratios and ROI origins. No common georeferencing transform between those clips established; sum of nonempty cells counts clip-local cells, not unique geographical patches. Point occupancy is an empirical use trace, not a walkway, curb, lane marking or permanent traversability polygon. Full-video supplied GSOC background could incorporate future frames; a prospective map must use image frames available at or before the current prediction and omit the target’s own future.',
         'per_clip':summary,'sums_of_clip_local_counts':dict(allcounts),'largest_clip_window_fraction':max(v['windows'] for v in summary.values())/len(keys)}
    (D/'spatial_coverage.json').write_text(json.dumps(out,indent=2)+'\n');print(out['sums_of_clip_local_counts']);print(out['largest_clip_window_fraction'])
if __name__=='__main__':main()
