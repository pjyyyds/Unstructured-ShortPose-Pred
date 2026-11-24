"""Apply the frozen t0-only source-category radius audit without reading any forecasts."""
import collections, hashlib,json,math
from pathlib import Path
from statistics import median
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_bytes())
def summary(items):
    return {'windows':len(items),'people':len({(k[0],k[1]) for k,obs in items}),'sequences':len({k[0] for k,obs in items})}
def run():
    frozen=ROOT/'cohort3_neighbor_freeze.json';assert hashlib.sha256(frozen.read_bytes()).hexdigest()=='534ae1bca28ec0010bb100352cee003ab4db11019da1175085c8ba7b39bbde65'
    protocol=load(frozen);keys=load(ROOT/'third_enrollment_keys.json');assert len(keys)==54 and hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest()==protocol['cohort_keys_sha256']
    d=ROOT/'private_expansion/metadata/V2X-Seq-SPD/infrastructure-side';frames={(r['sequence_id'],int(r['frame_id'])):r for r in load(d/'data_info.json')}
    groups=protocol['groups'];rows=[];source=collections.Counter();sidemap={};bad_groups={};conflicts=[]
    for key in keys:
        s,pid,t0=key;frame=frames[s,t0];sidemap[s]=frame['intersection_loc'];obs=load(d/frame['label_camera_std_path'])
        ids=[(r['type'],str(r['track_id'])) for r in obs if r['type'] in {c for cs in groups.values() for c in cs}];duplicates=[pair for pair,n in collections.Counter(ids).items() if n>1];bad_groups[tuple(key)]={g for g,types in groups.items() if any(pair[0] in types for pair in duplicates)}
        if duplicates:conflicts.append({'source_sequence':s,'duplicate_categories':[c for c,id_ in duplicates]})
        target=[r for r in obs if r['type']=='Pedestrian' and int(r['track_id'])==pid];assert len(target)==1
        C=target[0]['3d_location'];x,y=float(C['x']),float(C['y']);sel={g:[] for g in groups}
        for obj in obs:
            category=obj['type'];source[category]+=1
            if category=='Pedestrian' and int(obj['track_id'])==pid:continue
            c=obj['3d_location'];distance=math.hypot(float(c['x'])-x,float(c['y'])-y);assert math.isfinite(distance)
            for g,types in groups.items():
                if category in types:sel[g].append(distance)
        rows.append((key,sel))
    def report(sub):
        out={'support':summary(sub),'within_10m':{},'overlapping_10m':{}}
        for g in groups:
            valid=[(k,obs) for k,obs in sub if g not in bad_groups[tuple(k)]];with_g=[(k,obs) for k,obs in valid if any(v<=10. for v in obs[g])];distances=[min(obs[g]) for k,obs in with_g]
            out['within_10m'][g]={**summary(with_g),'evaluable_windows':len(valid),'duplicate_id_unevaluable_windows':len(sub)-len(valid),'neighbor_instances':sum(sum(v<=10. for v in obs[g]) for k,obs in with_g),
                'nearest_m_min_median_max_present_within_10m':[min(distances),median(distances),max(distances)] if distances else None}
        out['jointly_evaluable_windows']=sum(not bad_groups[tuple(k)] for k,obs in sub)
        for k,obs in sub:
            if bad_groups[tuple(k)]:continue
            hit=','.join(g for g in groups if any(v<=10. for v in obs[g])) or 'none'
            out['overlapping_10m'][hit]=out['overlapping_10m'].get(hit,0)+1
        return out
    result={'study':'Roadside source labels at last observed camera time for 54 predefined pedestrian windows. Category support only. Labelled geometric proximity does not establish avoidance/yielding.','protocol_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),'keys_sha256':protocol['cohort_keys_sha256'],'release':'https://github.com/AIR-THU/DAIR-V2X-Seq/tree/main/dataset/v2x-seq-spd','groups':groups,'duplicate_id_audit':{'affected_windows':len(bad_groups)-sum(not g for g in bad_groups.values()),'occurrences':conflicts,'policy':'Report the source duplicate as an error for that window and class-group. Joint overlaps use only windows with unambiguous IDs in every group; unaffected class coverage uses all valid windows. No duplicated instance is silently selected or counted.'},'unit':'window at t0; a physical image or actor can contribute to multiple candidate windows; each count is a window-neighbour incidence','raw_label_counts_across_54_target_frames':dict(source),'overall':report(rows),'by_site':{s:report([(k,o) for k,o in rows if sidemap[k[0]]==s]) for s in sorted(set(sidemap.values())|{'yizhuang08'})}}
    (ROOT/'neighbor_coverage.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(result['overall'])
if __name__=='__main__':run()
