"""Count frame/ID/odometry intersections in three already-used official-author training releases. No scoring."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from fractions import Fraction

SCENES=('hewlett-packard-intersection-2019-01-24_0','huang-lane-2019-02-12_0','tressider-2019-03-16_1')
OFFSETS=(-6,-5,-3,-2,0)  # Five distinct 15-Hz frames over 0.4 s, earlier frame at both .5 ties.
SHA={
 'tracks':('29adfca6b4bfcf886d2832217d65b2eb7530c9cd4b9a15edfa94b476f1b1e818','682752cfae6a08771d3547f233023b0d8d444346303baed2d103ce57b027affd','d0da8c09a5593a8491ca825449d77547a64d5f3ecd35f185906b5ef5a64b76bc'),
 'poses':('5db2360f7e0507dab0d7eaf734ce8f2d942253028b555d90c19d282d4b245e45','51eb289d6f2ca68d83a5da7d5845e5eb8df7ff89d698b682f8261ba458b682cb','b728921be8fb6b6a4de6ac410558c219ff344a13a2f40205ae446757c2a124b9'),
 'odom':('25ba8073598b743de7e47f04ab09ed4b8ccc254eecc996d411fb31465e92b80d','9f028bcf79854e581148ae8cfeef62ec5df19930839856586d4d6a754c014c8d','b66ba960bb8be864aa1dc9e04c9b05c8e15e785cf5f400600ccc9151cb0e4c40')}

def summarize(rows,hi):
    selected=collections.defaultdict(list)
    for person,t in sorted(rows):
        previous=selected[person]
        if not previous or t-6>previous[-1]+hi:previous.append(t)
    return {'all_anchor_person_rows':len(rows),'people':len(selected),
            'per_track_earliest_greedy_nonoverlapping':sum(map(len,selected.values()))}

def check(root):
    scenes={};hashes={}
    for si,scene in enumerate(SCENES):
        paths={'tracks':root/'jrdb/data/train'/(scene+'_shift_0.ndjson'),
               'poses':root/'labels_3d_keypoints/train'/(scene+'.json'),
               'odom':root/'odometry/train'/(scene+'.json')}
        raw={k:p.read_bytes() for k,p in paths.items()}
        for k,b in raw.items():
            digest=hashlib.sha256(b).hexdigest();assert digest==SHA[k][si]
            hashes[str(paths[k].relative_to(root))]=digest
        gt=set();encoded=set();fps=set()
        for line in raw['tracks'].splitlines():
            row=json.loads(line)
            if 'scene' in row:fps.add(row['scene']['fps']);continue
            x=row['track'];pt=(int(x['p']),int(x['f']));assert pt not in encoded;encoded.add(pt)
            if math.isfinite(x['x']) and math.isfinite(x['y']) and (x['x']!=0 or x['y']!=0):gt.add(pt)
        assert fps=={2.5} and {t%6 for p,t in gt}=={0}
        of=json.loads(raw['odom'])['odometry']
        odom={int(Path(name).stem) for name,r in of.items() if all(math.isfinite(v) for k in ('position','orientation') for v in r[k].values())}
        assert len(odom)==len(of)
        joint=set();raw_joint=set();fields=set()
        for name,rows in json.loads(raw['poses'])['labels'].items():
            t=int(Path(name).stem)
            for r in rows:
                fields.update(r);p=int(r['label_id'].removeprefix('pedestrian:'));xyz=r['keypoints'];assert len(xyz)==99
                assert (p,t) not in raw_joint;raw_joint.add((p,t))
                if all(math.isfinite(x) for x in xyz):joint.add((p,t))
        assert fields=={'label_id','keypoints'}
        overlap=joint & gt
        # HST on actual source frame identifiers. No manufactured positions between observations.
        any_five={(p,t) for p,t in joint if all((p,t+i) in joint and (t+i) in odom for i in OFFSETS)}
        hold=any_five & gt
        with_ground_previous={k for k in hold if (k[0],k[1]-6) in gt}
        all5ground={k for k in hold if all((k[0],k[1]+i) in gt for i in OFFSETS)}
        end45={(p,t) for p,t in gt if (p,t+45) in gt}
        seven={(p,t) for p,t in with_ground_previous if all((p,t+6*j) in gt and t+6*j in odom for j in range(1,8))}
        eight={(p,t) for p,t in with_ground_previous if all((p,t+6*j) in gt and t+6*j in odom for j in range(1,9))}
        times=sorted({t for p,t in gt});jf=sorted({t for p,t in joint})
        scenes[scene]=dict(valid_ground_person_frames=len(gt),distinct_ground_frames=len(times),ground_frame_range=[times[0],times[-1]],
          ground_modulo6=sorted({t%6 for t in times}),odometry_frame_range=[min(odom),max(odom)],odometry_frames=len(odom),
          all33_finite_hst_person_frames=len(joint),all_hst_person_frames=len(raw_joint),finite_hst_frame_range=[min(jf),max(jf)],
          hst_ground_person_frame_overlap=len(overlap),five_hst_frames_with_odom_any_anchor=summarize(any_five,0),
          five_hst_with_final_ground_anchor=summarize(hold,0),five_hst_with_both_endpoint_historic_ground=summarize(with_ground_previous,0),
          five_ground_positions_at_actual_five_hst_frames=len(all5ground),ground_endpoint_plus45_simultaneous_anchor=len(end45),
          five_hst_and_2_ground_history_and_seven_actual_future_steps=summarize(seven,42),
          five_hst_and_2_ground_history_and_eight_actual_future_steps=summarize(eight,48),official_measured_2d_and_ground_windows_certified=0)
    return {'status':'Coverage audit only; three previously used outdoor train sequences. No forecast or new test scene.',
            'raw_member_sha256':hashes,'clocks':{'raw_image_nominal_hz':15,'manual_2d_pose_nominal_hz':7.5,'official_upsampled_nominal_hz':15,
            'ground_derived_hz':2.5,'raw_frames_in_0_1s':str(Fraction(15,10)),'five_manual_poses_min_span_s':str(Fraction(8,15)),
            'nearby_five_real_hst_frame_offsets':OFFSETS,'nearby_intervals_seconds':[str(Fraction(b-a,15)) for a,b in zip(OFFSETS,OFFSETS[1:])],
            'nominal_3_seconds_in_raw_frames':45,'manual_box_flags_or_capture_ns_in_local_derivatives':False},
            'raw_readout_scope':'HST 33 body-relative 3D estimates per source ID, no original RGB, 2D joint confidence, raw measured box, or capture timestamps in these nine training members.',
            'scenes':scenes}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    r=check(a.root);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    for scene,rows in r['scenes'].items():
        print(scene,rows['five_hst_frames_with_odom_any_anchor'],rows['five_hst_and_2_ground_history_and_seven_actual_future_steps'],rows['five_hst_and_2_ground_history_and_eight_actual_future_steps'])
