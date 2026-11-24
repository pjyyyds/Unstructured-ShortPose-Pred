"""Test whether the archived fifth predecessor search changes any five-frame graph input."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np


FIELDS=("ego_type_oh","nbr_edges","nbr_type_oh","nbr_mask")
CHANNELS=("dx_m","dy_m","dvx_m_per_s","dvy_m_per_s","distance_m","radial_m_per_s","speed_valid","rho_neighbor","rho_ego")


def sha(data):return hashlib.sha256(data).hexdigest()


def npz_fields(raw,fields):
    with np.load(io.BytesIO(raw),allow_pickle=False) as a:
        return {k:a[k] for k in fields}


def load_archives(sidecar,pose,graph):
    assert sha(sidecar.read_bytes())=="47453819b0542cefadb0122685d5a45876596191c21d0a855d35d596c8ee8fe7"
    with np.load(sidecar,allow_pickle=False) as z:
        original={tuple(k):s for k,s in zip(zip(z['clip'].tolist(),z['person_id'].tolist(),z['t0'].tolist()),z['split'].tolist())}
    assert len(original)==7283 and Counter(original.values())=={"train":5686,"val":1100,"test":497}
    with zipfile.ZipFile(pose) as z:
        raw=z.read("pose.npz");assert sha(raw)=="b493a47536664ddb4fd66603c1bbf6b2c2e33f553520382027b3ae96b53447d0"
        pilot=npz_fields(raw,("clip","person_id","t0","ego_type","hist_mask","hist_pos_relative"))
    with zipfile.ZipFile(graph) as z:
        assert sha(graph.read_bytes())=="0b3e590b8d7cb2bf99670fd27d3bdbf2a40fa0e77bf29f7ac0423b73535589f8"
        net=npz_fields(z.read("private.npz"),("clip","person_id","t0","nbr_edges","nbr_type_oh","nbr_mask","velocity_gap_ticks",
                                             "source_class","neighbor_id"))
    index=lambda z:{tuple(k):i for i,k in enumerate(zip(z["clip"].tolist(),z["person_id"].tolist(),z["t0"].tolist()))}
    pi,gi=index(pilot),index(net);assert set(pi)==set(gi) and len(pi)==1482 and len(gi)==1482
    return original,pilot,pi,net,gi


def modified(fn,changes):
    src=inspect.getsource(fn)
    for old,new in changes:
        if src.count(old)!=1:raise ValueError(f"Frozen code changed: {old}")
        src=src.replace(old,new)
    env=dict(fn.__globals__)
    exec(compile(src,"<local diagnostic counterfactual>","exec"),env)
    return env[fn.__name__]


class Aggregate:
    def __init__(self):
        self.count=Counter()
        self.changed=np.zeros(9,np.int64);self.low=np.full(9,np.inf);self.high=np.full(9,-np.inf)
        self.changed_h=Counter()
    def add(self,old,new,hist):
        e,n0,ty,m=(old[k] for k in FIELDS)
        e1,n1,ty1,m1=(new[k] for k in FIELDS)
        active=m.astype(bool)
        changed_e=np.any(n0!=n1,axis=1);changed_t=np.any(ty!=ty1,axis=1)
        different=bool(not np.array_equal(e,e1) or not np.array_equal(m,m1) or changed_e.any() or changed_t.any())
        lack=active & np.isclose(n0[:,7],.2,rtol=0,atol=1e-7)
        self.count.update(windows=1,ego_standing=int(e[2]),ego_type_changes=int(not np.array_equal(e,e1)),
                exposed_ego_with_only_t0=int(hist==1),windows_with_all_five_positions=int(hist==5),
                windows_with_any_neighbor=int(active.any()),selected_slots=int(active.sum()),neighbor_velocity_valid=int(n0[active,6].sum()),
                selected_person_slots=int(np.sum(active & (ty[:,6]==0))),selected_car_slots=int(np.sum(active & (ty[:,6]==1))),
                selected_neighbor_slots_without_previous_point_at_lags_1_to_4=int(lack.sum()),
                windows_with_such_neighbor=int(lack.any()),changed_graph_windows=int(different),
                changed_neighbor_value_slots=int(changed_e.sum()),changed_neighbor_type_slots=int(changed_t.sum()),
                changed_membership_mask_slots=int((m!=m1).sum()))
        self.changed_h[int(hist)]+=1
        d=n0.astype(np.float64)-n1.astype(np.float64)
        if active.any():
            v=d[active];self.changed+=np.sum(v!=0,axis=0)
            self.low=np.minimum(self.low,v.min(0)); self.high=np.maximum(self.high,v.max(0))
    def as_dict(self):
        return dict(self.count,histogram_valid_ego_positions_in_last_five=dict(sorted(self.changed_h.items())),
               channels={k:{"changed_slots":int(self.changed[i]),"min_signed_change":float(self.low[i]),
                           "max_signed_change":float(self.high[i])} for i,k in enumerate(CHANNELS)})


def all_source_gaps(frames):
    tracks=defaultdict(set)
    for t,objects in frames.items():
        for o in objects:tracks[o["type"],o["id"]].add(t)
    result=defaultdict(Counter)
    for (kind,_),ticks in tracks.items():
        for t in ticks:
            g=next((i for i in range(1,6) if t-i in ticks),0)
            result[kind][g]+=1
    return result


ARCHIVE_COMMIT = "c5344f1e5c1efea657381f7d2e6d4e74dd1db977"
ARCHIVE_FILE = "src/data/pinns_extractor.py"
ARCHIVE_SHA256 = "26521a7284675d38b2cf60806ea303327834e8a65e5bf80fd92933651fc29d9b"


def archived_assembler(repo, current, explicit):
    if explicit is None:
        raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{ARCHIVE_COMMIT}:{ARCHIVE_FILE}"])
    else:
        raw = explicit.read_bytes()
    if sha(raw) != ARCHIVE_SHA256:
        raise ValueError("The prior extractor does not match the released V2 source")
    text = raw.decode()
    archive = next(node for node in ast.parse(text).body
                   if isinstance(node, ast.FunctionDef) and node.name == current.__name__)
    source = ast.get_source_segment(text, archive) + "\n"
    scope = dict(current.__globals__)
    exec(compile(ast.Module(body=[archive], type_ignores=[]), "<archived PINNS assembly>", "exec"), scope)
    # Check that the real corrected function differs only at the search bound
    # and in its later addition of person_id metadata to identify the returned row.
    expected = ast.parse(source.replace("for back in range(1, 6):", "for back in range(1, 5):")).body[0]
    actual = ast.parse(inspect.getsource(current)).body[0]
    metadata_found = False
    for node in ast.walk(actual):
        if not isinstance(node, ast.Dict):
            continue
        names = [key.value if isinstance(key, ast.Constant) else None for key in node.keys]
        if "seq_id" in names and "person_id" in names:
            index = names.index("person_id")
            if metadata_found or ast.unparse(node.values[index]) != "int(oid)":
                raise ValueError("Unexpected person identity output")
            node.keys.pop(index)
            node.values.pop(index)
            metadata_found = True
    if not metadata_found or ast.dump(actual) != ast.dump(expected):
        raise ValueError("The formal implementation has additional changes beyond the audited scope")
    return scope[current.__name__], source


def run(repo,root,sidecar,pose,graph,prior):
    sys.path.insert(0,str(repo.resolve()))
    from src.data.pinns_extractor import _assemble_sequence_windows
    from scripts.build_pinns_causal_maps import clips_in_order
    from scripts.audit_pinns_map_sources import target_id
    from reviews.pinns_graph_control.run_graph_control import graph as control_graph,typed_frame
    original,pilot,pi,net,gi=load_archives(sidecar,pose,graph)
    assemble,archive_source = archived_assembler(repo, _assemble_sequence_windows, prior)
    stricter = _assemble_sequence_windows
    broader=modified(control_graph,[
        ("if len(history)!=5:","if len(history)!=6:"),
        ("ego_xy=np.stack([snap[ego] for snap in history])","ego_xy=np.stack([snap[ego] for snap in history[-5:]])"),
        ("count=sum(k in snap for snap in history)","count=sum(k in snap for snap in history[-5:])"),
        ("d for d in range(1,5)","d for d in range(1,6)")])
    counts=defaultdict(Aggregate); gaps=defaultdict(Counter)
    seen_original=set();seen_held=set()
    raw_graph_fields=Counter()
    labels=Counter()
    for item in clips_in_order(root):
        stem=item.label.stem;frames=item.frames
        before=assemble(stem,frames,{},[],item.triggers,item.calibration["v_size"],4)
        after=stricter(stem,frames,{},[],item.triggers,item.calibration["v_size"],4)
        assert len(before)==len(after)
        tracks=defaultdict(dict)
        for t,obs in frames.items():
            for r in obs:
                if r["type"]=="Person":tracks[r["id"]][t]=np.asarray([r["wx"],r["wy"]],dtype=np.float32)
                labels[r["type"],r["fixed"]]+=1
        for kind,stats in all_source_gaps(frames).items():gaps[kind].update(stats)
        for old,new in zip(before,after):
            t=old["t0_frame_10hz"];k=stem,target_id(old,tracks),t
            assert k not in seen_original and original[k]==old["split"]==new["split"] and t==new["t0_frame_10hz"]
            assert k[1] == new["person_id"]
            np.testing.assert_array_equal(old["p_t0"],new["p_t0"]); seen_original.add(k)
            for group in ["archived/all","archived/"+old["split"],"archived/"+item.site+"/"+old["split"]]:
                counts[group].add(old,new,int(old["hist_mask_20"][-5:].sum()))
        snaps={t:typed_frame(v) for t,v in frames.items()}
        for k,j in pi.items():
            if k[0]!=stem:continue
            _,pid,t=k; jj=gi[k];ego=("Person",pid)
            his=[snaps.get(d,{}) for d in range(t-5,t+1)]
            xy=np.stack([s[ego] for s in his[-5:]]);rel=xy-xy[-1]
            np.testing.assert_allclose(rel,pilot["hist_pos_relative"][j],rtol=0,atol=1e-4)
            assert np.all(pilot["hist_mask"][j]==1)
            old=control_graph(his[-5:],pid,rel); new=broader(his,pid,rel)
            for ii,key in enumerate(("nbr_edges","nbr_type_oh","nbr_mask","neighbor_id","source_class","velocity_gap_ticks")):
                np.testing.assert_array_equal(old[ii],net[key][jj]);raw_graph_fields[key]+=1
            source_ego=next(o for o in frames[t] if o['type']=="Person" and o['id']==pid)
            source_fixed=source_ego["fixed"]
            gettype=lambda history:2 if source_fixed==1 or np.linalg.norm(history[-1][ego]-history[-2][ego])/.1<.20 else 0
            code=gettype(his[-5:]); code1=gettype(his)
            assert code==code1==pilot["ego_type"][j]
            def record(c,g):return dict(zip(FIELDS,(np.eye(8)[c],*g[:3])))
            for group in ["heldout/all","heldout/"+item.site]:
                counts[group].add(record(code,old),record(code1,new),5)
            seen_held.add(k)
    assert seen_original==set(original) and seen_held==set(pi)
    return {"source_label_and_calibration_sha256":"0b276b34ab1b01dae91f67c56bfaea5cc931511b05b4fc1c92686a891e6f6801",
        "archived_function_sha256":sha(archive_source.encode()),
        "current_function_sha256":sha(inspect.getsource(stricter).encode()), "archived_commit":ARCHIVE_COMMIT,
        "archived_source_file_sha256":ARCHIVE_SHA256, "historical_and_production_assembly_both_executed":True,
        "current_assembly_ast_changes":"Uses lags 1..4 in place of 1..5 and returns the same independently recovered person ID in sample metadata.","control_graph_function_sha256":sha(inspect.getsource(control_graph).encode()),
        "program_sha256":sha(Path(__file__).read_bytes()),"numpy":np.__version__,
        "source_counts_by_nearest_t0_predecessor_lag_1_to_5_zero_none":{k:{str(i):c[i] for i in range(6)} for k,c in gaps.items()},
        "source_t0_fixed_by_class":{k:{str(f):v for (typ,f),v in labels.items() if typ==k} for k in gaps},
        "frozen_key_counts":{"historical":len(original),"historical_test":497,"four_site":len(pi)},
        "strict_heldout_graph_exact_match_to_returned_tensor_fields_by_window":dict(raw_graph_fields),
        "cohorts":{k:v.as_dict() for k,v in sorted(counts.items())},"interpretation":[
         "Raw IDs are typed and clip-local; source scan is over every observed object at raw frame 1+3*t. Zero means no same-object predecessor in the previous five ticks.",
         "The 7,283 cohort and splits match the causal-sidecar keys. The 497-test subset is contained in 7,283. Historical assembly checks t0-5 for membership if the previous four ticks are missing. When absent, both paths emit identical fallback, and never read a position at that tick.",
         "The pinned full archival assembly and current checked-in production assembly run separately; the latter uses four predecessor ticks and adds person ID metadata; no pose model, prediction network, optimizer, forecast score or image is used. Future availability is used by the original archived enrollment, not to form the graph.",
         "Actual LOSO pilot M0/M1 consume ego type and motion (M1 also pose and quality); neither receives neighbor edge tensors. Its separate real-graph control only receives t0-4..t0: the table compares its true five snapshots vs a hypothetical extra predecessor, preserving the five-tick visibility ratio.",
         "Rows without a qualifying predecessor are counted even when source speed is unavailable. The code can distinguish re-appearance at a five-tick gap on other data.",
         "Archive channel signed changes are legacy minus strict; heldout changes are registered strict minus hypothetical broader. Per-channel extrema cover active slots only; masks and categorical one-hots are compared separately."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, help="Original labels, calibrations and archive member inventory")
    parser.add_argument("sidecar", type=Path, help="Original causal-map maps.npz, used only for frozen cohort keys and splits")
    parser.add_argument("pilot", type=Path, help="Original shortpose_heldout.zip with pose.npz")
    parser.add_argument("graph", type=Path, help="Original pinns_graph_control.zip with recorded graph inputs")
    parser.add_argument("output", type=Path, help="Aggregate JSON output")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--prior-extractor", type=Path, help="Exact V2 source file if the commit is absent from a shallow checkout")
    args = parser.parse_args()
    report = run(args.repo, args.metadata, args.sidecar, args.pilot, args.graph, args.prior_extractor)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name in ("archived/all", "archived/test", "heldout/all"):
        row = report["cohorts"][name]
        print(name, row["windows"], "windows,", row["selected_slots"], "edges,", row["changed_graph_windows"], "changed")
