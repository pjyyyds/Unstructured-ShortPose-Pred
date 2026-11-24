"""Measure only the five original annotated positions in each registered DUT input."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

def summarize(cohort,coverage,data):
    report={}
    lengths=[]
    keys=[]
    actual_spans=[]
    for meta in coverage["clips"]:
        name=meta["clip"]
        points={}
        with (data/"data"/"trajectories"/(name+"_traj_ped.csv")).open() as source:
            for row in csv.DictReader(source):
                assert row["label"]=="ped"
                identifier=(int(row["id"]),int(row["frame"]))
                if identifier in points:
                    raise ValueError("Duplicate original source key")
                points[identifier]=[float(row["x"]),float(row["y"])]
        group=[]
        for r in cohort:
            if r["key"][0]!=name:continue
            _,actor,t0=r["key"]
            observed=r["frames_1based"][:5]
            assert len(set(observed))==5 and observed[-1]==t0
            xy=np.array([points[(actor,frame)] for frame in observed])/meta["pixels_per_metre"]
            length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())
            group.append(length);keys.append(r["key"])
            actual_spans.append(r["time_s"][4]-r["time_s"][0])
        assert len(group)==meta["windows"]
        lengths.extend(group)
        report[name]={"windows":len(group),"below_0p5_m":sum(v<0.5 for v in group),
                      "median_m":float(np.median(group)),"pixels_per_metre":meta["pixels_per_metre"]}
    raw=json.dumps(keys,separators=(",",":")).encode()
    return {"scope":"Path length from four adjacent ground-plane displacements in the five historical image timestamps only. Future positions, neural forecasts and joint estimates do not enter the statistic.",
            "ordered_keys_sha256":hashlib.sha256(raw).hexdigest(),"windows":len(lengths),"short_windows":sum(v<0.5 for v in lengths),
            "median_m":float(np.median(lengths)),"q25_q75_m":np.quantile(lengths,[0.25,0.75]).tolist(),
            "observed_span_s_min_median_max":np.quantile(actual_spans,[0,0.5,1]).tolist(),"by_clip":report}
if __name__=="__main__":
    base,out=map(Path,sys.argv[1:])
    result=summarize(json.loads((base/"private_cohort.json").read_text()),json.loads((base/"coverage.json").read_text()),base/"private_source")
    assert len(result["by_clip"])==11 and result["windows"]==3066
    assert result["ordered_keys_sha256"]==json.loads((base/"coverage.json").read_text())["key_manifest_sha256"]
    out.write_text(json.dumps(result,indent=2)+"\n")
