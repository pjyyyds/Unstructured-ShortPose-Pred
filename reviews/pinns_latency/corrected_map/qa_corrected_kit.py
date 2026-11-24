"""Validate corrected private delivery without any measured or real-data model forward."""
import argparse
import ast
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import zipfile

import nbformat
import numpy as np


def sha(x):return hashlib.sha256(x).hexdigest()


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def verify(package,original,repo,metadata,bevs):
    import sys
    sys.path.insert(0,str(repo.resolve()))
    from scripts.build_pinns_causal_maps import edges_from_frames
    from src.data.scene_graph import build_local_polylines
    from src.data.pinns_extractor import _parse_pinns_label_file,_check_right_handed_homography
    with zipfile.ZipFile(package) as outer,zipfile.ZipFile(original) as old:
        assert outer.testzip() is None
        for line in outer.read("SHA256SUMS.txt").decode().splitlines():
            expected,k=line.split("  ",1);assert sha(outer.read(k))==expected
        summary=json.loads(outer.read("来源与指纹.json"))
        nb=[]
        for k in sorted(x for x in outer.namelist() if x.endswith(".ipynb")):
            doc=nbformat.reads(outer.read(k).decode(),as_version=4);nbformat.validate(doc)
            for cell in doc.cells:
                if cell.cell_type=="code":compile(cell.source,"kaggle","exec")
            nb.append(doc)
        assert len(nb)==2
        binary="pinns_sync_latency_source_label_private.bin";b=outer.read(binary)
        assert sha(b)==summary["files_sha256"][binary]
        oldbin=old.read("pinns_sync_latency_private.bin")
        with zipfile.ZipFile(io.BytesIO(b)) as new,zipfile.ZipFile(io.BytesIO(oldbin)) as prev:
            assert new.testzip() is None and len(new.namelist())==16
            assert set(new.namelist())==set(prev.namelist())
            assert all(new.read(k)==prev.read(k) for k in new.namelist() if k.startswith("model/"))
            assert sha(new.read("sync/protocol.json"))==summary["files_sha256"]["sync/protocol.json"]
            mapped=0
            for t in (5,10,20):
                k=f"sync/sync_T{t}.npz"
                assert sha(new.read(k))==summary["files_sha256"][k]
                with np.load(io.BytesIO(new.read(k)),allow_pickle=False) as x,np.load(io.BytesIO(prev.read(k)),allow_pickle=False) as y:
                    assert len(x.files)==14 and set(x.files)==set(y.files)
                    for field in set(x.files)-{"p_env"}:
                        assert x[field].dtype==y[field].dtype and x[field].shape==y[field].shape
                        assert x[field].tobytes()==y[field].tobytes()
                    store={key:x[key] for key in ("clip","t0","person_id","p_env","m_map")}
                    for clip in sorted(set(store["clip"])):
                        cond,id=clip.rsplit("_",1); c=json.loads((metadata/"videos"/cond/"calib.json").read_bytes())
                        f=_check_right_handed_homography(c)
                        src=_parse_pinns_label_file(metadata/"labels"/(clip+".txt"),f,c["y_min"],c["y_max"])
                        lines=edges_from_frames(bevs/"videos"/cond/"videos"/f"bev_{id}.mp4",[0],c,f,bev_convention="source_label")
                        for i in np.flatnonzero(store["clip"]==clip):
                            rows=[r for r in src[int(store["t0"][i])] if r["type"]=="Person" and r["id"]==int(store["person_id"][i])]
                            assert len(rows)==1
                            xy=np.array([rows[0]["wx"],rows[0]["wy"]],np.float32)
                            corrected,m=build_local_polylines(lines,xy,bool(lines))
                            np.testing.assert_array_equal(corrected,store["p_env"][i])
                            assert m==float(store["m_map"][i,0])
                            mapped+=1
            assert mapped==72
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);mount=root/"input";working=root/"working";mount.mkdir();working.mkdir()
            (mount/binary).write_bytes(b)
            # Run the actual notebook setup code with local temporary staging directories.
            # The measured sections are only compiled.
            ctx={}
            src=nb[0].cells[1].source.replace("/kaggle/input",str(mount)).replace("/kaggle/working",str(working))
            src=src.replace('mkdtemp(prefix=',f'mkdtemp(dir={str(root)!r}, prefix=')
            exec(src,ctx)
            checked=json.loads((ctx["stage_dir"]/"checked.json").read_bytes())
            assert checked["status"]=="check_only_no_forwards" and len(checked["plan"])==72 and "cells" not in checked
            ctx2={}; src=nb[1].cells[1].source.replace('mkdtemp(prefix=',f'mkdtemp(dir={str(root)!r}, prefix=')
            exec(src,ctx2); path=ctx2["code_dir"]
            ab=module(path/"paired_pose_compute.py","validated_paired_source")
            assert sha((path/"paired_pose_compute.py").read_bytes())==summary["files_sha256"]["paired_pose_compute.py"]
            assert ab.BUNDLE==binary and ab.BUNDLE_SHA==sha(b)
            paused=root/"paired";paused.mkdir()
            baseline=ab.stage(mount,paused,path/"pose_inference.py")
            assert baseline.PROTOCOL_HASH==summary["files_sha256"]["sync/protocol.json"]
            proto,_,groups=baseline.load_fixture(paused/"sync")
            assert proto["version"]=="source_label_sync_v2" and sum(map(len,groups.values()))==6
            assert "pinns_paired_pose_compute_source_label.zip" in nb[1].cells[2].source
        print("Both notebooks compile and validate; baseline real check-only passed; paired staging locks and 72 independently rebuilt per-person maps pass; no real-data model forward.")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("package","original_latency_zip","repo","metadata","bev"):p.add_argument(k,type=Path)
    a=p.parse_args();verify(a.package,a.original_latency_zip,a.repo,a.metadata,a.bev)
