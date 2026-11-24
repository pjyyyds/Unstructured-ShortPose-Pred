"""Create one private input and two source-label notebooks, preserving both original archived executables."""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
import zlib

PARENT_KIT = "2e4646ca4bf54fcfb2c714fe79f4da7d938a9992181716b3f31d131a92a8e339"
PARENT_NB = "94b86d434b08f9273a20ed70cd9b39c3f087301ef30fb20d1a7303d58389cf0d"
OLD_BIN_SHA = "362efbaa576340495749b0820f398d768b82e000e714492265b91183ef061873"
OLD_PLAN = "b3f33a5b8a4216fb4240d4b3412145315a510d0bd4eaad21cce542de7137f52c"
NEW_PLAN = "933949fa5a8df237f785027499a18da55bb43966d4191dfc16d4c2d30fa51bda"
PRIVATE = "pinns_sync_latency_source_label_private.bin"
OLD_BIN = "pinns_sync_latency_private.bin"
DEF_NAME = "01_PINNS_source_label_sync_latency.ipynb"
PAIR_NAME = "02_PINNS_source_label_pose_ab.ipynb"


def sha(raw):return hashlib.sha256(raw).hexdigest()

def dump(d):return (json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()

def archive(contents):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w") as z:
        for k,v in sorted(contents.items()):
            info=zipfile.ZipInfo(k,date_time=(2026,1,1,0,0,0));info.external_attr=0o100644<<16
            z.writestr(info,v,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with zipfile.ZipFile(buf) as z:assert z.testzip() is None
    return buf.getvalue()

def exactly(raw,old,new):
    assert raw.count(old)==1, old
    return raw.replace(old,new)

def nb_source(c):return "".join(c["source"])

def nb_patch(c,text):c["source"]=text.splitlines(keepends=True)

def build(parent, parent_ab, sync, public, zip_path):
    assert sha(parent.read_bytes())==PARENT_KIT and sha(parent_ab.read_bytes())==PARENT_NB
    with zipfile.ZipFile(parent) as z:
        oldbin=z.read(OLD_BIN);default=json.loads(z.read("PINNS_sync_latency.ipynb"))
    assert sha(oldbin)==OLD_BIN_SHA
    with zipfile.ZipFile(io.BytesIO(oldbin)) as z:
        assert z.testzip() is None
        items={k:z.read(k) for k in z.namelist()}
    plan=(sync/"protocol.json").read_bytes()
    assert sha(plan)==NEW_PLAN
    doc=json.loads(plan);assert doc["parent"]["protocol_sha256"]==OLD_PLAN and doc["version"]=="source_label_sync_v2"
    for f,h in doc["files_sha256"].items():
        raw=(sync/f).read_bytes();assert sha(raw)==h;items["sync/"+f]=raw
    items["sync/protocol.json"]=plan
    old_runner=items["measure_latency.py"];runner=exactly(old_runner,OLD_PLAN.encode(),NEW_PLAN.encode())
    items["measure_latency.py"]=runner
    private=archive(items);private_h=sha(private)
    nb_patch(default["cells"][0],nb_source(default["cells"][0]).replace(OLD_BIN,PRIVATE)
             .replace("同一时刻 12 名行人的预测前向耗时","修正鸟瞰坐标后的同刻多行人前向耗时")
             .replace("pinns_sync_latency_results.zip","pinns_sync_latency_source_label_results.zip"))
    nb_patch(default["cells"][1],nb_source(default["cells"][1]).replace(OLD_BIN,PRIVATE).replace(OLD_BIN_SHA,private_h)
             .replace("pinns_sync_latency_results.zip","pinns_sync_latency_source_label_results.zip"))
    raw_nb=dump(default)
    paired=json.loads(parent_ab.read_bytes())
    prep=nb_source(paired["cells"][1])
    node=next(n for n in ast.parse(prep).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="sources")
    payload=ast.literal_eval(node.value)
    decoded={k:zlib.decompress(base64.b64decode(v[1])) for k,v in payload.items()}
    assert all(sha(v)==payload[k][0] for k,v in decoded.items())
    old_paired=decoded["paired_pose_compute.py"]
    paired_runner=exactly(old_paired,sha(old_runner).encode(),sha(runner).encode())
    paired_runner=exactly(paired_runner,OLD_BIN_SHA.encode(),private_h.encode())
    paired_runner=exactly(paired_runner,OLD_BIN.encode(),PRIVATE.encode())
    paired_runner=exactly(paired_runner,b"attach the original private .bin",b"attach the corrected source-label private .bin")
    updated=dict(payload)
    updated["paired_pose_compute.py"]=[sha(paired_runner),base64.b64encode(zlib.compress(paired_runner,9)).decode()]
    nb_patch(paired["cells"][1],exactly(prep,repr(payload),repr(updated)))
    nb_patch(paired["cells"][0],nb_source(paired["cells"][0]).replace(OLD_BIN,PRIVATE)
             .replace("上一次同刻时延实验的私有输入","修正 BEV 坐标的共用私有输入")
             .replace("姿态图乘法的 CUDA 配对验证","修正地图上的姿态图乘法 CUDA 配对验证")
             .replace("pinns_paired_pose_compute.zip","pinns_paired_pose_compute_source_label.zip"))
    nb_patch(paired["cells"][2],nb_source(paired["cells"][2]).replace("pinns_paired_pose_compute.zip","pinns_paired_pose_compute_source_label.zip"))
    paired_nb=dump(paired)
    public.mkdir(parents=True,exist_ok=True)
    info={"variant":"source_label_sync_v2","files_sha256":{
        PRIVATE:private_h,DEF_NAME:sha(raw_nb),PAIR_NAME:sha(paired_nb),"measure_latency.py":sha(runner),
        "paired_pose_compute.py":sha(paired_runner),"pose_inference.py":sha(decoded["pose_inference.py"]),
        "sync/protocol.json":NEW_PLAN,**{"sync/"+k:v for k,v in doc["files_sha256"].items()}},
        "freeze":{"original_latency_package_sha256":PARENT_KIT,"original_paired_notebook_sha256":PARENT_NB,
                  "original_public_latency_runner_sha256":sha(old_runner),"original_public_paired_runner_sha256":sha(old_paired),
                  "model_sha256":doc["checkpoint_sha256"]},"input_provenance":doc,
        "scope":"Two scenes, 24 people, 3 durations and 72 timed loads. Source-label map units fixed. No real forward or timing at package preparation."}
    for k,v in {"SOURCE_LOCK.json":dump(info),"measure_latency.py":runner,"paired_pose_compute.py":paired_runner,
                "sync_protocol.json":plan}.items():(public/k).write_bytes(v)
    records={PRIVATE:private,DEF_NAME:raw_nb,PAIR_NAME:paired_nb,"来源与指纹.json":dump(info),
             "使用说明.md":(Path(__file__).parent/"使用说明.md").read_bytes()}
    records["SHA256SUMS.txt"]="".join(f"{sha(v)}  {k}\n" for k,v in sorted(records.items())).encode()
    zip_path.write_bytes(archive(records))
    print(private_h,sha(runner),sha(paired_runner),sha(zip_path.read_bytes()),zip_path.stat().st_size)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("original_latency_zip","original_ab_notebook","corrected_sync","public_directory","new_package"):p.add_argument(arg,type=Path)
    a=p.parse_args();build(a.original_latency_zip,a.original_ab_notebook,a.corrected_sync,a.public_directory,a.new_package)
