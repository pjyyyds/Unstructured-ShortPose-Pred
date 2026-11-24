"""Only the 11 author-identified shared-space video members and source metadata."""
import concurrent.futures, hashlib, io, json, os, re, subprocess, sys, urllib.request, zlib
from pathlib import Path
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'python_deps'))
import rarfile
FREEZE=json.loads((D/'shared_space_freeze.json').read_text());CLIPS=FREEZE['clips']
BASE='https://raw.githubusercontent.com/dongfang-steven-yang/vci-dataset-dut/'+FREEZE['source_commit']+'/'
ARCHIVE='https://drive.usercontent.google.com/download?id=1J0PW4NoL2mi_eaN_8_SgkWkc0rk7MJZC&confirm=t';SIZE=104243676
def span(a,b):
    req=urllib.request.Request(ARCHIVE,headers={'Range':f'bytes={a}-{b}','Accept-Encoding':'identity'})
    with urllib.request.urlopen(req,timeout=40) as r:
        assert r.status==206 and r.headers.get('Content-Range')==f'bytes {a}-{b}/{SIZE}';blob=r.read(b-a+2);assert len(blob)==b-a+1;return blob
class Remote(io.RawIOBase):
    def __init__(self):self.at=0;self.cache={}
    def seekable(self):return True
    def seek(self,offset,whence=0):self.at=offset if whence==0 else (self.at+offset if whence==1 else SIZE+offset);return self.at
    def tell(self):return self.at
    def read(self,size=-1):
        assert size!=-1 and size<=2_000_000
        if self.at>=SIZE:return b''
        n=min(size,SIZE-self.at);out=b''
        while len(out)<n:
            pos=self.at+len(out);idx=pos//4096;lo=idx*4096
            if idx not in self.cache:self.cache[idx]=span(lo,min(SIZE-1,lo+4095))
            out+=self.cache[idx][pos-lo:pos-lo+n-len(out)]
        self.at+=n
        return out
def stage_meta():
    tree=json.loads((D/'author_tree.json').read_text())['tree'];lookup={r['path']:r for r in tree}
    wanted=[name for clip in CLIPS for name in ['data/trajectories/'+clip+'_traj_'+typ+'.csv' for typ in ('ped','veh')]+['data/ratios/'+clip+'_ratio_pixel2meter.txt','legacy/clips/rois/'+clip+'_roi']]
    def fetch(n):
        path=D/'private_source'/n;path.parent.mkdir(parents=True,exist_ok=True);b=path.read_bytes() if path.exists() else urllib.request.urlopen(BASE+n,timeout=40).read();meta=lookup[n]
        assert len(b)==meta['size'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==meta['sha'];path.write_bytes(b)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(fetch,wanted))
def catalog():
    remote=Remote();rf=rarfile.RarFile(remote);out={}
    for i in rf.infolist():
        if i.is_file():out[i.filename]={'compressed':i.compress_size,'uncompressed':i.file_size,'offset':i.header_offset,'data_offset':i.data_offset,'crc':i.CRC,'method':i.compress_type}
    (D/'author_video_catalog.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':
    stage_meta();c=catalog();print(len(c),list(c.items())[:2]);print([k for k in c if 'roundabout' in k])
