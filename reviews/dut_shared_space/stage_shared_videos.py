"""Read only the eleven selected official RAR members and verify uncompressed CRC/size."""
from stage_sources import CLIPS,D,SIZE,span
import binascii,concurrent.futures,json,subprocess
def run():
    cat=json.loads((D/'author_video_catalog.json').read_text());names=['vci-raw-dut/'+s+'_roi.mp4' for s in CLIPS];full=[cat[s] for s in names];buf_start=full[0]['offset'];end=full[-1]['data_offset']+full[-1]['compressed'];assert all(a['data_offset']+a['compressed']==b['offset'] for a,b in zip(full,full[1:]))
    tar=D/'private_shared_videos.rar'
    if not tar.exists():
        pieces=[(i,min(end,i+3_000_000)-1) for i in range(buf_start,end,3_000_000)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as p:body=b''.join(p.map(lambda x:span(*x),pieces))
        tar.write_bytes(span(0,min(v['offset'] for v in cat.values())-1)+body+span(end,SIZE-1))
    root=D/'private_source';subprocess.run(['bsdtar','-xf',str(tar),'-C',str(root),*names],check=True)
    for name,info in zip(names,full):
        b=(root/name).read_bytes();assert len(b)==info['uncompressed'] and (binascii.crc32(b)&0xffffffff)==info['crc']
    print('archive header and selected span:',buf_start,end,sum(f['compressed'] for f in full),'bytes; extracted',len(full))
if __name__=='__main__':run()
