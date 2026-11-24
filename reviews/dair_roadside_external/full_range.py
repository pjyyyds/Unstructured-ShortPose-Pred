"""Read the author's two public full train/validation ZIP directories and requested members by HTTP range."""
import concurrent.futures, hashlib, json, re, struct, time, urllib.request, zipfile, zlib
from pathlib import Path
BASE=Path(__file__).resolve().parent
VOLUMES={'metadata':('1SuJOyI3G8MstCY-EHS5tbWvaMh2iZuef',234358463),'images':('12B4FNZM9G3Q-kRr-h9vdt-p2c5bhGBcB',3560631820)}
def span(volume,lo,hi):
    fid,size=VOLUMES[volume]
    if not 0<=lo<=hi<size: raise ValueError('range')
    uri='https://drive.usercontent.google.com/download?id='+fid+'&confirm=t'
    for n in range(4):
        try:
            req=urllib.request.Request(uri,headers={'Range':f'bytes={lo}-{hi}','Accept-Encoding':'identity','User-Agent':'Roadside-pose-coverage/1'})
            with urllib.request.urlopen(req, timeout=35) as response:
                if response.status!=206 or response.headers['Content-Range']!=f'bytes {lo}-{hi}/{size}':raise ValueError('Unexpected HTTP byte range')
                blob=response.read(hi-lo+2)
            if len(blob)!=hi-lo+1:raise ValueError('Short range')
            return blob
        except (OSError,ValueError):
            if n==3:raise
            time.sleep(n+1)
def directory(volume):
    path=BASE/('full_'+volume+'_central.bin')
    size=VOLUMES[volume][1];tail=span(volume,size-22,size-1)
    assert tail[:4]==b'PK\x05\x06' and tail[-2:]==bytes(2)
    count,byte_len,start=struct.unpack_from('<HII',tail,10)
    if not path.exists():
        groups=[(i,min(start+byte_len-1,i+2_000_000-1)) for i in range(start,start+byte_len,2_000_000)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:payload=b''.join(pool.map(lambda g:span(volume,*g),groups))
        path.write_bytes(payload)
    payload=path.read_bytes();assert len(payload)==byte_len
    frozen={'metadata':'ebc52abdb00872e46676f7e04c3ce4a6c85949206ebf33358a5e2f990f883186','images':'9d3e248b520b5cd769df1916f6d0f82f0ac8eeb5853aa4082907a291db575f08'}
    if hashlib.sha256(payload).hexdigest()!=frozen[volume]:raise ValueError('Public ZIP directory changed: '+volume)
    entries={};p=0
    while p<len(payload):
        assert payload[p:p+4]==b'PK\x01\x02'
        method=struct.unpack_from('<H',payload,p+10)[0]
        crc,comp,raw=struct.unpack_from('<III',payload,p+16)
        namesize,extra,comment=struct.unpack_from('<HHH',payload,p+28)
        offset=struct.unpack_from('<I',payload,p+42)[0]
        name=payload[p+46:p+46+namesize].decode()
        if not name.endswith('/'):
            if name in entries or not re.fullmatch(r'[\w./+-]+',name) or name.startswith('/') or '..' in Path(name).parts:raise ValueError(name)
            entries[name]=(offset,comp,raw,method,crc)
        p+=46+namesize+extra+comment
    assert p==byte_len
    return entries
def ranges(volume,entries,names):
    sorted_all=sorted(entries,key=lambda k:entries[k][0]);groups=[];ids={k:i for i,k in enumerate(sorted_all)}
    wanted=sorted(set(names),key=ids.__getitem__);out=BASE/'private_expansion'/volume
    for k in wanted:
        p=out/k;idx=ids[k]
        if p.is_file() and zlib.crc32(p.read_bytes())&0xffffffff==entries[k][4]:continue
        if not groups or groups[-1][-1][1]+1!=idx or entries[k][0]-entries[groups[-1][0][0]][0]>2_000_000:groups.append([])
        groups[-1].append((k,idx))
    def materialize(group):
        names=[k for k,_ in group];first=entries[names[0]][0];last,comp,*_=entries[names[-1]];hdr=span(volume,last,last+29)
        assert hdr[:4]==b'PK\x03\x04';fn,extra=struct.unpack_from('<HH',hdr,26)
        blob=span(volume,first,last+30+fn+extra+comp-1)
        for name in names:
            off,comp,raw,method,crc=entries[name];pos=off-first;assert blob[pos:pos+4]==b'PK\x03\x04'
            n,e=struct.unpack_from('<HH',blob,pos+26)
            assert blob[pos+30:pos+30+n].decode()==name
            b=blob[pos+30+n+e:pos+30+n+e+comp]
            if method==zipfile.ZIP_DEFLATED:b=zlib.decompress(b,-15)
            elif method!=zipfile.ZIP_STORED:raise ValueError(method)
            assert len(b)==raw and zlib.crc32(b)&0xffffffff==crc
            p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:list(pool.map(materialize,groups))
    return {k:out/k for k in wanted}
if __name__=='__main__':
    for v in VOLUMES:
        d=directory(v)
        print(v,len(d),hashlib.sha256((BASE/('full_'+v+'_central.bin')).read_bytes()).hexdigest(),[n for n in d if n.endswith('data_info.json')],list(d)[:2],flush=True)
