"""Fetch only the four-sequence public author sample, using checked ranges; never fetch a model or test split."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,re,time
from pathlib import Path
from urllib.request import Request,urlopen

HERE=Path(__file__).resolve().parent
REC='https://zenodo.org/api/records/14811016'
name='imptc_samples.tar.gz'
def get_range(task):
    url,start,stop,total=task
    for retry in range(4):
        try:
            req=Request(f'{url}?bytes={start}-{stop-1}',headers={'Range':f'bytes={start}-{stop-1}','User-Agent':'IMPTC-academic-metadata-audit/1.0'})
            with urlopen(req,timeout=50) as response:
                assert response.status==206 and response.headers['Content-Range']==f'bytes {start}-{stop-1}/{total}'
                blob=response.read(stop-start+1)
            assert len(blob)==stop-start
            return blob
        except Exception:
            if retry==3: raise
            time.sleep(2**retry)
def main():
    with urlopen(REC,timeout=30) as response:
        record=json.load(response)
    spec=next(i for i in record['files'] if i['key']==name)
    total,url=spec['size'],spec['links']['self']
    assert spec['checksum']=='md5:d2bf9579b127b74927150bd3666a034f' and total==347692691
    output=HERE/name
    if output.exists() and hashlib.file_digest(output.open('rb'),'md5').hexdigest()==spec['checksum'][4:]:return
    part=output.with_suffix('.part')
    size=8*1024*1024
    with ThreadPoolExecutor(4) as pool,part.open('wb') as writer:
        for blob in pool.map(get_range,[(url,s,min(total,s+size),total) for s in range(0,total,size)]):
            writer.write(blob)
    assert part.stat().st_size==total and hashlib.file_digest(part.open('rb'),'md5').hexdigest()==spec['checksum'][4:]
    part.replace(output)
if __name__=='__main__':main()
