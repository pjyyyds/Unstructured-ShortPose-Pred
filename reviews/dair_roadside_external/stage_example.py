"""Read only named members from an author-published ZIP using exact byte ranges and CRC."""
import concurrent.futures,io,json,re,struct,sys,time,urllib.request,zipfile,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXAMPLE='https://drive.usercontent.google.com/download?id=1gjOmGEBMcipvDzu2zOrO9ex_OscUZMYY&confirm=t'
SIZE=585329481
def span(a,b):
 for attempt in range(4):
  try:
   req=urllib.request.Request(EXAMPLE,headers={'Range':f'bytes={a}-{b}','User-Agent':'Roadside-pose-coverage/1','Accept-Encoding':'identity'})
   with urllib.request.urlopen(req,timeout=24) as f:
    if f.status!=206 or f.headers.get('Content-Range')!=f'bytes {a}-{b}/{SIZE}':raise ValueError('Wrong Range '+str(f.headers.get('Content-Range')))
    data=f.read(b-a+2)
   if len(data)!=b-a+1:raise ValueError('Truncated range')
   return data
  except (OSError,ValueError):
   if attempt==3:raise
   time.sleep(1+attempt)
def catalog():
 out=ROOT/'source_directory.json'
 if out.exists():return json.loads(out.read_text())
 tail=span(SIZE-22,SIZE-1)
 assert tail[:4]==b'PK\x05\x06' and tail[-2:]==b'\0\0'
 length,start=struct.unpack_from('<II',tail,12)
 directory=span(start,start+length-1)
 with zipfile.ZipFile(io.BytesIO(directory+tail)) as z:
  values={i.filename:[i.header_offset+start,i.compress_size,i.file_size,i.compress_type,i.CRC] for i in z.infolist() if not i.is_dir()}
 out.write_text(json.dumps(values,separators=(',',':')))
 return values
def save(item):
 name,(offset,compressed,size,method,crc)=item
 if not re.fullmatch(r'[\w./-]+',name) or '..' in Path(name).parts or name.startswith('/'):raise ValueError('Invalid path')
 path=ROOT/'private_source'/name
 if path.is_file() and (zlib.crc32(path.read_bytes())&0xffffffff)==crc:return path
 header=span(offset,offset+29)
 if header[:4]!=b'PK\x03\x04':raise ValueError(f'Invalid header {name}')
 fn,extra=struct.unpack_from('<HH',header,26);start=offset+30+fn+extra
 payload=span(start,start+compressed-1)
 raw=payload if method==zipfile.ZIP_STORED else zlib.decompress(payload,-15) if method==zipfile.ZIP_DEFLATED else None
 if raw is None or len(raw)!=size or zlib.crc32(raw)&0xffffffff!=crc:raise ValueError('Integrity '+name)
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);return path
def stage(names):
 directory=catalog()
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
  return list(pool.map(save,[(n,directory[n]) for n in names]))
if __name__=='__main__':
 d=catalog();print('Directory members',len(d));print('\n'.join([k for k in d if k.endswith('data_info.json')]))
 print('Historical image members in example',sum('/infrastructure-side/image/' in k for k in d))

def stage_ranges(names):
 """Coalesce consecutive requested local members; do not fetch an intervening unrequested file payload."""
 items=sorted(catalog().items(),key=lambda x:x[1][0]);want=set(names);runs=[]
 for i,(name,params) in enumerate(items):
  if name not in want:continue
  if not runs or i!=runs[-1][-1]+1 or params[0]-items[runs[-1][0]][1][0]>2_000_000:runs.append([])
  runs[-1].append(i)
 def extract(group):
  first=items[group[0]][1][0];idx=group[-1];key_last,parm_last=items[idx];offset,size,*_=parm_last
  hdr=span(offset,offset+29);fn,extra=struct.unpack_from('<HH',hdr,26);end=offset+30+fn+extra+size
  block=span(first,end-1);out=[]
  for i in group:
   name,(off,comp,rawsize,meth,crc)=items[i];at=off-first;assert block[at:at+4]==b'PK\x03\x04'
   n,e=struct.unpack_from('<HH',block,at+26);assert block[at+30:at+30+n].decode()==name
   data=block[at+30+n+e:at+30+n+e+comp];value=data if meth==0 else zlib.decompress(data,-15)
   assert len(value)==rawsize and zlib.crc32(value)&0xffffffff==crc
   path=ROOT/'private_source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value);out.append(path)
  return out
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:return [k for g in executor.map(extract,runs) for k in g]
