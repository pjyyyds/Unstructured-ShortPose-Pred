"""Read the GitHub release API and ZIP central directory only."""
import hashlib
import io
import json
from pathlib import Path
import struct
import urllib.request
import zipfile

BASE='https://api.github.com/repos/vita-epfl/social-transmotion'
HEAD={'User-Agent':'official-social-transmotion-feasibility'}
release=json.load(urllib.request.urlopen(urllib.request.Request(BASE+'/releases/tags/ckpt_data',headers=HEAD),timeout=30))
asset,=release['assets']
size=asset['size']
def window(start,stop):
    req=urllib.request.Request(asset['browser_download_url'],headers={**HEAD,'Range':f'bytes={start}-{stop}'})
    with urllib.request.urlopen(req,timeout=30) as response:
        assert response.headers.get('Content-Range')==f'bytes {start}-{stop}/{size}'
        data=response.read()
    assert len(data)==stop-start+1
    return data
tail=window(size-22,size-1)
assert tail.startswith(b'PK\x05\x06') and tail[-2:]==bytes(2)
length,start=struct.unpack_from('<II',tail,12)
directory=window(start,start+length-1)
with zipfile.ZipFile(io.BytesIO(directory+tail)) as z:
    members=[x for x in z.infolist() if not x.is_dir() and '__MACOSX/' not in x.filename and '/.' not in x.filename]
summary={'tag':release['tag_name'],'asset':asset['name'],'asset_url':asset['browser_download_url'],
         'asset_bytes':size,'published_at':release['published_at'],'directory_sha256':hashlib.sha256(directory).hexdigest(),
         'checkpoints':{x.filename:{'bytes':x.file_size,'crc32':f'{x.CRC:08x}'} for x in members if '.pth' in x.filename},
         'license_or_readme_names':[x.filename for x in members if any(k in x.filename.lower() for k in ('license','licence','copying','readme'))],
         'split_files':{},'bytes_downloaded':length+22,'data_members_opened':0}
for dataset in ('jta','jrdb'):
    summary['split_files'][dataset]={}
    for split in ('train','val','test'):
        names=sorted(x.filename for x in members if x.filename.startswith(f'{dataset}/data/{split}/') and x.filename.endswith('.ndjson'))
        summary['split_files'][dataset][split]=names if dataset=='jrdb' else len(names)
Path(__file__).with_name('release_inventory.json').write_text(json.dumps(summary,indent=2)+'\n')
print(summary['checkpoints'],{k:{s:len(n) if isinstance(n,list) else n for s,n in v.items()} for k,v in summary['split_files'].items()})
