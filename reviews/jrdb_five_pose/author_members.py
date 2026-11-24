from pathlib import Path
import hashlib,sys,time,urllib.error,urllib.request
def digest(payload):return hashlib.sha256(payload).hexdigest()

def request(url,headers,span=None):
    last=None
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'JRDB-two-frame-release-research',**headers})
            with urllib.request.urlopen(req,timeout=60) as r:
                if span and (r.status!=206 or r.headers.get('Content-Range')!=span):
                    raise ValueError('Author archive did not return the requested exact range')
                return r.read()
        except (TimeoutError,urllib.error.URLError,ValueError) as exc:
            last=exc
            if attempt<2:time.sleep(1+attempt)
    raise RuntimeError('Failed to stage fixed public source: '+str(last)) from last

def get_members(location,helpers):
    sys.path.insert(0,str(helpers))
    import evaluate_outdoor_motion as ref
    # This only opens the nine pinned training members. Any full-response fallback is rejected.
    class SourceZip(ref.TrainingArchive):
        def read(self,start,stop):
            sep='&' if '?' in self.url else '?'
            url=self.url+sep+'jrdb_byte_range='+str(start)+'-'+str(stop)
            raw=request(url,{'Range':f'bytes={start}-{stop}','Accept-Encoding':'identity'},
                        f'bytes {start}-{stop}/{self.size}')
            if len(raw)!=stop-start+1:raise ValueError('Truncated author archive range')
            return raw
    # A mounted authorized extraction can be used if all nine member hashes match.
    mount=Path('/kaggle/input')
    for candidate in sorted(mount.rglob(ref.MEMBERS['poses'].format(ref.SCENES[0]))):
        root=candidate.parents[2]
        if all((root/p.format(s)).is_file() and digest((root/p.format(s)).read_bytes())==ref.DIGESTS[g][i]
               for g,p in ref.MEMBERS.items() for i,s in enumerate(ref.SCENES)):
            return root
    for group,address in ref.ARCHIVES.items():
        archive=SourceZip(*address)
        for i,scene in enumerate(ref.SCENES):
            member=ref.MEMBERS[group].format(scene)
            raw=archive.member(member)
            if digest(raw)!=ref.DIGESTS[group][i]:raise ValueError('Incorrect training member '+member)
            path=location/member;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        print('Verified three author members',group,flush=True)
    return location

