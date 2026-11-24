from stage_sources import D,BASE,CLIPS,author_tree
import concurrent.futures,hashlib,json,urllib.request
def main():
    tr=author_tree();names=['data/backgrounds/'+s+'_background_GSOC.png' for s in CLIPS]
    names+=['legacy/clips/backgrounds/roundabout_04_background_GSOC.png']
    names+=['legacy/clips/roi_clips/roundabout_04_roi_box.mp4','legacy/clips/stabilized/stabilized_roundabout_04.mp4','legacy/clips/trajectories_filtered/roundabout_04_traj_veh_filtered.csv']
    names+=['legacy/clips/trajectories_filtered/'+s+'_traj_ped_filtered.csv' for s in CLIPS]
    def fetch(n):
        pp=D/'private_reference'/n;pp.parent.mkdir(parents=True,exist_ok=True);b=pp.read_bytes() if pp.exists() else urllib.request.urlopen(BASE+n,timeout=55).read()
        assert len(b)==tr[n]['size'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==tr[n]['sha'];pp.write_bytes(b)
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as ex:list(ex.map(fetch,names))
    print('Verified',len(names),'author reference members')
if __name__=='__main__':main()
