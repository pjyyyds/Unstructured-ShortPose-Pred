"""Targeted fail-safe checks; never score real external motion or images."""
import ast,base64,hashlib,io,json,tempfile,unittest,zipfile,zlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from detector_pilot import associate,axis_tiles,dedup
from fusion_cloud import D,load_pose_cache
def sha(b):return hashlib.sha256(b).hexdigest()
class Pipeline(unittest.TestCase):
    def test_embedded_sources(self):
        for n in ('kaggle_dut_shared_space_motion.py','kaggle_dut_detector_pilot.py','kaggle_dut_pose_cache.py','kaggle_dut_shared_space_fusion.py'):
            tree=ast.parse((D/n).read_text());ss=ast.literal_eval(next(x.value for x in tree.body if isinstance(x,ast.Assign) and getattr(x.targets[0],'id','')=='SOURCES'))
            for file,(digest,packed) in ss.items():
                b=zlib.decompress(base64.b85decode(packed));self.assertEqual(sha(b),digest);self.assertEqual(b,(D/file).read_bytes())
    def test_same_worker_or_mounted_cache_and_reuse(self):
        tree=ast.parse((D/'kaggle_dut_shared_space_fusion.py').read_text());r=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run')
        funcs=[x for x in r.body if isinstance(x,ast.FunctionDef)];self.assertEqual([x.name for x in funcs],['find_cache','verify'])
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp);mnt=t/'input';wrk=t/'working';mnt.mkdir();wrk.mkdir();env={'sha':sha,'root':mnt,'out':wrk/'fusion.zip','zipfile':zipfile,'json':json,'SOURCES':{'fusion_cloud.py':('source',None),'motion_and_detector_freeze.json':('protocol',None)}}
            exec(compile(ast.Module(body=funcs,type_ignores=[]),'<actual generated cell>', 'exec'),env);find=env['find_cache']
            with self.assertRaises(FileNotFoundError):find()
            cache=wrk/'dut_shared_space_pose_cache.zip';cache.write_bytes(b'first');self.assertEqual(cache,find())
            other=mnt/cache.name;other.write_bytes(b'first');self.assertEqual(cache,find())
            other.write_bytes(b'second')
            with self.assertRaises(ValueError):find()
            other.unlink()
            rep={'pose_cache_sha256':sha(b'first'),'scorer_sha256':'source','protocol_sha256':'protocol','by_axis_and_availability':{'0':{'all':{'M0':{'windows':3066}}}}}
            b=json.dumps(rep).encode()
            with zipfile.ZipFile(env['out'],'w') as z:z.writestr('aggregate.json',b);z.writestr('sha256.json',json.dumps({'aggregate.json':sha(b)}))
            env['verify']();cache.write_bytes(b'changed')
            with self.assertRaises(AssertionError):env['verify']()
    def test_ambiguity_and_tiles(self):
        a=dedup([np.array([0,0,8,8,.9]),np.array([0,0,8,8,.6]),np.array([6,6,12,12,.7])]);self.assertEqual(len(a),2)
        self.assertEqual(associate(a,{1:(2,2),2:(10,10),3:(15,15)})[3],('missing',-1))
        self.assertEqual(associate(a,{1:(7,7),2:(8,8)})[1],('multiple_boxes_and_people',-1));self.assertEqual(associate(a[:1],{1:(1,1),2:(2,2)})[1],('multiple_people_in_box',-1))
        for size in (100,384,385,1058,1530):
            hit=np.zeros(size,bool)
            for x in axis_tiles(size):hit[x:x+384]=True
            self.assertTrue(hit.all())
    def test_pose_cache_identity_and_missing_observations(self):
        k=[('testclip',1,5)];fr=np.arange(1,36).reshape(1,35);mask=np.array([[True,False,False,False,False]])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cache.zip'
            def make(variant):
                bb=np.zeros((1,5,4),np.float32);bb[0,0]=[21,30,8,10];j=np.zeros((1,5,17,3),np.float32);j[0,0,0]=[2,3,.6]
                if variant=='invalid_missing':j[0,2,0,2]=.5
                tags=[['testclip',1,f,'unique' if f==1 else 'missing'] for f in range(1,6)]
                if variant=='wrong_status':tags[1][-1]='unique'
                f=fr[:,:5].copy()
                if variant=='wrong_frame':f[0,-1]=6
                b=io.BytesIO();np.savez_compressed(b,keys_json=np.array(json.dumps(k)),source_frame_1based=f,kpts=j,bbox=bb,matched=mask,association_json=np.array(json.dumps(tags)))
                src=(D/'detector_pilot.py').read_bytes();r={'windows':1,'mode':'all_windows' if variant!='pilot' else 'prespecified_first_window_each_clip','script_sha256':sha(src),'full_cohort_keys_sha256':'keys','score_protocol_sha256':'protocol','matched_frames_per_window_0_to_5':[0,1,0,0,0,0],'matched_observations':1,'requested_historical_person_observations':5}
                members={'aggregate.json':json.dumps(r).encode(),'private.npz':b.getvalue(),'source.py':src}
                with zipfile.ZipFile(p,'w') as z:
                    for name,data in members.items():z.writestr(name,data)
                    z.writestr('sha256.json',json.dumps({n:sha(v) for n,v in members.items()}))
            make('valid');_,j,b,actual=load_pose_cache(p,k,fr,'keys','protocol');np.testing.assert_array_equal(actual,mask)
            for variant in ('invalid_missing','wrong_status','wrong_frame','pilot'):
                make(variant)
                with self.assertRaises((AssertionError,ValueError)):load_pose_cache(p,k,fr,'keys','protocol')
    def test_m1_fusion_fallback_on_synthetic_inputs(self):
        import torch
        from src.data.features import build_pose_and_quality_features
        from src.models.predictor import UnstructuredShortPosePredictor
        torch.manual_seed(901);model=UnstructuredShortPosePredictor('M1').eval();x=torch.zeros(2,5,9);x[:,:,5]=1.;x[:,:,6]=1.;x[:,2,0]=.25
        j=np.zeros((5,17,3),np.float32);b=np.zeros((5,4),np.float32);xp,q,_,_=build_pose_and_quality_features(j,b,np.ones(5),600.);self.assertEqual(q.tolist(),[0.,0.,0.,1.])
        hm=model.motion_enc(x);hp=torch.randn_like(hm);qq=torch.as_tensor(q).repeat(2,1)
        mixed,g=model.quality_gate(hm,hp,qq);plain,gplain=model.quality_gate(hm)
        torch.testing.assert_close(g,torch.zeros_like(g),atol=0,rtol=0);torch.testing.assert_close(mixed,model.quality_gate.out_norm(hm),atol=0,rtol=0);torch.testing.assert_close(mixed,plain,atol=0,rtol=0)
        qq[0,:]=torch.tensor([.25,.8,.4,1.]);_,gate=model.quality_gate(hm,hp,qq);self.assertGreater(gate[0,0].item(),0);self.assertEqual(gate[1,0].item(),0)
if __name__=='__main__':unittest.main()
