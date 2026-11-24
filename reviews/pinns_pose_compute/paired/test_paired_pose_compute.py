"""Synthetic methods checks. No real person input is loaded."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

import paired_pose_compute as P


def scores(n=2):
    d = {k: np.zeros((n, *tail), np.float32) for k, tail in P.EXPECTED.items()}
    d["mode_probs"][:] = [.4, .2, .15, .12, .1, .03]
    return d


class TestAB(unittest.TestCase):
    def test_ranks_keep_near_far_and_lower_reversals(self):
        x = np.array([[.30001,.3,.2,.14,.04,.01999], [.4,.3,.15,.07,.05,.03], [.35,.35,.1,.1,.05,.05]])
        y = x.copy()
        y[:2, 0], y[:2, 1] = x[:2, 1], x[:2, 0]
        y[2, 4], y[2, 5] = .04999, .05001
        out=P.ranks(x,y)
        self.assertEqual(out["top1_changed"],2)
        self.assertEqual(out["original_near_top_ties"],2)
        self.assertEqual(out["top1_changed_within_near_gap"],1)
        self.assertEqual(out["full_order_changed"],3)
        self.assertEqual(out["inverted_pairs"],3)
        self.assertEqual(out["inverted_pairs_within_near_gap"],2)
        self.assertEqual(out["original_exact_top_ties"],1)
        self.assertEqual(P.ranks(x,x)["full_order_changed"],0)

    def test_mismatches_are_reported(self):
        x, y = scores(), scores()
        y["trajs"][0,0,0,0] = .0003
        y["act_probs"][0,0] = np.nan
        y["act_probs"][1,1] = .8
        y["mode_probs"][0, 0] += .02
        c=P.compare(x,y,2)
        self.assertFalse(c["all_values_within_bound"])
        self.assertEqual(c["by_field"]["trajs"]["above_tolerance"],1)
        self.assertEqual(c["by_field"]["act_probs"]["nonfinite_pairs"],1)
        self.assertEqual(c["actions_at_0_5"]["changed_by_class"],[0,1,0,0,0])
        self.assertIn("mode_probs: invalid normalization",c["checks"])
        y["mode_probs"]=y["mode_probs"][:1]
        self.assertNotIn("ranking",P.compare(x,y,2))
        del y["trajs"]
        self.assertIn("trajs: missing or unexpected output",P.compare(x,y,2)["checks"])
        exact = P.compare(x,x,2)
        self.assertTrue(exact["all_values_within_bound"])
        self.assertEqual(P.overall([{"agents":2,"comparison":exact}], "comparison")["fields"]["trajs"]["values"],720)

    def test_schedule_and_sync_order(self):
        seq=P.order_for("scene",20,12)
        self.assertEqual(seq.count("AB"),50)
        self.assertEqual(seq.count("BA"),50)
        self.assertEqual(seq,P.order_for("scene",20,12))
        self.assertNotEqual(seq,P.order_for("else",20,12))
        with self.assertRaises(ValueError): P.order_for("a",5,1,3)
        calls=[]
        tick=iter([0,8,10,17,20,24,30,36])
        sample=P.measure_pairs({k:(lambda _, arm=k: calls.append(arm)) for k in "AB"}, {},
                               ["AB","BA"],lambda: calls.append("sync"),lambda:next(tick))
        self.assertEqual(calls,["sync","A","sync","sync","B","sync","sync","B","sync","sync","A","sync"])
        self.assertEqual(sample,[{"order":"AB","A":.000008,"B":.000007},{"order":"BA","B":.000004,"A":.000006}])

    def test_primary_uses_only_two_target_cells_equally(self):
        cells=[]
        for t,n,scene,d in [(20,12,"east",2.),(20,12,"west",-3.),(5,12,"east",8.),(20,8,"west",9.)]:
            rows=[{"order":o,"A":10., "B":10.-d} for o in ("AB","BA")]
            cells.append({"t_obs":t,"agents":n,"scene":scene,"input_support":{"site":scene},"time":P.paired_summary(rows)})
        summary=P.primary(cells)
        self.assertEqual(summary["equal_scene_paired_saving_ms"],-.5)
        self.assertEqual(summary["equal_scene_paired_saving_percent"],-5.)
        with self.assertRaises(ValueError): P.primary(cells[:1])

    @unittest.skipUnless(os.environ.get("SHORTPOSE_REPO"),"provide the frozen model repo for synthetic-forward tests")
    def test_actual_transform_with_synthetic_inputs(self):
        root=Path(os.environ["SHORTPOSE_REPO"])
        baseline = os.environ.get("SHORTPOSE_LATENCY_RUNNER", str(root / "reviews/pinns_latency/measure_latency.py"))
        self.assertEqual(P.sha(Path(baseline).read_bytes()),P.RUNNER_SHA)
        self.assertEqual(P.sha((root/"src/models/pose_inference.py").read_bytes()),P.TRANSFORM_SHA)
        spec=importlib.util.spec_from_file_location("baseline",baseline)
        base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
        torch.set_num_threads(1)
        a=base.load_model(root)
        from src.models.pose_inference import prepare_pose_inference
        b=prepare_pose_inference(a)
        self.assertEqual(P.check_states(a,b)["archived_state"],161)
        rng=np.random.default_rng(19)
        with torch.inference_mode():
            for t,n in ((5,1),(10,4),(20,12)):
                d={k:torch.from_numpy(rng.normal(size=(n,*s)).astype(np.float32)) for k,s in base.shapes(t).items()}
                d["x_p"][:,:,:,3]=torch.from_numpy((rng.uniform(size=(n,t,17))>.3).astype(np.float32))
                d["q_vec"].sigmoid_();d["q_part"].sigmoid_()
                x,y=P.as_cpu(a(d)),P.as_cpu(b(d))
                outcome=P.compare(x,y,n)
                self.assertTrue(outcome["all_values_within_bound"],outcome)
                self.assertEqual(outcome["ranking"]["valid_rows"],n)
                self.assertEqual(outcome["by_field"]["trajs"]["values"],n*6*30*2)


if __name__=="__main__":
    unittest.main()
