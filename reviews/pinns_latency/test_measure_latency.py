"""Synthetic guards and a synthetic CPU forward, with no device timing."""
import os
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import torch

import measure_latency as L


def fake(t):
    d = {k: np.full((24, *s), 0.25, dtype=np.float32) for k, s in L.shapes(t).items()}
    d["nbr_mask"][:] = 1
    d["m_map"][:] = 1
    d["x_m"][:, :, 6:] = 1
    d["ego_type_oh"][:] = 0
    d["ego_type_oh"][:, 0] = 1
    d["clip"] = np.repeat(["a", "b"], 12)
    d["site"] = np.repeat(["AA", "BB"], 12)
    d["t0"] = np.repeat([10, 15], 12)
    d["person_id"] = np.tile(np.arange(11, -1, -1), 2)
    return d


class Checks(unittest.TestCase):
    def test_real_hardware_gates(self):
        for name in ("Tesla T4", "NVIDIA GeForce RTX 4090 Laptop GPU", "NVIDIA GeForce RTX 4090 D"):
            self.assertIsNone(L.target(name, "x86_64", "", ""))
        self.assertEqual(L.target("NVIDIA GeForce RTX 4090", "x86_64", "", "")["strict_target_ms"], 10)
        self.assertIsNone(L.target("Orin", "aarch64", "Jetson Orin Nano", "nvidia,p3768-0000+p3767-0005"))
        self.assertIsNone(L.target("Orin", "aarch64", "Jetson AGX Orin", "nvidia,p3701-0000"))
        self.assertIsNone(L.target("Tesla T4", "aarch64", "Jetson Orin NX", ""))
        self.assertEqual(L.target("Orin", "aarch64", "", "nvidia,p3768-0000+p3767-0001")["strict_target_ms"], 25)
        with self.assertRaises(ValueError):
            L.cuda_device("cpu")

    def test_roster_and_temporal_keys(self):
        d = fake(5)
        g = L.validate_arrays(d, 5)
        self.assertEqual(d["person_id"][g["a", 10][:4]].tolist(), [0, 1, 2, 3])
        all_groups = {t: g for t in L.TIMES}
        cells = L.plan(all_groups, range(1, 13))
        self.assertEqual(len(set(cells)), 72)
        self.assertEqual(cells, L.plan(all_groups, range(1, 13)))
        for i in range(0, 72, 2):
            self.assertEqual(cells[i][:2], cells[i + 1][:2])
            self.assertNotEqual(cells[i][2], cells[i + 1][2])
        for d in [dict(fake(5), future=np.ones(2)), dict(fake(5), x_m=np.zeros((24, 5, 9))),
                  dict(fake(5), t0=np.arange(24)), dict(fake(5), person_id=np.ones(24, dtype=int))]:
            with self.assertRaises(ValueError):
                L.validate_arrays(d, 5)
        missing = fake(5)
        del missing["nbr_mask"]
        with self.assertRaises(ValueError):
            L.validate_arrays(missing, 5)

    def test_wall_threshold_and_data_checks(self):
        s = L.stats([1.0, 2.0, 3.0, 4.0, 5.0], {"strict_target_ms": 3})
        self.assertEqual(s["fraction_below_target"], .4)
        self.assertFalse(s["median_below_target"])
        self.assertAlmostEqual(s["p95_ms"], 4.8)
        self.assertNotIn("strict_target_ms", L.stats([.1, .2]))
        for values in [[], [0, .1], [.1, float("nan")]]:
            with self.assertRaises(ValueError):
                L.stats(values)

    def test_timer_sequences_and_repetitions(self):
        calls = []
        class Event:
            def __init__(self, **kwargs):
                self.name = str(sum(k == "event_new" for k in calls))
                calls.append("event_new")
                assert kwargs["enable_timing"]
            def record(self, stream):
                calls.append(self.name + "_record")
            def synchronize(self):
                calls.append(self.name + "_sync")
            def elapsed_time(self, other):
                calls.append("elapsed")
                return .012
        def run(_):
            calls.append("forward")
        with patch.object(L.torch.cuda, "Event", Event), patch.object(L.torch.cuda, "current_stream", return_value=object()), \
                patch.object(L.torch.cuda, "synchronize", side_effect=lambda _: calls.append("sync")), \
                patch.object(L.time, "perf_counter_ns", side_effect=[0, 1000, 4000, 7000]):
            values = L.timed(run, {}, "mock", 2, 2)
        self.assertEqual(calls, ["forward", "forward", "sync"] + 2 * ["sync", "forward", "sync"] +
                         ["event_new", "event_new", "0_record", "1_record", "1_sync"] +
                         2 * ["0_record", "forward", "1_record", "1_sync", "elapsed"])
        self.assertEqual(values, {"synchronized_host_wall": [.001, .003], "cuda_event_stream": [.012, .012]})

    def test_save_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.json"
            L.save_new(p, {"state": 1})
            with self.assertRaises(ValueError):
                L.save_new(p, {"state": 2})
            self.assertIn('"state": 1', p.read_text())

    @unittest.skipUnless(os.environ.get("SHORTPOSE_REPO"), "supply SHORTPOSE_REPO to verify real architecture with synthetic inputs")
    def test_frozen_model_on_synthetic_rows(self):
        torch.set_num_threads(1)
        model = L.load_model(Path(os.environ["SHORTPOSE_REPO"]))
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            for t, n in ((5, 1), (10, 2), (20, 4)):
                d = fake(t)
                batch = {k: torch.from_numpy(d[k][:n].copy()) for k in L.shapes(t)}
                tr = torch.jit.trace(model, (batch,), strict=False, check_trace=False)
                for changed in (False, True):
                    if changed:
                        batch["x_p"] = batch["x_p"] * .7
                    L.check_trace_outputs(model(batch), tr(batch), n)

    def test_check_only_path_never_forwards(self):
        import sys
        from contextlib import redirect_stdout
        from io import StringIO
        d = {t: fake(t) for t in L.TIMES}
        g = {t: L.validate_arrays(d[t], t) for t in L.TIMES}
        with tempfile.TemporaryDirectory() as folder:
            args = ["cmd", "--repo", folder, "--fixture-dir", folder, "--check-only", "--output", folder + "/x.json"]
            with patch.object(sys, "argv", args), patch.object(L, "load_fixture", return_value=({}, d, g)), \
                    patch.object(L, "load_model"), patch.object(L, "measure", side_effect=AssertionError("executed")), \
                    redirect_stdout(StringIO()):
                L.main()


if __name__ == "__main__":
    unittest.main()
