"""Check held-out splits, checkpoint reuse, and equal-site/equal-seed reduction."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from scripts import run_heldout_seed_sensitivity as runner


class TestFiveSeeds(unittest.TestCase):
    def test_transporting_a_clip_across_sites_or_reusing_an_index_fails(self):
        index = {"a": np.array([0]), "b": np.array([1])}
        with self.assertRaisesRegex(ValueError, "clip"):
            runner.split_indices(index, [("same", 1, 1), ("same", 2, 6)], "a")
        index["b"] = np.array([0])
        with self.assertRaisesRegex(ValueError, "partition"):
            runner.split_indices(index, [("one", 1, 1), ("two", 2, 6)], "a")

    def test_resuming_validates_id_seed_weights_and_errors(self):
        state = {"weight": torch.tensor([1.0])}
        errors = torch.tensor([[1.0, 2.0]])
        keys = [["clip", 5, 10]]
        identity = {"seed": 1024}
        cp = dict(identity=identity, window_keys=keys, model_state=state, errors=errors,
                  payload_sha256=runner.payload_digest(state, errors))
        np.testing.assert_array_equal(runner.validate_checkpoint(cp, identity, keys), errors.numpy())
        for replacement in (dict(identity={"seed": 3407}), dict(window_keys=[["clip", 5, 11]]),
                            dict(model_state={"weight": state["weight"] + 1}), dict(errors=errors + 1)):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                runner.validate_checkpoint({**cp, **replacement}, identity, keys)

    def test_all_fits_resume_and_pack_with_site_and_seed_macro(self):
        sites = list(runner.SITE_COUNTS)
        sizes = np.arange(1, len(sites) + 1)
        keys = [(site, j, 10) for site, size in zip(sites, sizes) for j in range(size)]
        index = {s: np.array([i for i, key in enumerate(keys) if key[0] == s]) for s in sites}
        pilot = {"identity": {"torch_version": str(torch.__version__)}, "window_key_fields": [],
                 "window_protocol": "synthetic", "selection": "synthetic", "metric_definitions": "synthetic"}
        def values(site, model, seed):
            mean = sites.index(site) + (seed / 1000 if model == "M1" else 0)
            return np.full((len(index[site]), 2), mean)
        baseline = {s: {m: values(s, m, 42) for m in runner.MODELS} for s in sites}
        fitted = []
        def fake_fit(model, tensors, train, test, device, protocol):
            site = keys[test[0]][0]
            self.assertEqual(set(train), set(range(len(keys))) - set(test))
            fitted.append((protocol.seed, site, model))
            return {"weight": torch.tensor([protocol.seed])}, values(site, model, protocol.seed)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "pose.npz").write_bytes(b"synthetic pose")
            originals = {"report.json": tmp / "seed42_report.json"}
            originals["report.json"].write_text("{}")
            for site in sites:
                for name in ("fold.json", "M0.pt", "M1.pt"):
                    path = tmp / "original" / site / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"synthetic pilot")
                    originals[f"{site}/{name}"] = path
            args = SimpleNamespace(output_dir=tmp / "out", pilot_dir=tmp, device="cuda", summarize_only=False)
            params = (args, {}, index, keys, pilot, baseline, originals)
            with redirect_stdout(io.StringIO()), patch.object(runner, "configure_training"), patch.object(
                    runner, "fit_predict", side_effect=fake_fit), patch.object(
                    torch.cuda, "get_device_name", return_value="synthetic"):
                runner.run_locked(*params)
                self.assertEqual(len(set(fitted)), 32)
                self.assertNotIn(42, [seed for seed, _, _ in fitted])
                lost = args.output_dir / "seed_3407" / sites[-1] / "M1.pt"
                lost.unlink()
                args.summarize_only = True
                with self.assertRaisesRegex(FileNotFoundError, "Missing seed_3407"):
                    runner.run_locked(*params)
                args.summarize_only = False
                runner.run_locked(*params)
                self.assertEqual(len(fitted), 33)
                self.assertEqual(fitted[-1], (3407, sites[-1], "M1"))
                args.summarize_only = True
                runner.run_locked(*params)
                self.assertEqual(len(fitted), 33)
            with zipfile.ZipFile(args.output_dir / "shortpose_heldout_five_seeds.zip") as archive:
                hashes = json.loads(archive.read("sha256.json"))
                self.assertEqual(set(hashes), set(archive.namelist()) - {"sha256.json"})
                self.assertEqual(sum(name.endswith(".pt") for name in hashes), 40)
                for name, digest in hashes.items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
                report = json.loads(archive.read("report.json"))
                self.assertAlmostEqual(report["four_site_macro_by_seed"]["42"]["M0"]["ade_m"], 1.5)
                mean_sd = report["five_seed_site_macro"][runner.PAIR]["fde_m"]
                self.assertAlmostEqual(mean_sd["mean"], np.mean(runner.FIXED_SEEDS) / 1000)
                self.assertAlmostEqual(mean_sd["sample_sd"], np.std(runner.FIXED_SEEDS, ddof=1) / 1000)


if __name__ == "__main__":
    unittest.main()
