"""The five-observation PINNS graph may use at most four elapsed 10 Hz steps."""
import unittest

import numpy as np

from src.data.pinns_extractor import _assemble_sequence_windows


def at(x, person_id):
    return {"type": "Person", "id": person_id, "wx": x, "wy": 1.0, "ox": x,
            "oy": 1.0, "w": 2.0, "h": 3.0, "fixed": -1}


def graph_sample(previous_tick):
    # Ego has full history and future coverage. The neighbor returns at t0=5.
    frames = {t: [at(t * .1, 1)] for t in range(40)}
    for t in (previous_tick, 5):
        frames[t].append(at(2.5, 2))
    samples = _assemble_sequence_windows("synthetic", frames, {}, [], {5: "test"}, 100., 1)
    assert len(samples) == 1
    return samples[0]


class GraphHistoryBoundary(unittest.TestCase):
    def test_fifth_predecessor_is_outside_observation(self):
        sample = graph_sample(0)
        self.assertEqual(sample["person_id"], 1)
        self.assertEqual(int(sample["ego_type_oh"].argmax()), 0)
        self.assertEqual(sample["nbr_mask"].sum(), 1)
        self.assertEqual(int(sample["nbr_type_oh"][0].argmax()), 0)
        edge = sample["nbr_edges"][0]
        np.testing.assert_allclose(edge, [2., 0., 0., 0., 2., 0., 0., .2, 1.], atol=1e-6)

    def test_fourth_predecessor_can_measure_velocity(self):
        sample = graph_sample(1)
        self.assertEqual(sample["nbr_mask"].sum(), 1)
        self.assertEqual(int(sample["nbr_type_oh"][0].argmax()), 2)
        edge = sample["nbr_edges"][0]
        np.testing.assert_allclose(edge, [2., 0., -1., 0., 2., 1., 1., .4, 1.], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
