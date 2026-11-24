"""Synthetic checks independent of dataset rows."""
import unittest

import numpy as np
from diagnose_projection import fit_h, project


class TestProjection(unittest.TestCase):
    def test_calibration_and_heldout(self):
        known=np.array([[.04,.03,-3],[.01,.09,12],[.00017,.00061,1.]])
        corners=np.array([[0.,0.],[1000.,50.],[900.,500.],[60.,600.]])
        train=np.random.default_rng(48).uniform([0.,0.],[1200.,950.],size=(80,2))
        held=np.random.default_rng(29).uniform([0.,0.],[1200.,950.],size=(48,2))
        for u in [corners,train]:
            v=project(u,known)
            inferred=fit_h(u,v)
            np.testing.assert_allclose(project(held,inferred),project(held,known),rtol=1e-12,atol=1e-12)
        with self.assertRaises(AssertionError):
            fit_h(np.arange(8.,dtype=float).reshape(4,2),corners)


if __name__=='__main__':unittest.main()
