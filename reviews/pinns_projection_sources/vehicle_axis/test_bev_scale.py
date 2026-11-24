import unittest
import numpy as np
from check_bev_scale import camera_orientation, declared_metric, geometry, reflect_y


class Tests(unittest.TestCase):
    def setUp(self):
        self.c=dict(x_min=1.,x_max=21.,y_min=-4.,y_max=6.,bev_width=200,bev_height=100,u_axis="x",v_axis="-y",
                    pts_2d=[[0.,0.],[200.,0.],[200.,100.],[0.,100.]],pts_3d=[[1.,6.],[21.,6.],[21.,-4.],[1.,-4.]])
    def test_axes_and_continuous_corners(self):
        at=np.array([[0.,0.],[200.,100.],[120.,70.]])
        q=declared_metric(at,self.c)
        np.testing.assert_allclose(q,[[1,6],[21,-4],[13,-1]])
        np.testing.assert_allclose(reflect_y(q,self.c),[[1,-4],[21,6],[13,3]])
        self.assertLess(camera_orientation(self.c),0.)
        exchanged=dict(self.c,u_axis="-y",v_axis="x")
        np.testing.assert_allclose(declared_metric(at,exchanged),[[1,6],[21,-4],[15,0]])
        np.testing.assert_allclose(declared_metric(np.array([[199.,99.]]),self.c,True),[[21,-4]])
    def test_bad_geometry(self):
        for change in (dict(x_max=1),dict(y_min=6),dict(bev_width=1),dict(bev_height=99.5),
                       dict(u_axis="z"),dict(v_axis="x"),dict(y_max=np.nan)):
            with self.assertRaises(ValueError):geometry(dict(self.c,**change))

if __name__=="__main__":unittest.main()
