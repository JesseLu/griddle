import numpy as np
import unittest
import __init__
from grid import Grid

class TestGrid(unittest.TestCase):
    """ Test the Grid class. """

    def setUp(self):
        """ Create grids of various sizes and data types to test on. """
        shapes = [(10000,), (100000,), (1000000,), (10000000,), \
            (100,100), (1000,1000), \
            (10,20,30), (40,50,60), (100,100,100)]
        shapes = [(10000,), (100,100,100)]
        dtypes = [np.float32, np.float64, np.complex64, np.complex128] 

        self.a = [] # Numpy arrays.
        self.d = [] # Grids.
        for dtype in dtypes:
            for shape in shapes:
                test_data = np.random.randn(*shape).astype(dtype)
                self.a.append(test_data)
                self.d.append(Grid(test_data))

    def test_init(self):
        """ Test the Grid class constructor. """
        # Make sure we reject bad input in the constructor.
        self.assertRaises(TypeError, Grid, 'abc')

    def test_get(self):
        """ Test the get function. """
        # Make sure we can pass in and retreive our data.
        for k in range(len(self.a)):
            self.assertTrue((self.a[k] == self.d[k].get()).all(), \
                'Data retrieval inconsistent.')

    def test_dup(self):
        """ Test the duplication function. """
        for d in self.d:
            dup_grid = d.dup()
            self.assertTrue((dup_grid.get() == d.get()).all(), \
                'Data duplication inconsistent.')

#     def test_dot(self):
#         """ Test the dot product function. """
#         for k in range(len(self.a)):
#             gres = self.d[k].dot(self.d[k]) 
#             cres = np.dot(self.a[k].flatten(), self.a[k].flatten())
#             rel_err = abs(gres - cres) / abs(cres)
#             self.assertTrue(rel_err < 1e-10)
# 
#     def test_aby(self):
#         for k in range(len(self.a)):
#             # d = self.d[k].dup()
#             print self.d[k].get().flatten()
#             self.d[k].aby(1, 2, self.d[k]) 
#             cres = 1 * self.a[k] + 2 * self.a[k]
#             rel_err = np.linalg.norm(self.d[k].get().flatten() - cres.flatten()) \
#                         / np.linalg.norm(cres.flatten())
#             print self.d[k].get().flatten(), rel_err
#             self.assertTrue(rel_err < 1e-10)
# 


if __name__ == '__main__':
    unittest.main()

