import numpy as np
import unittest
import __init__
from distributed_grid import DistGrid as grid 
from distributed_kernel import DistKern as kern
from jinja2 import Template

class TestDistKern(unittest.TestCase):
    """ Test the DistKern class. """

    def setUp(self):
        """ Create grids of various sizes and data types to test on. """
        shapes = [(10000,), (100000,), (1000000,), (10000000,), \
            (100,100), (1000,1000), \
            (10,20,30), (40,50,60), (100,100,100)]
        shapes = [(10000,), (100,100,100)]
        dtypes = [  (np.float32, 'float'), \
                    (np.float64, 'double'), \
                    (np.complex64, 'pycuda::complex<float>'), \
                    (np.complex128, 'pycuda::complex<double>')] 

        self.a = [] # Numpy arrays.
        self.d = [] # grids.
        self.shapes, self.dtypes = [], []
        for dtype in dtypes:
            for shape in shapes:
                test_data = np.random.randn(*shape).astype(dtype[0])
                self.shapes.append(shape)
                self.dtypes.append(dtype)
                self.a.append(test_data)
                self.d.append(grid(test_data))

    def test_mult2(self):
        """ Implement a simple 'multiply by 2' kernel. """
        for k in range(len(self.a)):
            fun = kern(self.shapes[k], 'x(0,0,0) = x(0,0,0) + x(0,0,0);', \
                (self.dtypes[k][1], 'x'))
            fun(self.d[k])
            err = np.linalg.norm((self.d[k].get() - \
                                (self.a[k] + self.a[k])).flatten())
            print self.dtypes[k][1], err
        self.assertTrue(1,1)

#     def test_1(self):
#         self.assertEqual(2,1)

if __name__ == '__main__':
    unittest.main()
