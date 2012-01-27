import numpy as np
import unittest
import __init__
from grid import Grid 
from kernel import Kernel
from jinja2 import Template

class TestKernel(unittest.TestCase):
    """ Test the DistKern class. """

    def setUp(self):
        """ Create grids of various sizes and data types to test on. """
        shapes = [(10,20,30), (40,50,60), (100,100,100)]
        # shapes = [(10000,), (100,100,100)]
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
                self.d.append(Grid(test_data))

    def test_mult2(self):
        """ Implement a simple 'multiply by 2' kernel. """
        for k in range(len(self.a)):
            code = Template('x(0,0,0) = {{ cuda_type }}(2) * x(0,0,0);')
            rendered_code = code.render(cuda_type=self.dtypes[k][1])
            fun = Kernel(self.shapes[k], rendered_code, (self.dtypes[k][1], 'x'))
            fun(self.d[k])
            err = np.linalg.norm((self.d[k].get() - \
                                (self.a[k] + self.a[k])).flatten())
            self.assertTrue(err == 0)

#     def test_sum(self):
#         """ Test a simple reduction kernel -- sum. """
#         for k in range(len(self.a)):

#     def test_1(self):
#         self.assertEqual(2,1)

if __name__ == '__main__':
    unittest.main()
