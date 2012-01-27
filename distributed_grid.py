""" Defines the DistGrid class, which stores the data in the grid. """

from pycuda import gpuarray as ga
from pycuda import driver
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
import numpy as np

# Custom kernel implementing x = a*x + b*y.
_axby = ElementwiseKernel(
            """ pycuda::complex<double> a, pycuda::complex<double> *x,
                pycuda::complex<double> b, pycuda::complex<double> *y""", \
           ' x[i] = a * x[i] + b * y[i]')

# Custom kernel implementing norm calculation.
_norm = ReductionKernel(np.complex128, neutral="0",
        reduce_expr="a+b", map_expr="pow(abs(x[i]), 2)",
                arguments="pycuda::complex<double> *x")


class DistGrid:
    """ Manages pycuda GPUArrays across multiple GPUs. """

    def __init__(self, array=None):
        """ Initialize a DistGrid.

        DistGrid() creates an empty grid.
        DistGrid(array) creates a grid filled with the values of array,
            where array is a numpy array.
        """
        if array is not None:
            if type(array) is np.ndarray:
                self.g = ga.to_gpu(array) # Copy data to the GPU.
            else:
                raise TypeError('Input must be of type numpy.ndarray.')
                    

    def get(self):
        """ Return the data as a numpy ndarray. """
        return self.g.get()

    def dup(self):
        """ Create a duplicate grid and return it. """
        dup_grid = DistGrid()
        dup_grid.g = ga.empty_like(self.g)
        driver.memcpy_dtod(dup_grid.g.gpudata, self.g.gpudata, self.g.nbytes)
        return dup_grid
        
    def dot(self, y):
        """ Return the dot product of the grid with y (another grid). """
        # return ga.dot(self.g, y.g).get()
        return np.dot(self.g.get().flatten(), y.g.get().flatten())

    def aby(self, a, b, y):
        """ Perform x = a*x + b*y. """
        _axby(np.array(a).astype(np.complex128), self.g, \
            np.array(b).astype(np.complex128), y.g)

    def norm(self):
        """ Calculate the norm over the elements of the grid."""
        # return np.sqrt(ga.sum(pow(abs(self.g),2)))
        # return np.sqrt(np.sum(np.abs(self.g.get())**2))
        return _norm(self.g).get()



