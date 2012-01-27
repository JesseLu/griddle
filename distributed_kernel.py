from pycuda import compiler
from jinja2 import Environment, PackageLoader
import numpy as np

# Execute when module is loaded.
# Load the jinja environment.
jinja_env = Environment(loader=PackageLoader(__name__, 'templates'))

class DistKern():
    def __init__(self, space, code, *params):
        """Return a cuda function that will execute on a grid.
        """

        # Initialize parameters.

        if len(space) < 3:
            space= ((1, 1, 1) + space)[-3:]
        self.shape = space # Size of the operation.
        self.block_shapes, self.grid_shapes = _get_shapes(self.shape)

        # Get the template and render it using jinja2.
        template = jinja_env.get_template('griddle_kernel.cu') 
        cuda_source = template.render(  params=params, \
                                        dims=self.shape, \
                                        loop_code=code, \
                                        flat_tag='_f')
        f = open('/tmp/code', 'w')
        f.write(cuda_source)

        # Compile the code into a callable cuda function.
        mod = compiler.SourceModule(cuda_source)
        self.fun = mod.get_function('griddle_kernel')


    def __call__(self, *grids):
        """ Execute the kernel. """
#         # TODO: allow keyword arguments to be passed to the pycuda function call.
#         # TODO: allow for default optimized block and grid shape function call.
        self.fun(*[grid.g.gpudata for grid in grids], \
                block=self.block_shapes[::-1], grid=self.grid_shapes[2:0:-1])

def _get_shapes(shape):
    """ Obtain the different thread block shapes to try. """
    block_shapes = (1, 20, 20)
    grid_shapes = (1, int(np.ceil(shape[1]/block_shapes[1]) + 1), \
                    int(np.ceil(shape[2]/block_shapes[2]) + 1))
    return block_shapes, grid_shapes

    
