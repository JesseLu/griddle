What does Griddle do?
=====================
Griddle makes it easy to write extremely fast 3D finite=element applications 
  for multi=GPU environments. 


How does Griddle work?
======================
Griddle provides a simple interface for manipulating gridded 3D data
  that is spread across multiple GPUs.
This interface consists primarily of distributed memory (DistGrid)
  and execution (DistKern) objects
  which are used to define the application in a simple, abstract way.
Griddle then takes care of all the communication and synchronization details
    and also optimizes runtime parameters for fast execution.


What is Griddle built on?
=========================
Griddle is heavily dependent on PyCUDA.


Interface overview
==================

Space
-----
Spaces form the context for colocating grids and kernels.
For example, creating two grids on the same space tells Griddle that
  these two grids should be overlaid on top of each other.
In the same way, defining a kernel on the space defines which grid elements
  will be updated.

Grid
----
Grids represent three-dimensional fields. 
To efficiently operate on Grids, every element in a Grid has limited visibility.
This means that when operating on a Grid (with a Kernel),
  only the certain adjacent neighboring elements may be accessed.
Specifically, only elements within a cube of length 2n+1 
  (where n is specified by the user) may be accessed.

Const
-----
Consts are global constant arrays that be accessed by any element of any grid.
However, the values of Const elements may not be changed,
  since they are not synchronized across processors.

Kernel
------
Kernels perform operations on Grids and 
  accept Grids, Consts, and constant scalars as input parameters.
Kernels perform both update and sum functions.
Multiple Kernels cannot be run in parallel.
