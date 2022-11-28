###################################################
Use mpi4py-ve with homebrew classes (without NLCPy)
###################################################

*mpi4py-ve* allows objects with the *__ve_array_interface__* attribute to be specified as arguments to the communication API.

******************************
VE Array Interface (Version 1)
******************************
The *VE Array Interface* (or VAI) is created for interoperability between different implementations
of VE array-like objects in various projects. The idea is borrowed from the `NumPy array interface <https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface__>`_
and `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_.

------------------------------
Python Interface Specification
------------------------------

    Note

    Experimental feature. Specification may change.

The  ``__ve_array_interface__``  attribute returns a dictionary ( ``dict`` ) that must contain the
following entries:

* **shape**:  ``(integer, ...)``
  A tuple of ``int``  (or  ``long`` ) representing the size of each dimension.

* **typestr**:  ``str``
  The type string. This has the same definition as ``typestr`` in the `numpy array interface <https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface__>`_.

* **data**:  ``(integer, boolean)``
  The data is a 2-tuple. The first element is the data pointer to VEO HMEM (Heterogenious
  Memory) as a Python  ``int``  (or  ``long`` ). For zero-size arrays, use ``0``  here. The second element
  is the read-only flag as a Python  ``bool`` .

* **version**:  ``integer``
  An integer for the version of the interface being exported. The current version is *1*.

The followings are optional entries:

* **strides**: ``None`` or  ``(integer, ...)``
  If **strides** is not given, or it is  ``None`` , the array is in C-contiguous layout. Otherwise, a tuple
  of  ``int``  (or  ``long`` ) is explicitly given for representing the number of bytes to skip to access
  the next element at each dimension.

* **descr**:
  This is for describing more complicated types. This follows the same specification as in
  the `numpy array interface <https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface__>`_.

* **mask**: ``None`` or object exposing the ``__ve_array_interface__``
  If ``None`` then all values in **data** are valid. All elements of the mask array should be
  interpreted only as true or not true indicating which elements of this array are valid. This
  has the same definition as ``mask`` in the `numpy array interface <https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface__>`_.

      Note

      mpi4py-ve does not currently support working with masked VE arrays and will raise
      a exception if one is passed to a function.

* **veo_ctxt**: ``None`` or ``integer``
  The pointer of ``veo_thr_ctxt`` as a Python ``int`` (or ``long``).

*************************************************
Example code for mpi4py-ve using homebrew classes
*************************************************

-----------
source code
-----------
* mpi_send_recv.py: Main script to communicate between objects that have ``__ve_array_interface__`` attribute.

.. code-block:: python

  from mpi4pyve import MPI
  from mpi4pyve import util
  import numpy as np
  import veo_Py_wrapper
  
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()
  
  veo = veo_Py_wrapper.Veo(rank)            # create VE process
  x = np.array([123, 456, 789], dtype=int)  # create buffer on VH
  x_hmem = veo.alloc_hmem(x.dtype, x.size)  # create buffer on VE
  
  if rank == 0:
      x_hmem.set_value(x)  # set value into VE buffer
      comm.Send(x_hmem, dest=1)
      comm.Recv(x_hmem, source=1)
  elif rank == 1:
      comm.Recv(x_hmem, source=0)
      comm.Send(x_hmem, dest=0)
  comm.Barrier()
  
  res = np.all(x == x_hmem.get_value())  # result check
  print('Result {} (rank={})'.format('Success' if res else 'Failed', rank))
  
  del x_hmem

* veo_Py_wrapper.py: Sub script to call veo C APIs and to create an object that have ``__ve_array_interface__``.

.. code-block:: python

  from mpi4pyve import veo
  import ctypes
  import numpy as np
  import atexit
  
  _veo_proc_destroyed = False
  
  
  class VEMem(object):
      """
      Object that controls VE memory
      """
  
      def __init__(self, libveo, proc, ctxt, dtype, nelem):
          """
          Allocate VE memory
          """
          self.libveo = libveo
          self.proc = proc
          self.ctxt = ctxt
          self.dtype = dtype
          self.nelem = nelem
          self.nbytes = dtype.itemsize * nelem
          self.hmem = veo.alloc_hmem(self.proc, self.nbytes)
  
      def __del__(self):
          """
          Free VE memory
          """
          if not _veo_proc_destroyed:
              veo.free_hmem(self.hmem)
  
      def set_value(self, val):
          """
          Set value into VE memory
          """
          val = np.asarray(val, dtype=self.dtype)
          if val.size != self.nelem:
              raise ValueError
          src = ctypes.c_void_p(val.ctypes.data)
          dst = ctypes.c_void_p(self.hmem)
          ret = self.libveo.veo_hmemcpy(dst, src, self.nbytes)
          if ret:
            raise RuntimeError("ret = %d" % ret)
  
      def get_value(self):
          """
          Retrieve value from VE memory
          """
          vhbuf = np.empty(self.nelem, dtype=self.dtype)
          dst = ctypes.c_void_p(vhbuf.ctypes.data)
          src = ctypes.c_void_p(self.hmem)
          ret = self.libveo.veo_hmemcpy( dst, src, self.nbytes)
          if ret:
              raise RuntimeError("ret = %d" % ret)
          return vhbuf

      @property
      def __ve_array_interface__(self):
          """
          VE array interface for interoperating Python VE libraries.
          """
          return {
              'shape': (self.nelem,),
              'typestr': self.dtype.str,
              'version': 1,
              'strides': None,
              'data': (self.hmem, False)}
  
  class Veo(object):
  
      def __init__(self, venode, libpath='/opt/nec/ve/veos/lib64/libveo.so.1'):
          # Load shared object
          self.libveo = ctypes.cdll.LoadLibrary(libpath)
  
          #
          # Register argument types and return type for veo C APIs.
          #
          # veo_proc_create
          self.libveo.veo_proc_create.argtypes = (ctypes.c_int32,)
          self.libveo.veo_proc_create.restype = ctypes.c_uint64
          # veo_context_open
          self.libveo.veo_context_open.argtypes = (ctypes.c_void_p,)
          self.libveo.veo_context_open.restype = ctypes.c_uint64
          # veo_hmemcpy
          self.libveo.veo_hmemcpy.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
          self.libveo.veo_hmemcpy.restype = ctypes.c_int32
          # veo_proc_destroy
          self.libveo.veo_proc_destroy.argtypes = (ctypes.c_void_p,)
          self.libveo.veo_proc_destroy.restype = ctypes.c_int32
          # veo_context_close
          self.libveo.veo_context_close.argtypes = (ctypes.c_void_p,)
          self.libveo.veo_context_close.restype = ctypes.c_int32

          #
          # Call veo C APIs for initialization.
          #
          self.proc = self.libveo.veo_proc_create(ctypes.c_int32(venode))
          self.ctxt = self.libveo.veo_context_open(ctypes.c_void_p(self.proc))

          def finalize(libveo, ctxt, proc):
              # Close veo context and destroy veo process.
              libveo.veo_context_close(ctypes.c_void_p(ctxt))
              libveo.veo_proc_destroy(ctypes.c_void_p(proc))
              global _veo_proc_destroyed
              _veo_proc_destroyed = True

          # Register function that calls at exit time.
          atexit.register(finalize, self.libveo, self.ctxt, self.proc)
  
      def alloc_hmem(self, dtype, nelem):
          return VEMem(self.libveo, self.proc, self.ctxt, dtype, nelem)

| The above example uses ctypes to call veo C APIs from a Python script, although there are other ways to call them.
| e.g.) ctypes, cython, pybind, Python C API, etc.

---------
Execution
---------

::

  $ mpirun -veo -np 2 python mpi_send_recv.py
  Result Success (rank=0)
  Result Success (rank=1)


