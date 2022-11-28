from mpi4pyve import MPI
from numpy.testing import assert_array_equal
from functools import reduce
import os


if os.environ.get("MPI4PYVE_TEST_PATTERN") == "small":
    _shapes = [5, (2, 20), (2, 10, 20), (2, 10, 20, 4)]
else:
    _shapes = [5, 5**10,
               (2,), (2, 2), (2, 2, 2), (2, 20), (20, 2),
               (2, 10, 20), (10, 2, 20), (10, 20, 2),
               (2, 10, 20, 4), (10, 2, 4, 20), (4, 10, 20, 2)]

np = None
vp = None
_devices = None

if os.environ.get("MPI4PYVE_TEST_DEVICE") == "vh":
    import numpy as _np
    _devices = [_np]
    np = _np
elif os.environ.get("MPI4PYVE_TEST_DEVICE") == "ve":
    import nlcpy as _vp
    import numpy as _np
    _devices = [_vp]
    vp = _vp
    np = _np
else:
    import nlcpy as _vp
    import numpy as _np
    _devices = [_np, _vp]
    vp = _vp
    np = _np

_dtypes = ['int32', 'int64',
           'uint32', 'uint64',
           'float32', 'float64',
           'complex64', 'complex128',
           'bool']
_flush_dtypes = ['int32']
_order = ['C', 'F']
_patterns = [(dev1, dev2, shape, dtype, order)
             for dev1 in _devices
             for dev2 in _devices
             for shape in _shapes
             for dtype in _dtypes
             for order in _order]
_flush_test_patterns = [(dev1, dtype, order)
                        for dev1 in _devices
                        for dtype in _flush_dtypes
                        for order in _order]
_default_fill_value = -1
_rbuf_np_bool_size_adjust = 4


def _get_array(a):
    if vp is not None and isinstance(a, vp.ndarray):
        return a.get()
    return a


def _get_type(dtype):
    if dtype == 'int32':
        return MPI.INT
    elif dtype == 'int64':
        return MPI.LONG
    elif dtype == 'uint32':
        return MPI.UNSIGNED
    elif dtype == 'uint64':
        return MPI.UNSIGNED_LONG
    elif dtype == 'float32':
        return MPI.FLOAT
    elif dtype == 'float64':
        return MPI.DOUBLE
    elif dtype == 'complex64':
        return MPI.COMPLEX
    elif dtype == 'complex128':
        return MPI.DOUBLE_COMPLEX
    elif dtype == 'bool':
        return MPI.BOOL


def _get_sbuf(dev, shape, dtype, order):
    if dtype != 'bool':
        if isinstance(shape, tuple) or isinstance(shape, list):
            n = reduce((lambda x, y: x * y), shape)
            return dev.arange(n, dtype=dtype).reshape(shape, order=order)
        else:
            return dev.arange(shape, dtype=dtype).reshape(shape, order=order)
    else:
        return dev.random.randint(0, 2, shape).astype('?', order=order)


def _get_rbuf(dev, shape, dtype, order, fromdev=None):
    if dtype == 'bool':
        val = False
    else:
        val = _default_fill_value
    return dev.full(shape, val, dtype=dtype, order=order)


def _assert_array(a, desired):
    if desired is None:
        return np.all(a == 0)
    else:
        return assert_array_equal(a, desired)


IS_MULTI_HOST = None

try:
    if IS_MULTI_HOST is None:
        comm = MPI.COMM_WORLD
        nodes = comm.allgather(os.environ['MPINODEID'])
        IS_MULTI_HOST = (len(list(set(nodes))) != 1)
except KeyError:
    pass
