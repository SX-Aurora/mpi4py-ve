import unittest  # NOQA
from unittest import TestCase  # NOQA
from parameterized import parameterized  # NOQA
from mpi4pyve import MPI  # NOQA
from mpi4pyve import util  # NOQA
from numpy.testing import (
    assert_array_equal,
)
import coverage_device_util  # NOQA
from coverage_device_util import (
    _patterns, _get_type, _get_sbuf, _get_rbuf, vp, np
)


class TestCoverageDeviceDataType(unittest.TestCase):

    COMM = MPI.COMM_WORLD

    @parameterized.expand(_patterns)
    def test_PackUnpack(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        peer = rank ^ 1
        if peer >= size:
            peer = MPI.PROC_NULL
        mtype = _get_type(dtype)

        desired = _get_sbuf(np, shape, dtype, order)
        comm.Bcast([desired, mtype], root=0)

        x = dev1.array(desired, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype, order, dev1)
        size_x = mtype.Pack_size(x.size, self.COMM)
        size_y = mtype.Pack_size(y.size, self.COMM)
        z = np.empty(max(size_x, size_y) * 16, dtype='b')
        mtype.Pack(x, z, 0, self.COMM)
        mtype.Unpack(z, 0, y, self.COMM)

        assert_array_equal(x, y)


if __name__ == '__main__':
    unittest.main()
