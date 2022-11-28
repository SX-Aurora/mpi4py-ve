import unittest  # NOQA
from unittest import TestCase  # NOQA
from parameterized import parameterized  # NOQA
from mpi4pyve import MPI  # NOQA
from mpi4pyve import util  # NOQA
from numpy.testing import (
    assert_equal,
)
import coverage_device_util  # NOQA
from coverage_device_util import (
    _patterns, _get_array, _get_type, _get_sbuf, _get_rbuf, vp, np
)


class TestCoverageDeviceMessage(unittest.TestCase):

    COMM = MPI.COMM_WORLD

    @parameterized.expand(_patterns)
    def test_Recv(self, dev1, dev2, shape, dtype, order):
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

        actual = _get_sbuf(np, shape, dtype=dtype, order=order)
        comm.Bcast(actual, root=0)
        x = dev1.array(actual, dtype=dtype, order=order)
        if rank % 2 == 0:
            comm.Send([x, mtype], dest=peer)
        else:
            y = _get_rbuf(dev2, shape, dtype=dtype, order=order)
            m = MPI.Message.Probe(comm)
            m.Recv(y)

            assert_equal(_get_array(y), actual)

    @parameterized.expand(_patterns)
    def test_Irecv(self, dev1, dev2, shape, dtype, order):
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
        actual = _get_sbuf(np, shape, dtype=dtype, order=order)
        comm.Bcast(actual, root=0)
        x = dev1.array(actual, dtype=dtype, order=order)
        if rank % 2 == 0:
            comm.Send([x, mtype], dest=peer)
        else:
            y = _get_rbuf(dev2, shape, dtype=dtype, order=order)
            m = MPI.Message.Probe(comm)
            m.Irecv(y).Wait()

            assert_equal(_get_array(y), actual)


if __name__ == '__main__':
    unittest.main()
