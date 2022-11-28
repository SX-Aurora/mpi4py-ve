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
    _patterns, _get_array, _get_sbuf, _get_rbuf, vp, np,
    _flush_test_patterns
)


class TestCoverageDeviceWin(unittest.TestCase):

    COMM = MPI.COMM_WORLD
    INFO = MPI.INFO_NULL

    def memzero(self, m):
        try:
            m[:] = 0
        except IndexError:  # cffi buffer
            m[0:len(m)] = b'\0' * len(m)

    def setUp(self):
        nbytes = 5 ** 11 * MPI.DOUBLE.size
        try:
            self.mpi_memory = MPI.Alloc_mem(nbytes)
            self.memory = self.mpi_memory
            self.memzero(self.memory)
        except MPI.Exception:
            import array
            self.mpi_memory = None
            self.memory = array.array('B', [0] * nbytes)
        self.WIN = MPI.Win.Create(self.memory, 1, self.INFO, self.COMM)
        try:
            self.actual_mpi_memory = MPI.Alloc_mem(nbytes)
            self.actual_memory = self.actual_mpi_memory
            self.memzero(self.actual_memory)
        except MPI.Exception:
            import array
            self.actual_mpi_memory = None
            self.actual_memory = array.array('B', [0] * nbytes)
        self.actual_WIN = MPI.Win.Create(self.actual_memory, 1, self.INFO,
                                         self.COMM)

    def tearDown(self):
        self.WIN.Free()
        if self.mpi_memory:
            MPI.Free_mem(self.mpi_memory)
        self.actual_WIN.Free()
        if self.actual_mpi_memory:
            MPI.Free_mem(self.actual_mpi_memory)

    @parameterized.expand(_patterns)
    def test_PutGet(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')

        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        peer = rank ^ 1
        if peer >= size:
            peer = MPI.PROC_NULL

        actual_y = _get_sbuf(np, shape, dtype=dtype, order=order)
        comm.Bcast(actual_y, root=0)
        x = dev1.array(actual_y, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype=dtype, order=order)
        target = x.itemsize
        self.WIN.Fence()
        self.WIN.Put(x, rank, target)
        self.WIN.Fence()
        self.WIN.Get(y, rank, target)
        self.WIN.Fence()

        assert_equal(_get_array(y), actual_y)

    @parameterized.expand(_patterns)
    def test_Accumulate(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        peer = rank ^ 1
        if peer >= size:
            peer = MPI.PROC_NULL

        x = _get_sbuf(dev1, shape, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype=dtype, order=order)
        actual_x = _get_sbuf(np, shape, dtype=dtype, order=order)
        actual_y = _get_rbuf(np, shape, dtype=dtype, order=order)
        for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
            self.WIN.Fence()
            self.WIN.Accumulate(x, rank, op=op)
            self.WIN.Fence()
            self.WIN.Get(y, rank)
            self.WIN.Fence()

            self.actual_WIN.Fence()
            self.actual_WIN.Accumulate(actual_x, rank, op=op)
            self.actual_WIN.Fence()
            self.actual_WIN.Get(actual_y, rank)
            self.actual_WIN.Fence()

            assert_equal(_get_array(x), actual_x)
            assert_equal(_get_array(y), actual_y)

    @parameterized.expand(_patterns)
    def test_GetAccumulate(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        peer = rank ^ 1
        if peer >= size:
            peer = MPI.PROC_NULL

        x = _get_sbuf(dev1, shape, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype=dtype, order=order)
        actual_x = _get_sbuf(np, shape, dtype=dtype, order=order)
        actual_y = _get_rbuf(np, shape, dtype=dtype, order=order)
        for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN, MPI.NO_OP):
            self.WIN.Fence()
            self.WIN.Put(x, rank)
            self.WIN.Get_accumulate(x, y, rank, op=op)
            self.WIN.Fence()
            self.WIN.Get(y, rank)
            self.WIN.Fence()

            self.actual_WIN.Fence()
            self.actual_WIN.Put(actual_x, rank)
            self.actual_WIN.Get_accumulate(actual_x, actual_y, rank, op=op)
            self.actual_WIN.Fence()
            self.actual_WIN.Get(actual_y, rank)
            self.actual_WIN.Fence()

            assert_equal(_get_array(x), actual_x)
            assert_equal(_get_array(y), actual_y)

    @parameterized.expand(_patterns)
    def test_Fetch_and_op(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        peer = rank ^ 1
        if peer >= size:
            peer = MPI.PROC_NULL

        x = _get_sbuf(dev1, 1, dtype=dtype, order=order)
        y = _get_rbuf(dev2, 1, dtype=dtype, order=order)
        x.fill(1)
        y.fill(-1)
        actual_x = _get_sbuf(np, 1, dtype=dtype, order=order)
        actual_y = _get_rbuf(np, 1, dtype=dtype, order=order)
        actual_x.fill(1)
        actual_y.fill(-1)
        for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN, MPI.REPLACE,
                   MPI.NO_OP):
            self.WIN.Fence()
            self.WIN.Fetch_and_op(x, y, rank, 0, op=op)
            self.WIN.Fence()

            self.actual_WIN.Fence()
            self.actual_WIN.Fetch_and_op(actual_x, actual_y, rank, 0, op=op)
            self.actual_WIN.Fence()

            assert_equal(_get_array(x), actual_x)
            assert_equal(_get_array(y), actual_y)

    @parameterized.expand(_patterns)
    def test_Compare_and_swap(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        peer = rank ^ 1
        if peer >= size:
            peer = MPI.PROC_NULL

        x = _get_sbuf(dev1, 1, dtype=dtype, order=order)
        y = _get_rbuf(dev2, 1, dtype=dtype, order=order)
        z = _get_rbuf(dev2, 1, dtype=dtype, order=order)
        x.fill(1)
        y.fill(0)
        z.fill(-1)
        self.WIN.Fence()
        self.WIN.Compare_and_swap(x, y, z, rank, 0)
        self.WIN.Fence()

        actual_x = _get_sbuf(np, 1, dtype=dtype, order=order)
        actual_y = _get_rbuf(np, 1, dtype=dtype, order=order)
        actual_z = _get_rbuf(np, 1, dtype=dtype, order=order)
        actual_x.fill(1)
        actual_y.fill(0)
        actual_z.fill(-1)
        self.actual_WIN.Fence()
        self.actual_WIN.Compare_and_swap(actual_x, actual_y, actual_z, rank, 0)
        self.actual_WIN.Fence()

        assert_equal(_get_array(x), actual_x)
        assert_equal(_get_array(y), actual_y)
        assert_equal(_get_array(z), actual_z)

    @parameterized.expand(_patterns)
    def test_Rput_Rget(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')

        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        peer = rank ^ 1
        if peer >= size:
            peer = MPI.PROC_NULL

        actual_y = _get_sbuf(np, shape, dtype=dtype, order=order)
        comm.Bcast(actual_y, root=0)
        x = dev1.array(actual_y, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype=dtype, order=order)
        self.WIN.Fence()
        self.WIN.Rput(x, rank).Wait()
        self.WIN.Fence()
        self.WIN.Rget(y, rank).Wait()
        self.WIN.Fence()

        assert_equal(_get_array(y), actual_y)

    @parameterized.expand(_patterns)
    def test_Raccumulate(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        peer = rank ^ 1
        if peer >= size:
            peer = MPI.PROC_NULL

        x = _get_sbuf(dev1, shape, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype=dtype, order=order)
        actual_x = _get_sbuf(np, shape, dtype=dtype, order=order)
        actual_y = _get_rbuf(np, shape, dtype=dtype, order=order)
        for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN, MPI.REPLACE):
            self.WIN.Fence()
            x.fill(1)
            self.WIN.Rput(x, rank).Wait()
            self.WIN.Fence()
            self.WIN.Raccumulate(x, rank, op=op).Wait()
            self.WIN.Fence()
            self.WIN.Rget(y, rank).Wait()
            self.WIN.Fence()

            self.actual_WIN.Fence()
            actual_x.fill(1)
            self.actual_WIN.Rput(actual_x, rank).Wait()
            self.actual_WIN.Fence()
            self.actual_WIN.Raccumulate(actual_x, rank, op=op).Wait()
            self.actual_WIN.Fence()
            self.actual_WIN.Rget(actual_y, rank).Wait()
            self.actual_WIN.Fence()

            assert_equal(_get_array(x), actual_x)
            assert_equal(_get_array(y), actual_y)

    @parameterized.expand(_patterns)
    def test_Rget_accumulate(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        peer = rank ^ 1
        if peer >= size:
            peer = MPI.PROC_NULL

        x = _get_sbuf(dev1, shape, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype=dtype, order=order)
        actual_x = _get_sbuf(np, shape, dtype=dtype, order=order)
        actual_y = _get_rbuf(np, shape, dtype=dtype, order=order)
        for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN, MPI.REPLACE,
                   MPI.NO_OP):
            self.WIN.Fence()
            x.fill(1)
            self.WIN.Rput(x, rank).Wait()
            self.WIN.Fence()
            self.WIN.Rget_accumulate(x, y, rank, op=op).Wait()
            self.WIN.Fence()
            self.WIN.Rget(y, rank).Wait()
            self.WIN.Fence()

            self.actual_WIN.Fence()
            actual_x.fill(1)
            self.actual_WIN.Rput(actual_x, rank).Wait()
            self.actual_WIN.Fence()
            self.actual_WIN.Rget_accumulate(actual_x, actual_y,
                                            rank, op=op).Wait()
            self.actual_WIN.Fence()
            self.actual_WIN.Rget(actual_y, rank).Wait()
            self.actual_WIN.Fence()

            assert_equal(_get_array(x), actual_x)
            assert_equal(_get_array(y), actual_y)

    # Fence : No NLCPy synchronization required.
    @parameterized.expand(_flush_test_patterns)
    def test_Fence_synchronize(self, dev1, dtype, order):
        comm = self.COMM
        rank = comm.Get_rank()

        n = dev1.array(0, dtype=dtype, order=order)
        expect = dev1.array(1, dtype=dtype, order=order)

        if rank == 0:
            win_n = MPI.Win.Create(n, comm=MPI.COMM_WORLD)
        else:
            win_n = MPI.Win.Create(None, comm=MPI.COMM_WORLD)
        if rank == 0:
            n.fill(1)
        win_n.Fence()
        if rank != 0:
            win_n.Get([n, MPI.INT], 0)
        win_n.Fence()
        win_n.Free()
        assert_equal(_get_array(n), _get_array(expect))

    # Sync : No NLCPy synchronization required.
    @parameterized.expand(_flush_test_patterns)
    def test_Sync_synchronize(self, dev1, dtype, order):
        comm = self.COMM
        rank = comm.Get_rank()

        n = dev1.array(0, dtype=dtype, order=order)
        expect = dev1.array(1, dtype=dtype, order=order)

        if rank == 0:
            win_n = MPI.Win.Create(n, comm=MPI.COMM_WORLD)
        else:
            win_n = MPI.Win.Create(None, comm=MPI.COMM_WORLD)
        if rank == 0:
            n.fill(1)
            if dev1 == vp:
                n.venode.synchronize()
        if rank != 0:
            win_n.Lock(MPI.LOCK_EXCLUSIVE, 0)
            n.fill(0)
            win_n.Sync()
            win_n.Get([n, MPI.INT], 0)
            win_n.Unlock(0)
        comm.Barrier()
        win_n.Free()
        assert_equal(_get_array(n), _get_array(expect))

    # Post : No NLCPy synchronization required.
    @parameterized.expand(_flush_test_patterns)
    def test_Post_synchronize(self, dev1, dtype, order):
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        peer = rank ^ 1
        if peer >= size:
            peer = MPI.PROC_NULL

        n = dev1.array(0, dtype=dtype, order=order)
        expect = dev1.array(1, dtype=dtype, order=order)
        comm_group = comm.Get_group()

        if rank % 2 == 0:
            win_n = MPI.Win.Create(n, comm=MPI.COMM_WORLD)
        else:
            win_n = MPI.Win.Create(None, comm=MPI.COMM_WORLD)
        if peer != MPI.PROC_NULL:
            group = comm_group.Incl(list([peer]))
            if rank % 2 == 0:
                n.fill(1)
                win_n.Post(group)
                win_n.Wait()
            else:
                win_n.Start(group)
                win_n.Get([n, MPI.INT], peer)
                win_n.Complete()
            group.Free()
        win_n.Free()
        comm_group.Free()
        if peer != MPI.PROC_NULL:
            assert_equal(_get_array(n), _get_array(expect))

    # Start : No NLCPy synchronization required.
    @parameterized.expand(_flush_test_patterns)
    def test_Start_synchronize(self, dev1, dtype, order):
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        peer = rank ^ 1
        if peer >= size:
            peer = MPI.PROC_NULL

        n = dev1.array(0, dtype=dtype, order=order)
        expect = dev1.array(1, dtype=dtype, order=order)
        comm_group = comm.Get_group()

        if rank % 2 == 0:
            win_n = MPI.Win.Create(n, comm=MPI.COMM_WORLD)
        else:
            win_n = MPI.Win.Create(None, comm=MPI.COMM_WORLD)
        if peer != MPI.PROC_NULL:
            group = comm_group.Incl(list([peer]))
            if rank % 2 == 0:
                win_n.Post(group)
                win_n.Wait()
            else:
                n.fill(1)
                win_n.Start(group)
                win_n.Put([n, MPI.INT], peer)
                win_n.Complete()
            group.Free()
        win_n.Free()
        comm_group.Free()
        if peer != MPI.PROC_NULL:
            assert_equal(_get_array(n), _get_array(expect))

    # Lock : NLCPy synchronization required.
    @parameterized.expand(_flush_test_patterns)
    def test_Lock_synchronize(self, dev1, dtype, order):
        comm = self.COMM
        rank = comm.Get_rank()

        n = dev1.array(0, dtype=dtype, order=order)
        expect = dev1.array(1, dtype=dtype, order=order)

        if rank == 0:
            win_n = MPI.Win.Create(n, comm=MPI.COMM_WORLD)
        else:
            win_n = MPI.Win.Create(None, comm=MPI.COMM_WORLD)
        if rank == 0:
            n.fill(1)
            if dev1 == vp:
                n.venode.synchronize()
        comm.Barrier()
        if rank != 0:
            win_n.Lock(MPI.LOCK_EXCLUSIVE, 0)
            win_n.Get([n, MPI.INT], 0)
            win_n.Unlock(0)
        comm.Barrier()
        win_n.Free()
        assert_equal(_get_array(n), _get_array(expect))

    # Lock_all : NLCPy synchronization required.
    @parameterized.expand(_flush_test_patterns)
    def test_Lock_all_synchronize(self, dev1, dtype, order):
        comm = self.COMM
        rank = comm.Get_rank()

        n = dev1.array(0, dtype=dtype, order=order)
        expect = dev1.array(1, dtype=dtype, order=order)

        if rank == 0:
            win_n = MPI.Win.Create(n, comm=MPI.COMM_WORLD)
        else:
            win_n = MPI.Win.Create(None, comm=MPI.COMM_WORLD)
        if rank == 0:
            n.fill(1)
            if dev1 == vp:
                n.venode.synchronize()
        comm.Barrier()
        if rank != 0:
            win_n.Lock_all()
            win_n.Get([n, MPI.INT], 0)
            win_n.Unlock_all()
        comm.Barrier()
        win_n.Free()
        assert_equal(_get_array(n), _get_array(expect))


if __name__ == '__main__':
    unittest.main()
