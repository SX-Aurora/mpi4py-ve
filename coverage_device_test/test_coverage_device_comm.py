import unittest  # NOQA
from unittest import TestCase  # NOQA
from parameterized import parameterized  # NOQA
from mpi4pyve import MPI  # NOQA
from numpy.testing import (
    assert_equal,
)
import coverage_device_util  # NOQA
from coverage_device_util import (
    _patterns, _get_array, _get_type, _get_sbuf,
    _get_rbuf, _assert_array, vp, np
)


def create_topo_comms(comm):
    size = comm.Get_size()
    rank = comm.Get_rank()
    # Cartesian
    n = int(size ** 1 / 2.0)
    m = int(size ** 1 / 3.0)
    if m * m * m == size:
        dims = [m, m, m]
    elif n * n == size:
        dims = [n, n]
    else:
        dims = [size]
    periods = [True] * len(dims)
    yield comm.Create_cart(dims, periods=periods)
    # Graph
    index, edges = [0], []
    for i in range(size):
        pos = index[-1]
        index.append(pos + 2)
        edges.append((i - 1) % size)
        edges.append((i + 1) % size)
    yield comm.Create_graph(index, edges)
    # Dist Graph
    sources = [(rank - 2) % size, (rank - 1) % size]
    destinations = [(rank + 1) % size, (rank + 2) % size]
    yield comm.Create_dist_graph_adjacent(sources, destinations)


def get_neighbors_count(comm):
    topo = comm.Get_topology()
    if topo == MPI.CART:
        ndim = comm.Get_dim()
        return 2 * ndim, 2 * ndim
    if topo == MPI.GRAPH:
        rank = comm.Get_rank()
        nneighbors = comm.Get_neighbors_count(rank)
        return nneighbors, nneighbors
    if topo == MPI.DIST_GRAPH:
        indeg, outdeg, w = comm.Get_dist_neighbors_count()
        return indeg, outdeg
    return 0, 0


class TestComm(unittest.TestCase):

    COMM = MPI.COMM_WORLD

    @parameterized.expand(_patterns)
    def test_Send_Recv(self, dev1, dev2, shape, dtype, order):
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
        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            comm.Send([x, mtype], dest=peer)
        else:
            y = _get_rbuf(dev2, shape, dtype, order, dev1)
            comm.Recv([y, mtype], source=peer)

            self.assertFalse(_assert_array(y, desired))

    @parameterized.expand(_patterns)
    def test_send_recv_offset(self, dev1, dev2, shape, dtype, order):
        if dev1 is np or dev2 is np:
            self.skipTest('buffer is not contiguous case is not testable')
        if isinstance(shape, tuple):
            self.skipTest('unsupported tuple offset case is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        peer = rank ^ 1
        if peer >= size:
            peer = MPI.PROC_NULL

        mtype = _get_type(dtype)
        offset = 2

        desired = _get_sbuf(np, shape, dtype, order)[offset:]
        comm.Bcast([desired, mtype], root=0)
        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            comm.Send([x, mtype], dest=peer)
        else:
            y = _get_rbuf(dev2, shape - offset, dtype, order, dev1)
            comm.Recv([y, mtype], source=peer)

            self.assertFalse(_assert_array(y, desired))

    @parameterized.expand(_patterns)
    def test_Sendrecv(self, dev1, dev2, shape, dtype, order):
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

        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            a = _get_rbuf(dev2, shape, dtype, order, dev1)
            comm.Sendrecv([x, mtype], dest=peer, sendtag=0, recvbuf=[a, mtype],
                          source=peer, recvtag=0)
            if peer != MPI.PROC_NULL:
                self.assertFalse(_assert_array(a, x))
        else:
            y = dev1.array(desired, dtype=dtype, order=order)
            z = _get_rbuf(dev2, shape, dtype, order, dev1)
            comm.Sendrecv([y, mtype], dest=peer, sendtag=0, recvbuf=[z, mtype],
                          source=peer, recvtag=0)
            if peer != MPI.PROC_NULL:
                self.assertFalse(_assert_array(z, y))

    @parameterized.expand(_patterns)
    def test_Sendrecv_replace(self, dev1, dev2, shape, dtype, order):
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
        desired_0 = _get_sbuf(np, shape, dtype, order)
        desired_1 = _get_sbuf(np, shape, dtype, order)
        comm.Bcast([desired_0, mtype], root=0)
        comm.Bcast([desired_1, mtype], root=1)

        if rank % 2 == 0:
            x = dev1.array(desired_1, dtype=dtype, order=order)
            comm.Sendrecv_replace([x, mtype], dest=peer, sendtag=0, source=peer,
                                  recvtag=0)
            if peer != MPI.PROC_NULL:
                self.assertFalse(_assert_array(x, desired_0))
        else:
            y = dev2.array(desired_0, dtype=dtype, order=order)
            comm.Sendrecv_replace([y, mtype], dest=peer, sendtag=0, source=peer,
                                  recvtag=0)
            if peer != MPI.PROC_NULL:
                self.assertFalse(_assert_array(y, desired_1))

    @parameterized.expand(_patterns)
    def test_Isend_Recv(self, dev1, dev2, shape, dtype, order):
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
        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            comm.Isend([x, mtype], dest=peer).Wait()
        else:
            y = _get_rbuf(dev2, shape, dtype, order, dev1)
            comm.Recv([y, mtype], source=peer)

            self.assertFalse(_assert_array(y, desired))

    @parameterized.expand(_patterns)
    def test_Send_Irecv(self, dev1, dev2, shape, dtype, order):
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
        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            comm.Send([x, mtype], dest=peer)
        else:
            y = _get_rbuf(dev2, shape, dtype, order, dev1)
            comm.Irecv([y, mtype], source=peer).Wait()

            self.assertFalse(_assert_array(y, desired))

    @parameterized.expand(_patterns)
    def test_Send_init(self, dev1, dev2, shape, dtype, order):
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
        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            req = comm.Send_init([x, mtype], dest=peer)
            req.Start()
            req.Wait()
        else:
            y = _get_rbuf(dev2, shape, dtype, order, dev1)
            comm.Recv([y, mtype], source=peer)

            self.assertFalse(_assert_array(y, desired))

    @parameterized.expand(_patterns)
    def test_Recv_init(self, dev1, dev2, shape, dtype, order):
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
        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            req = comm.Send([x, mtype], dest=peer)
        else:
            y = _get_rbuf(dev2, shape, dtype, order, dev1)
            req = comm.Recv_init([y, mtype], source=peer)
            req.Start()
            req.Wait()
            self.assertFalse(_assert_array(y, desired))

    @parameterized.expand(_patterns)
    def test_Rsend_Recv(self, dev1, dev2, shape, dtype, order):
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
        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            comm.Rsend([x, mtype], dest=peer)
        else:
            y = _get_rbuf(dev2, shape, dtype, order, dev1)
            comm.Recv([y, mtype], source=peer)
            self.assertFalse(_assert_array(y, desired))

    @parameterized.expand(_patterns)
    def test_Ssend_Recv(self, dev1, dev2, shape, dtype, order):
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
        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            comm.Ssend([x, mtype], dest=peer)
        else:
            y = _get_rbuf(dev2, shape, dtype, order, dev1)
            comm.Recv([y, mtype], source=peer)
            self.assertFalse(_assert_array(y, desired))

    @parameterized.expand(_patterns)
    def test_Issend_Recv(self, dev1, dev2, shape, dtype, order):
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
        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            comm.Issend([x, mtype], dest=peer).Wait()
        else:
            y = _get_rbuf(dev2, shape, dtype, order, dev1)
            comm.Recv([y, mtype], source=peer)
            self.assertFalse(_assert_array(y, desired))

    @parameterized.expand(_patterns)
    def test_Irsend_Recv(self, dev1, dev2, shape, dtype, order):
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
        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            comm.Irsend([x, mtype], dest=peer).Wait()
        else:
            y = _get_rbuf(dev2, shape, dtype, order, dev1)
            comm.Recv([y, mtype], source=peer)
            self.assertFalse(_assert_array(y, desired))

    @parameterized.expand(_patterns)
    def test_Ssend_init(self, dev1, dev2, shape, dtype, order):
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
        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            req = comm.Ssend_init([x, mtype], dest=peer)
            req.Start()
            req.Wait()
        else:
            y = _get_rbuf(dev2, shape, dtype, order, dev1)
            comm.Recv([y, mtype], source=peer)
            self.assertFalse(_assert_array(y, desired))

    @parameterized.expand(_patterns)
    def test_Rsend_init(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        peer = rank ^ 1
        if peer >= size:
            peer = MPI.PROC_NULL

        mtype = _get_type(dtype)
        desired = _get_sbuf(np, shape, dtype, order)
        comm.Bcast([desired, mtype], root=0)
        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            req = comm.Rsend_init([x, mtype], dest=peer)
            req.Start()
            req.Wait()
        else:
            y = _get_rbuf(dev2, shape, dtype, order, dev1)
            comm.Recv([y, mtype], source=peer)
            self.assertFalse(_assert_array(y, desired))

    @parameterized.expand(_patterns)
    def test_Bcast(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        mtype = _get_type(dtype)

        for root in range(size):
            sbuf = _get_sbuf(np, size, dtype, order)
            comm.Bcast([sbuf, mtype], root=root)

            if rank == root:
                x = dev1.array(sbuf, dtype=dtype, order=order)
            else:
                x = _get_rbuf(dev2, size, dtype, order, dev1)

            comm.Bcast([x, mtype], root=root)
            self.assertFalse(_assert_array(x, sbuf))

    @parameterized.expand(_patterns)
    def test_Gather(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        mtype = _get_type(dtype)

        for root in range(size):
            if rank == root:
                sbuf = dev1.full(root + 1, root, dtype=dtype, order=order)
                rbuf = dev2.full((size, root + 1), -1, dtype=dtype,
                                 order=order)
            else:
                sbuf = dev1.full(root + 1, root, dtype=dtype, order=order)
                rbuf = dev2.array([], dtype=dtype, order=order)
            comm.Gather([sbuf, mtype], [rbuf, mtype], root=root)

            if rank == root:
                desired = np.full((size, root + 1), root, dtype=dtype,
                                  order=order)
                assert_equal(_get_array(rbuf), desired)

    @parameterized.expand(_patterns)
    def test_Gatherv(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        mtype = _get_type(dtype)

        for root in range(size):
            if rank == root:
                sbuf = dev1.full(root + 1, root, dtype=dtype, order=order)
                rbuf = dev2.full((size, root + 1), -1, dtype=dtype,
                                 order=order)
            else:
                sbuf = dev1.full(root + 1, root, dtype=dtype, order=order)
                rbuf = dev2.array([], dtype=dtype)
            comm.Gatherv([sbuf, mtype], [rbuf, mtype], root=root)

            if rank == root:
                desired = np.full((size, root + 1), root, dtype=dtype,
                                  order=order)
                assert_equal(_get_array(rbuf), desired)

    @parameterized.expand(_patterns)
    def test_Scatter(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        mtype = _get_type(dtype)

        for root in range(size):
            if rank == root:
                sbuf = dev1.full((size, size), root, dtype=dtype, order=order)
                rbuf = dev2.full(size, -1, dtype=dtype, order=order)
            else:
                sbuf = dev1.array([], dtype=dtype)
                rbuf = dev2.full(size, -1, dtype=dtype, order=order)
            comm.Scatter([sbuf, mtype], [rbuf, mtype], root=root)

            desired = np.full(size, root, dtype=dtype, order=order)
            assert_equal(_get_array(rbuf), desired)

    @parameterized.expand(_patterns)
    def test_Scatterv(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        mtype = _get_type(dtype)

        for root in range(size):
            if rank == root:
                sbuf = dev1.full((size, size), root, dtype=dtype, order=order)
                rbuf = dev2.full(size, -1, dtype=dtype, order=order)
            else:
                sbuf = dev1.array([], dtype=dtype, order=order)
                rbuf = dev2.full(size, -1, dtype=dtype, order=order)
            comm.Scatterv([sbuf, mtype], [rbuf, mtype], root=root)

            desired = np.full(size, root, dtype=dtype, order=order)
            assert_equal(_get_array(rbuf), desired)

    @parameterized.expand(_patterns)
    def test_Allgather(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        mtype = _get_type(dtype)

        for root in range(size):
            sbuf = dev1.full(root + 1, root, dtype=dtype, order=order)
            rbuf = dev2.full((size, root + 1), -1, dtype=dtype, order=order)
            comm.Allgather([sbuf, mtype], [rbuf, mtype])

            desired = np.full((size, root + 1), root, dtype=dtype, order=order)
            assert_equal(_get_array(rbuf), desired)

    @parameterized.expand(_patterns)
    def test_Allgatherv(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        if isinstance(shape, tuple) or isinstance(shape, list):
            self.skipTest('shape case in tuple or list is not testable')
        comm = self.COMM
        size = comm.Get_size()
        # for terminated by signal(9).
        if size > 4 and np.isscalar(shape) and shape >= 5**10:
            shape = 5**5

        for root in range(size):
            sbuf = _get_sbuf(dev1, shape, dtype, order)
            rbuf = _get_rbuf(dev2, size * shape, dtype, order, dev1)
            self.COMM.Allgatherv(sbuf, rbuf)

            actual_sbuf = _get_sbuf(np, shape, dtype, order)
            actual_rbuf = _get_rbuf(np, size * shape, dtype, order, np)
            self.COMM.Allgatherv(actual_sbuf, actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Alltoall(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()

        for root in range(size):
            sbuf = _get_sbuf(dev1, (size, root + 1), dtype, order)
            rbuf = _get_rbuf(dev2, (size, root + 1), dtype, order, dev1)
            self.COMM.Alltoall(sbuf, rbuf)

            actual_sbuf = np.array(sbuf, dtype=dtype, order=order)
            actual_rbuf = _get_rbuf(np, (size, root + 1), dtype, order, np)
            self.COMM.Alltoall(actual_sbuf, actual_rbuf)
            assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Alltoallv(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()

        for root in range(size):
            sbuf = _get_sbuf(dev1, (size, size), dtype, order)
            rbuf = _get_rbuf(dev2, (size, size), dtype, order, dev1)
            self.COMM.Alltoallv(sbuf, rbuf)

            actual_sbuf = np.array(sbuf, dtype=dtype, order=order)
            actual_rbuf = _get_rbuf(np, (size, size), dtype, order, np)
            self.COMM.Alltoallv(actual_sbuf, actual_rbuf)
            assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Alltoallw(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        mtype = _get_type(dtype)

        sdt, rdt = mtype, mtype
        sbuf = _get_sbuf(dev1, (size, 1), dtype, order)
        rbuf = _get_rbuf(dev2, (size, 1), dtype, order, dev1)
        sdsp = list(range(0, size * sbuf.itemsize, sbuf.itemsize))
        rdsp = list(range(0, size * rbuf.itemsize, rbuf.itemsize))
        smsg = (sbuf, ([1] * size, sdsp), [sdt] * size)
        rmsg = (rbuf, ([1] * size, rdsp), [rdt] * size)
        self.COMM.Alltoallw(smsg, rmsg)

        actual_sbuf = np.array(sbuf, dtype=dtype, order=order)
        actual_rbuf = _get_rbuf(np, (size, 1), dtype, order, np)
        actual_sdsp = list(range(0, size * actual_sbuf.itemsize,
                                 actual_sbuf.itemsize))
        actual_rdsp = list(range(0, size * actual_rbuf.itemsize,
                                 actual_rbuf.itemsize))
        actual_smsg = (actual_sbuf, ([1] * size, actual_sdsp), [sdt] * size)
        actual_rmsg = (actual_rbuf, ([1] * size, actual_rdsp), [rdt] * size)
        self.COMM.Alltoallw(actual_smsg, actual_rmsg)
        assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Reduce(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        size = comm.Get_size()
        mtype = _get_type(dtype)

        for root in range(size):
            for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
                sbuf = dev1.array(range(size), dtype=dtype, order=order)
                rbuf = dev2.full(size, -1, dtype=dtype, order=order)

                comm.Reduce([sbuf, mtype], [rbuf, mtype], op, root)

                actual_sbuf = np.array(range(size), dtype=dtype, order=order)
                actual_rbuf = np.full(size, -1, dtype=dtype, order=order)
                comm.Reduce([actual_sbuf, mtype], [actual_rbuf, mtype], op,
                            root)
                assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Allreduce(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        size = comm.Get_size()
        mtype = _get_type(dtype)

        for root in range(size):
            for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
                sbuf = dev1.array(range(size), dtype=dtype, order=order)
                rbuf = dev2.full(size, -1, dtype=dtype, order=order)

                comm.Allreduce([sbuf, mtype], [rbuf, mtype], op)

                actual_sbuf = np.array(range(size), dtype=dtype, order=order)
                actual_rbuf = np.full(size, -1, dtype=dtype, order=order)
                comm.Allreduce([actual_sbuf, mtype], [actual_rbuf, mtype], op)
                assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Reduce_scatter(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        mtype = _get_type(dtype)

        for root in range(size):
            for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
                rcnt = list(range(1, size + 1))
                sbuf = dev1.array([rank + 1] * sum(rcnt), dtype=dtype,
                                  order=order)
                rbuf = dev2.full(rank + 1, -1, dtype=dtype, order=order)

                comm.Reduce_scatter([sbuf, mtype], [rbuf, mtype], None, op)

                actual_sbuf = np.array([rank + 1] * sum(rcnt), dtype=dtype,
                                       order=order)
                actual_rbuf = np.full(rank + 1, -1, dtype=dtype, order=order)
                comm.Reduce_scatter([actual_sbuf, mtype],
                                    [actual_rbuf, mtype], None, op)
                assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Reduce_scatter_block(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        mtype = _get_type(dtype)

        for root in range(size):
            for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
                for rcnt in range(1, size):
                    sbuf = dev1.array([rank] * rcnt * size, dtype=dtype,
                                      order=order)
                    rbuf = dev2.full(rcnt, -1, dtype=dtype, order=order)
                    if op == MPI.PROD:
                        sbuf = dev1.array([rank + 1] * rcnt * size,
                                          dtype=dtype, order=order)
                    comm.Reduce_scatter_block([sbuf, mtype], [rbuf, mtype],
                                              op=op)

                    actual_sbuf = np.array([rank] * rcnt * size, dtype=dtype,
                                           order=order)
                    actual_rbuf = np.full(rcnt, -1, dtype=dtype, order=order)
                    if op == MPI.PROD:
                        actual_sbuf = dev1.array([rank + 1] * rcnt * size,
                                                 dtype=dtype, order=order)
                    comm.Reduce_scatter_block([actual_sbuf, mtype],
                                              [actual_rbuf, mtype], op=op)
                    assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_send_recv(self, dev1, dev2, shape, dtype, order):
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        peer = rank ^ 1
        if peer >= size:
            peer = MPI.PROC_NULL
        mtype = _get_type(dtype)

        desired = _get_sbuf(np, shape, dtype, order)
        comm.Bcast([desired, mtype], root=0)
        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            comm.send(x, dest=peer)
        else:
            y = comm.recv(source=peer)
            self.assertFalse(_assert_array(y, desired))

    @parameterized.expand(_patterns)
    def test_Ibcast(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        mtype = _get_type(dtype)

        for root in range(size):
            sbuf = _get_sbuf(np, size, dtype, order)
            comm.Bcast([sbuf, mtype], root=root)

            if rank == root:
                x = dev1.array(sbuf, dtype=dtype, order=order)
            else:
                x = _get_rbuf(dev2, size, dtype, order, dev1)

            comm.Ibcast([x, mtype], root=root).Wait()
            self.assertFalse(_assert_array(x, sbuf))

    @parameterized.expand(_patterns)
    def test_Igather(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        mtype = _get_type(dtype)

        for root in range(size):
            if rank == root:
                sbuf = dev1.full(root + 1, root, dtype=dtype, order=order)
                rbuf = dev2.full((size, root + 1), -1, dtype=dtype,
                                 order=order)
            else:
                sbuf = dev1.full(root + 1, root, dtype=dtype, order=order)
                rbuf = dev2.array([], dtype=dtype, order=order)
            comm.Igather([sbuf, mtype], [rbuf, mtype], root=root).Wait()

            if rank == root:
                desired = np.full((size, root + 1), root, dtype=dtype,
                                  order=order)
                assert_equal(_get_array(rbuf), desired)

    @parameterized.expand(_patterns)
    def test_Igatherv(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        mtype = _get_type(dtype)

        for root in range(size):
            if rank == root:
                sbuf = dev1.full(root + 1, root, dtype=dtype, order=order)
                rbuf = dev2.full((size, root + 1), -1, dtype=dtype,
                                 order=order)
            else:
                sbuf = dev1.full(root + 1, root, dtype=dtype, order=order)
                rbuf = dev2.array([], dtype=dtype, order=order)
            comm.Igatherv([sbuf, mtype], [rbuf, mtype], root=root).Wait()

            if rank == root:
                desired = np.full((size, root + 1), root, dtype=dtype,
                                  order=order)
                assert_equal(_get_array(rbuf), desired)

    @parameterized.expand(_patterns)
    def test_Iscatter(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        mtype = _get_type(dtype)

        for root in range(size):
            if rank == root:
                sbuf = dev1.full((size, size), root, dtype=dtype, order=order)
                rbuf = dev2.full(size, -1, dtype=dtype, order=order)
            else:
                sbuf = dev1.array([], dtype=dtype, order=order)
                rbuf = dev2.full(size, -1, dtype=dtype, order=order)
            comm.Iscatter([sbuf, mtype], [rbuf, mtype], root=root).Wait()

            desired = np.full(size, root, dtype=dtype, order=order)
            assert_equal(_get_array(rbuf), desired)

    @parameterized.expand(_patterns)
    def test_Iscatterv(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        mtype = _get_type(dtype)

        for root in range(size):
            if rank == root:
                sbuf = dev1.full((size, size), root, dtype=dtype, order=order)
                rbuf = dev2.full(size, -1, dtype=dtype, order=order)
            else:
                sbuf = dev1.array([], dtype=dtype, order=order)
                rbuf = dev2.full(size, -1, dtype=dtype, order=order)
            comm.Iscatterv([sbuf, mtype], [rbuf, mtype], root=root).Wait()

            desired = np.full(size, root, dtype=dtype, order=order)
            assert_equal(_get_array(rbuf), desired)

    @parameterized.expand(_patterns)
    def test_Iallgather(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        mtype = _get_type(dtype)

        for root in range(size):
            sbuf = dev1.full(root + 1, root, dtype=dtype, order=order)
            rbuf = dev2.full((size, root + 1), -1, dtype=dtype, order=order)
            comm.Iallgather([sbuf, mtype], [rbuf, mtype]).Wait()

            desired = np.full((size, root + 1), root, dtype=dtype, order=order)
            assert_equal(_get_array(rbuf), desired)

    @parameterized.expand(_patterns)
    def test_Iallgatherv(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        mtype = _get_type(dtype)

        for root in range(size):
            actual_sbuf = _get_sbuf(np, size, dtype, order)
            actual_rbuf = _get_rbuf(np, size * size, dtype, order, np)
            self.COMM.Iallgatherv(actual_sbuf, actual_rbuf).Wait()

            sbuf = dev1.array(actual_sbuf, dtype=dtype, order=order)
            rbuf = _get_rbuf(dev2, size * size, dtype, order, dev1)
            self.COMM.Iallgatherv(sbuf, rbuf).Wait()

            assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Ialltoall(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()

        for root in range(size):
            actual_sbuf = _get_sbuf(np, (size, size), dtype, order)
            actual_rbuf = _get_rbuf(np, (size, size), dtype, order, np)
            self.COMM.Ialltoall(actual_sbuf, actual_rbuf).Wait()

            sbuf = dev1.array(actual_sbuf, dtype=dtype, order=order)
            rbuf = _get_rbuf(dev2, (size, size), dtype, order, dev1)
            self.COMM.Ialltoall(sbuf, rbuf).Wait()

            assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Ialltoallv(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()

        for root in range(size):
            actual_sbuf = _get_sbuf(np, (size, size), dtype, order)
            actual_rbuf = _get_rbuf(np, (size, size), dtype, order, np)
            self.COMM.Ialltoallv(actual_sbuf, actual_rbuf).Wait()
            sbuf = dev1.array(actual_sbuf, dtype=dtype, order=order)
            rbuf = _get_rbuf(dev2, (size, size), dtype, order, dev1)
            self.COMM.Ialltoallv(sbuf, rbuf).Wait()

            assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Ialltoallw(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        mtype = _get_type(dtype)

        sdt, rdt = mtype, mtype

        actual_sbuf = _get_sbuf(np, (size, 1), dtype, order)
        actual_rbuf = _get_rbuf(np, (size, 1), dtype, order, np)
        actual_sdsp = list(range(0, size * actual_sbuf.itemsize,
                                 actual_sbuf.itemsize))
        actual_rdsp = list(range(0, size * actual_rbuf.itemsize,
                                 actual_rbuf.itemsize))
        actual_smsg = (actual_sbuf, ([1] * size, actual_sdsp), [sdt] * size)
        actual_rmsg = (actual_rbuf, ([1] * size, actual_rdsp), [rdt] * size)
        self.COMM.Ialltoallw(actual_smsg, actual_rmsg).Wait()

        sbuf = dev1.array(actual_sbuf, dtype=dtype, order=order)
        rbuf = _get_rbuf(dev2, (size, 1), dtype, order, dev1)
        sdsp = list(range(0, size * sbuf.itemsize, sbuf.itemsize))
        rdsp = list(range(0, size * rbuf.itemsize, rbuf.itemsize))
        smsg = (sbuf, ([1] * size, sdsp), [sdt] * size)
        rmsg = (rbuf, ([1] * size, rdsp), [rdt] * size)
        self.COMM.Ialltoallw(smsg, rmsg).Wait()

        assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Ireduce(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        size = comm.Get_size()
        mtype = _get_type(dtype)

        for root in range(size):
            for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
                sbuf = dev1.array(range(size), dtype=dtype, order=order)
                rbuf = dev2.full(size, -1, dtype=dtype, order=order)

                comm.Ireduce([sbuf, mtype], [rbuf, mtype], op, root).Wait()

                actual_sbuf = np.array(range(size), dtype=dtype, order=order)
                actual_rbuf = np.full(size, -1, dtype=dtype, order=order)
                comm.Ireduce([actual_sbuf, mtype],
                             [actual_rbuf, mtype], op, root).Wait()
                assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Iallreduce(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        size = comm.Get_size()
        mtype = _get_type(dtype)

        for root in range(size):
            for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
                sbuf = dev1.array(range(size), dtype=dtype, order=order)
                rbuf = dev2.full(size, -1, dtype=dtype, order=order)

                comm.Iallreduce([sbuf, mtype], [rbuf, mtype], op).Wait()

                actual_sbuf = np.array(range(size), dtype=dtype, order=order)
                actual_rbuf = np.full(size, -1, dtype=dtype, order=order)
                comm.Iallreduce([actual_sbuf, mtype],
                                [actual_rbuf, mtype], op).Wait()
                assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Ireduce_scatter_block(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        mtype = _get_type(dtype)

        for root in range(size):
            for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
                for rcnt in range(1, size):
                    sbuf = dev1.array([rank] * rcnt * size, dtype=dtype,
                                      order=order)
                    rbuf = dev2.full(rcnt, -1, dtype=dtype, order=order)
                    if op == MPI.PROD:
                        sbuf = dev1.array([rank + 1] * rcnt * size,
                                          dtype=dtype, order=order)
                    comm.Ireduce_scatter_block([sbuf, mtype],
                                               [rbuf, mtype],
                                               op=op).Wait()

                    actual_sbuf = np.array([rank] * rcnt * size, dtype=dtype,
                                           order=order)
                    actual_rbuf = np.full(rcnt, -1, dtype=dtype, order=order)
                    if op == MPI.PROD:
                        actual_sbuf = np.array([rank + 1] * rcnt * size,
                                               dtype=dtype, order=order)
                    comm.Ireduce_scatter_block([actual_sbuf, mtype],
                                               [actual_rbuf, mtype],
                                               op=op).Wait()
                    assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_ssend_recv(self, dev1, dev2, shape, dtype, order):
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
        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            comm.ssend(x, dest=peer)
        else:
            y = comm.recv(None, source=peer)
            self.assertFalse(_assert_array(y, desired))

    @parameterized.expand(_patterns)
    def test_sendrecv(self, dev1, dev2, shape, dtype, order):
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

        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            a = comm.sendrecv(x, peer, source=peer)
            if peer != MPI.PROC_NULL:
                self.assertFalse(_assert_array(a, x))
        else:
            y = dev2.array(desired, dtype=dtype, order=order)
            z = comm.sendrecv(y, peer, source=peer)
            if peer != MPI.PROC_NULL:
                self.assertFalse(_assert_array(z, y))

    @parameterized.expand(_patterns)
    def test_isend_irecv(self, dev1, dev2, shape, dtype, order):
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

        # recv buffer size
        if not (isinstance(shape, tuple) or isinstance(shape, list)):
            if shape >= 5 ** 5:
                shape = 5 ** 4

        desired = _get_sbuf(np, shape, dtype, order)
        comm.Bcast([desired, mtype], root=0)
        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            comm.isend(x, dest=peer).wait()
        else:
            y = comm.irecv(source=peer).wait()
            self.assertFalse(_assert_array(y, desired))

    @parameterized.expand(_patterns)
    def test_issend_recv(self, dev1, dev2, shape, dtype, order):
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
        if rank % 2 == 0:
            x = dev1.array(desired, dtype=dtype, order=order)
            comm.issend(x, dest=peer).wait()
        else:
            y = comm.recv(source=peer)
            self.assertFalse(_assert_array(y, desired))

    @parameterized.expand(_patterns)
    def test_bcast(self, dev1, dev2, shape, dtype, order):
        comm = self.COMM
        size = comm.Get_size()
        mtype = _get_type(dtype)

        desired = _get_sbuf(np, shape, dtype, order)
        comm.Bcast([desired, mtype], root=0)

        x = dev1.array(desired, dtype=dtype, order=order)
        comm.bcast(x, root=0)
        self.assertFalse(_assert_array(x, desired))

    @parameterized.expand(_patterns)
    def test_gather(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()

        for root in range(size):
            if rank == root:
                sbuf = dev1.full(root + 1, root, dtype=dtype, order=order)
                rbuf = dev2.full((size, root + 1), -1, dtype=dtype,
                                 order=order)
            else:
                sbuf = dev1.full(root + 1, root, dtype=dtype, order=order)
                rbuf = dev2.array([], dtype=dtype, order=order)
            rbuf = comm.gather(sbuf, root=root)

            if rank == root:
                desired = np.full((size, root + 1), root, dtype=dtype,
                                  order=order)
                assert_equal(_get_array(rbuf), desired)

    @parameterized.expand(_patterns)
    def test_scatter(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()

        for root in range(size):
            if rank == root:
                sbuf = dev1.full((size, size), root, dtype=dtype, order=order)
                rbuf = dev2.full(size, -1, dtype=dtype, order=order)
            else:
                sbuf = dev1.array([], dtype=dtype, order=order)
                rbuf = dev2.full(size, -1, dtype=dtype, order=order)
            rbuf = comm.scatter(sbuf, root=root)

            desired = np.full(size, root, dtype=dtype, order=order)
            assert_equal(_get_array(rbuf), desired)

    @parameterized.expand(_patterns)
    def test_allgather(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()

        for root in range(size):
            if rank == root:
                sbuf = dev1.full(root + 1, root, dtype=dtype, order=order)
                rbuf = dev2.full((size, root + 1), -1, dtype=dtype,
                                 order=order)
            else:
                sbuf = dev1.full(root + 1, root, dtype=dtype, order=order)
                rbuf = dev2.full((size, root + 1), -1, dtype=dtype,
                                 order=order)
            rbuf = comm.allgather(sbuf)

            desired = np.full((size, root + 1), root, dtype=dtype, order=order)
            assert_equal(_get_array(rbuf), desired)

    @parameterized.expand(_patterns)
    def test_alltoall(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()

        for root in range(size):
            sbuf = dev1.full((size, root + 1), root, dtype=dtype, order=order)
            rbuf = self.COMM.alltoall(sbuf)

            actual_sbuf = np.full((size, root + 1), root, dtype=dtype,
                                  order=order)
            actual_rbuf = self.COMM.alltoall(actual_sbuf)
            assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_reduce(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        size = comm.Get_size()

        for root in range(size):
            for op in (MPI.SUM, MPI.PROD):
                sbuf = dev1.array(range(size), dtype=dtype, order=order)
                rbuf = comm.reduce(sbuf, op, root)

                actual_sbuf = np.array(range(size), dtype=dtype, order=order)
                actual_rbuf = comm.reduce(actual_sbuf, op, root)
                assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_allreduce(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        size = comm.Get_size()

        for root in range(size):
            for op in (MPI.SUM, MPI.PROD):
                sbuf = dev1.array(range(size), dtype=dtype, order=order)
                rbuf = comm.allreduce(sbuf, op)

                actual_sbuf = np.array(range(size), dtype=dtype, order=order)
                actual_rbuf = comm.allreduce(actual_sbuf, op)
                assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Scan(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        mtype = _get_type(dtype)
        size = self.COMM.Get_size()
        for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
            sbuf = _get_sbuf(dev1, size, dtype, order)
            rbuf = _get_rbuf(dev2, size, dtype, order, dev1)
            comm.Scan([sbuf, mtype], [rbuf, mtype], op)

            actual_sbuf = _get_sbuf(np, size, dtype, order)
            actual_rbuf = _get_rbuf(np, size, dtype, order, np)
            comm.Scan([actual_sbuf, mtype], [actual_rbuf, mtype], op)
            assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Exscan(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        mtype = _get_type(dtype)
        size = self.COMM.Get_size()

        for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
            sbuf = _get_sbuf(dev1, size, dtype, order)
            rbuf = _get_rbuf(dev2, size, dtype, order, dev1)
            comm.Exscan([sbuf, mtype], [rbuf, mtype], op)

            actual_sbuf = _get_sbuf(np, size, dtype, order)
            actual_rbuf = _get_rbuf(np, size, dtype, order, np)
            comm.Exscan([actual_sbuf, mtype], [actual_rbuf, mtype], op)
            assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Iscan(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')

        comm = self.COMM
        mtype = _get_type(dtype)
        size = self.COMM.Get_size()

        for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
            sbuf = _get_sbuf(dev1, size, dtype, order)
            rbuf = _get_rbuf(dev2, size, dtype, order, dev1)
            comm.Iscan([sbuf, mtype], [rbuf, mtype], op).Wait()

            actual_sbuf = _get_sbuf(np, size, dtype, order)
            actual_rbuf = _get_rbuf(np, size, dtype, order, np)
            comm.Iscan([actual_sbuf, mtype], [actual_rbuf, mtype], op).Wait()
            assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Iexscan(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        mtype = _get_type(dtype)
        size = self.COMM.Get_size()

        for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
            sbuf = _get_sbuf(dev1, size, dtype, order)
            rbuf = _get_rbuf(dev2, size, dtype, order, dev1)
            comm.Iexscan([sbuf, mtype], [rbuf, mtype], op).Wait()

            actual_sbuf = _get_sbuf(np, size, dtype, order)
            actual_rbuf = _get_rbuf(np, size, dtype, order, np)
            comm.Iexscan([actual_sbuf, mtype], [actual_rbuf, mtype], op).Wait()
            assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_scan(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        size = self.COMM.Get_size()

        for op in (MPI.SUM, MPI.PROD):
            sbuf = _get_sbuf(dev1, size, dtype, order)
            rbuf = comm.scan(sbuf, op)

            actual_sbuf = _get_sbuf(np, size, dtype, order)
            actual_rbuf = comm.scan(actual_sbuf, op)
            assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_exscan(self, dev1, dev2, shape, dtype, order):
        if dtype in ('complex64', 'complex128', 'bool'):
            self.skipTest('Datatype is not testable')
        comm = self.COMM
        size = self.COMM.Get_size()

        for op in (MPI.SUM, MPI.PROD):
            sbuf = _get_sbuf(dev1, size, dtype, order)
            rbuf = comm.exscan(sbuf, op)

            actual_sbuf = _get_sbuf(np, size, dtype, order)
            actual_rbuf = comm.exscan(actual_sbuf, op)
            assert_equal(_get_array(rbuf), actual_rbuf)

    @parameterized.expand(_patterns)
    def test_Neighbor_allgather(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        for comm in create_topo_comms(self.COMM):
            rsize, ssize = get_neighbors_count(comm)
            mtype = _get_type(dtype)

            actual_sbuf = _get_sbuf(np, 3, dtype, order)
            actual_rbuf = _get_rbuf(np, (rsize, 3), dtype, order, np)
            comm.Neighbor_allgather([actual_sbuf, mtype], [actual_rbuf, mtype])

            sbuf = dev1.array(actual_sbuf, dtype=dtype, order=order)
            rbuf = _get_rbuf(dev2, (rsize, 3), dtype, order, dev1)
            comm.Neighbor_allgather([sbuf, mtype], [rbuf, mtype])
            assert_equal(_get_array(rbuf), actual_rbuf)

            comm.Free()

    @parameterized.expand(_patterns)
    def test_Neighbor_allgatherv(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        for comm in create_topo_comms(self.COMM):
            rsize, ssize = get_neighbors_count(comm)
            mtype = _get_type(dtype)
            actual_sbuf = _get_sbuf(np, 3, dtype, order)
            actual_rbuf = _get_rbuf(np, (rsize, 3), dtype, order, np)
            comm.Neighbor_allgatherv([actual_sbuf, mtype],
                                     [actual_rbuf, mtype])

            sbuf = dev1.array(actual_sbuf, dtype=dtype, order=order)
            rbuf = _get_rbuf(dev2, (rsize, 3), dtype, order, dev1)
            comm.Neighbor_allgatherv([sbuf, mtype], [rbuf, mtype])

            assert_equal(_get_array(rbuf), actual_rbuf)

            comm.Free()

    @parameterized.expand(_patterns)
    def test_Ineighbor_allgather(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        for comm in create_topo_comms(self.COMM):
            rsize, ssize = get_neighbors_count(comm)
            mtype = _get_type(dtype)

            actual_sbuf = _get_sbuf(np, 3, dtype, order)
            actual_rbuf = _get_rbuf(np, (rsize, 3), dtype, order, np)
            comm.Ineighbor_allgather([actual_sbuf, mtype],
                                     [actual_rbuf, mtype]).Wait()

            sbuf = dev1.array(actual_sbuf, dtype=dtype, order=order)
            rbuf = _get_rbuf(dev2, (rsize, 3), dtype, order, dev1)
            comm.Ineighbor_allgather([sbuf, mtype], [rbuf, mtype]).Wait()
            assert_equal(_get_array(rbuf), actual_rbuf)

            comm.Free()

    @parameterized.expand(_patterns)
    def test_Ineighbor_allgatherv(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        for comm in create_topo_comms(self.COMM):
            rsize, ssize = get_neighbors_count(comm)
            mtype = _get_type(dtype)

            actual_sbuf = _get_sbuf(np, 3, dtype, order)
            actual_rbuf = _get_rbuf(np, (rsize, 3), dtype, order, np)
            comm.Ineighbor_allgatherv([actual_sbuf, mtype],
                                      [actual_rbuf, mtype]).Wait()

            sbuf = dev1.array(actual_sbuf, dtype=dtype, order=order)
            rbuf = _get_rbuf(dev2, (rsize, 3), dtype, order, dev1)
            comm.Ineighbor_allgatherv([sbuf, mtype], [rbuf, mtype]).Wait()
            assert_equal(_get_array(rbuf), actual_rbuf)

            comm.Free()

    @parameterized.expand(_patterns)
    def test_Neighbor_alltoall(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        for comm in create_topo_comms(self.COMM):
            rsize, ssize = get_neighbors_count(comm)
            mtype = _get_type(dtype)

            actual_sbuf = _get_sbuf(np, (ssize, 3), dtype, order)
            actual_rbuf = _get_rbuf(np, (rsize, 3), dtype, order, np)
            comm.Neighbor_alltoall([actual_sbuf, mtype], [actual_rbuf, mtype])

            sbuf = dev1.array(actual_sbuf, dtype=dtype, order=order)
            rbuf = _get_rbuf(dev2, (rsize, 3), dtype, order, dev1)
            comm.Neighbor_alltoall([sbuf, mtype], [rbuf, mtype])
            assert_equal(_get_array(rbuf), actual_rbuf)

            comm.Free()

    @parameterized.expand(_patterns)
    def test_Neighbor_alltoallv(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        for comm in create_topo_comms(self.COMM):
            rsize, ssize = get_neighbors_count(comm)
            mtype = _get_type(dtype)
            actual_sbuf = _get_sbuf(np, (ssize, 3), dtype, order)
            actual_rbuf = _get_rbuf(np, (rsize, 3), dtype, order, np)
            comm.Neighbor_alltoallv([actual_sbuf, mtype], [actual_rbuf, mtype])

            sbuf = dev1.array(actual_sbuf, dtype=dtype, order=order)
            rbuf = _get_rbuf(dev2, (rsize, 3), dtype, order, dev1)
            comm.Neighbor_alltoallv([sbuf, mtype], [rbuf, mtype])
            assert_equal(_get_array(rbuf), actual_rbuf)

            comm.Free()

    @parameterized.expand(_patterns)
    def test_Neighbor_alltoallw(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        for comm in create_topo_comms(self.COMM):
            rsize, ssize = get_neighbors_count(comm)
            mtype = _get_type(dtype)
            sdt, rdt = mtype, mtype

            actual_sbuf = _get_sbuf(np, (ssize, 1), dtype, order)
            actual_rbuf = _get_rbuf(np, (rsize, 1), dtype, order, np)
            actual_sdsp = list(range(0, ssize * actual_sbuf.itemsize,
                                     actual_sbuf.itemsize))
            actual_rdsp = list(range(0, rsize * actual_rbuf.itemsize,
                                     actual_rbuf.itemsize))
            actual_smsg = [actual_sbuf, ([1] * ssize, actual_sdsp),
                           [sdt] * ssize]
            actual_rmsg = (actual_rbuf, ([1] * rsize, actual_rdsp),
                           [rdt] * rsize)
            comm.Neighbor_alltoallw(actual_smsg, actual_rmsg)

            sbuf = dev1.array(actual_sbuf, dtype=dtype, order=order)
            rbuf = _get_rbuf(dev2, (rsize, 1), dtype, order, dev1)
            sdsp = list(range(0, ssize * sbuf.itemsize, sbuf.itemsize))
            rdsp = list(range(0, rsize * rbuf.itemsize, rbuf.itemsize))
            smsg = [sbuf, ([1] * ssize, sdsp), [sdt] * ssize]
            rmsg = (rbuf, ([1] * rsize, rdsp), [rdt] * rsize)
            comm.Neighbor_alltoallw(smsg, rmsg)

            assert_equal(_get_array(rbuf), actual_rbuf)

            comm.Free()

    @parameterized.expand(_patterns)
    def test_Ineighbor_alltoall(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        for comm in create_topo_comms(self.COMM):
            rsize, ssize = get_neighbors_count(comm)
            mtype = _get_type(dtype)

            actual_sbuf = _get_sbuf(np, (ssize, 3), dtype, order)
            actual_rbuf = _get_rbuf(np, (rsize, 3), dtype, order, np)
            comm.Ineighbor_alltoall([actual_sbuf, mtype],
                                    [actual_rbuf, mtype]).Wait()

            sbuf = dev1.array(actual_sbuf, dtype=dtype, order=order)
            rbuf = _get_rbuf(dev2, (rsize, 3), dtype, order, dev1)
            comm.Ineighbor_alltoall([sbuf, mtype], [rbuf, mtype]).Wait()

            assert_equal(_get_array(rbuf), actual_rbuf)

            comm.Free()

    @parameterized.expand(_patterns)
    def test_Ineighbor_alltoallv(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        for comm in create_topo_comms(self.COMM):
            rsize, ssize = get_neighbors_count(comm)
            mtype = _get_type(dtype)

            actual_sbuf = _get_sbuf(np, (ssize, 3), dtype, order)
            actual_rbuf = _get_rbuf(np, (rsize, 3), dtype, order, np)
            comm.Ineighbor_alltoallv([actual_sbuf, mtype],
                                     [actual_rbuf, mtype]).Wait()

            sbuf = dev1.array(actual_sbuf, dtype=dtype, order=order)
            rbuf = _get_rbuf(dev2, (rsize, 3), dtype, order, dev1)
            comm.Ineighbor_alltoallv([sbuf, mtype], [rbuf, mtype]).Wait()

            assert_equal(_get_array(rbuf), actual_rbuf)

            comm.Free()

    @parameterized.expand(_patterns)
    def test_Ineighbor_alltoallw(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        for comm in create_topo_comms(self.COMM):
            rsize, ssize = get_neighbors_count(comm)
            mtype = _get_type(dtype)
            sbuf = _get_sbuf(dev1, (ssize, 1), dtype, order)
            rbuf = _get_rbuf(dev2, (rsize, 1), dtype, order, dev1)
            sdt, rdt = mtype, mtype
            sdsp = list(range(0, ssize * sbuf.itemsize, sbuf.itemsize))
            rdsp = list(range(0, rsize * rbuf.itemsize, rbuf.itemsize))
            smsg = [sbuf, ([1] * ssize, sdsp), [sdt] * ssize]
            rmsg = (rbuf, ([1] * rsize, rdsp), [rdt] * rsize)
            comm.Ineighbor_alltoallw(smsg, rmsg).Wait()

            actual_sbuf = _get_sbuf(np, (ssize, 1), dtype, order)
            actual_rbuf = _get_rbuf(np, (rsize, 1), dtype, order, np)
            actual_sdsp = list(range(0, ssize * actual_sbuf.itemsize,
                                     actual_sbuf.itemsize))
            actual_rdsp = list(range(0, rsize * actual_rbuf.itemsize,
                                     actual_rbuf.itemsize))
            actual_smsg = [actual_sbuf, ([1] * ssize, actual_sdsp),
                           [sdt] * ssize]
            actual_rmsg = (actual_rbuf, ([1] * rsize, actual_rdsp),
                           [rdt] * rsize)
            comm.Ineighbor_alltoallw(actual_smsg, actual_rmsg).Wait()
            (_get_array(rbuf), _get_array(actual_rbuf))

            comm.Free()


if __name__ == '__main__':
    unittest.main()
