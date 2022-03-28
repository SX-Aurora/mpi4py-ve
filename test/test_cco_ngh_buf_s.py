from mpi4pyve import MPI
import mpiunittest as unittest
#import arrayimpl
import nlcpy_only_arrayimpl as arrayimpl

def create_topo_comms(comm):
    size = comm.Get_size()
    rank = comm.Get_rank()
    # Cartesian
    n = int(size**1/2.0)
    m = int(size**1/3.0)
    if m*m*m == size:
        dims = [m, m, m]
    elif n*n == size:
        dims = [n, n]
    else:
        dims = [size]
    periods = [True] * len(dims)
    yield comm.Create_cart(dims, periods=periods)
    # Graph
    index, edges = [0], []
    for i in range(size):
        pos = index[-1]
        index.append(pos+2)
        edges.append((i-1)%size)
        edges.append((i+1)%size)
    yield comm.Create_graph(index, edges)
    # Dist Graph
    sources = [(rank-2)%size, (rank-1)%size]
    destinations = [(rank+1)%size, (rank+2)%size]
    yield comm.Create_dist_graph_adjacent(sources, destinations)

def get_neighbors_count(comm):
    topo = comm.Get_topology()
    if topo == MPI.CART:
        ndim = comm.Get_dim()
        return 2*ndim, 2*ndim
    if topo == MPI.GRAPH:
        rank = comm.Get_rank()
        nneighbors = comm.Get_neighbors_count(rank)
        return nneighbors, nneighbors
    if topo == MPI.DIST_GRAPH:
        indeg, outdeg, w = comm.Get_dist_neighbors_count()
        return indeg, outdeg
    return 0, 0

def have_feature():
    cartcomm = MPI.COMM_SELF.Create_cart([1], periods=[0])
    try:
        cartcomm.neighbor_allgather(None)
        return True
    except NotImplementedError:
        return False
    finally:
        cartcomm.Free()

@unittest.skipIf(not have_feature(), 'mpi-neighbor')
class BaseTestCCONghBuf(object):

    COMM = MPI.COMM_NULL

    def test_neighbor_allgather(self):
        for comm in create_topo_comms(self.COMM):
            rsize, ssize = get_neighbors_count(comm)
            for array in arrayimpl.ArrayTypes:
                for typecode in arrayimpl.TypeMap:
                    for v in range(3):
                        sbuf = array( v, typecode, 3)
                        rbuf=comm.neighbor_allgather(sbuf.as_raw())
            comm.Free()

    def test_neighbor_alltoall(self):
        for comm in create_topo_comms(self.COMM):
            rsize, ssize = get_neighbors_count(comm)
            for array in arrayimpl.ArrayTypes:
                for typecode in arrayimpl.TypeMap:
                    for v in range(3):
                        sbuf = array( v, typecode, (ssize, 3))
                        rbuf=comm.neighbor_alltoall(sbuf.as_raw())
                        sbuf = array( v, typecode, (ssize, 3))
                        rbuf=comm.neighbor_alltoall(sbuf.as_raw())
            comm.Free()

class TestCCONghBufSelf(BaseTestCCONghBuf, unittest.TestCase):
    COMM = MPI.COMM_SELF

class TestCCONghBufWorld(BaseTestCCONghBuf, unittest.TestCase):
    COMM = MPI.COMM_WORLD

class TestCCONghBufSelfDup(TestCCONghBufSelf):
    def setUp(self):
        self.COMM = MPI.COMM_SELF.Dup()
    def tearDown(self):
        self.COMM.Free()

class TestCCONghBufWorldDup(TestCCONghBufWorld):
    def setUp(self):
        self.COMM = MPI.COMM_WORLD.Dup()
    def tearDown(self):
        self.COMM.Free()


name, version = MPI.get_vendor()
if name == 'Open MPI' and version < (1,8,4):
    _create_topo_comms = create_topo_comms
    def create_topo_comms(comm):
        for c in _create_topo_comms(comm):
            if c.size * 2 < sum(c.degrees):
                c.Free(); continue
            yield c


if __name__ == '__main__':
    unittest.main()
