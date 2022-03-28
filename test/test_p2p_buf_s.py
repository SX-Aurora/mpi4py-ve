from mpi4pyve import MPI
import mpiunittest as unittest
import nlcpy_only_arrayimpl as arrayimpl


class BaseTestP2PBuf_s(object):

    COMM = MPI.COMM_NULL

    def test_sendrecv(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        dest = (rank + 1) % size
        source = (rank - 1) % size
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for s in range(0, size):
                    sbuf = array( s, typecode, s)
                    rbuf=self.COMM.sendrecv(sbuf.as_raw(), dest,   0,
                                       None, source, 0)
                    for value in rbuf[:-1]:
                        self.assertEqual(value, s)

    def test_send_recv(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for s in range(0, size):
                    #
                    sbuf = array( s, typecode, s)
                    # rbuf = array(-1, typecode, s)
                    mem  = array( 0, typecode, 2*(s+MPI.BSEND_OVERHEAD)).as_raw()
                    if size == 1:
                        MPI.Attach_buffer(mem)
                        rbuf = sbuf
                        MPI.Detach_buffer()
                    elif rank == 0:
                        MPI.Attach_buffer(mem)
                        self.COMM.ibsend(sbuf.as_raw(), 1, 0).Wait()
                        self.COMM.bsend(sbuf.as_raw(), 1, 0)
                        MPI.Detach_buffer()
                        self.COMM.send(sbuf.as_raw(), 1, 0)
                        self.COMM.ssend(sbuf.as_raw(), 1, 0)
                        rbuf=self.COMM.recv(None,  1, 0)
                        rbuf=self.COMM.recv(None,  1, 0)
                        rbuf=self.COMM.recv(None, 1, 0)
                        rbuf=self.COMM.recv(None, 1, 0)
                    elif rank == 1:
                        rbuf=self.COMM.recv(None, 0, 0)
                        rbuf=self.COMM.recv(None, 0, 0)
                        rbuf=self.COMM.recv(None, 0, 0)
                        rbuf=self.COMM.recv(None, 0, 0)
                        MPI.Attach_buffer(mem)
                        self.COMM.ibsend(sbuf.as_raw(), 0, 0).Wait()
                        self.COMM.bsend(sbuf.as_raw(), 0, 0)
                        MPI.Detach_buffer()
                        self.COMM.send(sbuf.as_raw(), 0, 0)
                        self.COMM.ssend(sbuf.as_raw(), 0, 0)
                    else:
                        rbuf = sbuf
                    for value in rbuf:
                        self.assertEqual(value, s)
                    
                    rank = self.COMM.Get_rank()
                    sbuf = array( s, typecode, s)
                    rbuf = array(-1, typecode, s)
                    rreq = self.COMM.irecv(rbuf.as_raw(), rank, 0)
                    self.COMM.Rsend(sbuf.as_raw(), rank, 0)
                    rreq.Wait()
                    for value in rbuf:
                        self.assertEqual(value, s)
                    rbuf = array(-1, typecode, s)
                    rreq = self.COMM.irecv(rbuf.as_raw(), rank, 0)
                    self.COMM.Irsend(sbuf.as_raw(), rank, 0).Wait()
                    rreq.Wait()
                    for value in rbuf:
                        self.assertEqual(value, s)

    def test_sendrecv_exception(self):
        comm = MPI.COMM_WORLD.Dup()
        size = comm.Get_size()
        rank = comm.Get_rank()
        dest = (rank + 1) % size
        source = (rank - 1) % size
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for s in range(0, size):
                    sbuf = array( s, typecode, s)
                    rbuf = array(-1, typecode, s)
                    with self.assertRaises( Exception, msg='Message truncated' ):
                        rbuf=comm.sendrecv(sbuf.as_raw(), dest,   0,
                                           rbuf.as_raw(), source, 0)

    def test_Sendrecv_exception_1(self):
        comm = MPI.COMM_WORLD.Dup()
        size = comm.Get_size()
        rank = comm.Get_rank()
        dest = (rank + 1) % size
        source = (rank - 1) % size
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for s in range(0, size):
                    sbuf = array( s, typecode, s)
                    rbuf = array(-1, typecode, s)
                    with self.assertRaises( TypeError, msg='expecting buffer or list/tuple' ):
                        comm.Sendrecv(sbuf.as_mpi(), dest,   0,
                                      None, source, 0)

    def test_Sendrecv_exception_2(self):
        comm = MPI.COMM_WORLD.Dup()
        size = comm.Get_size()
        rank = comm.Get_rank()
        dest = (rank + 1) % size
        source = (rank - 1) % size
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for s in range(0, size):
                    sbuf = array( s, typecode, s)
                    rbuf = array(-1, typecode, s)
                    with self.assertRaises( TypeError, msg='expecting buffer or list/tuple' ):
                        comm.Sendrecv(sbuf.as_mpi(), dest,   sendtag=0,
                                      source=source, recvtag=0)

    def test_send_recv_exception(self):
        comm = MPI.COMM_WORLD.Dup()
        size = comm.Get_size()
        rank = comm.Get_rank()
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                sbuf = array( 1, typecode, 1)
                rbuf = array(-1, typecode, 1)
                if rank == 0:
                    comm.send(sbuf.as_raw(), dest=1)
                elif rank == 1:
                    with self.assertRaises( Exception, msg='Message truncated' ):
                        rbuf = comm.recv(rbuf.as_raw())

    def test_send_irecv_exception(self):
        comm = MPI.COMM_WORLD.Dup()
        size = comm.Get_size()
        rank = comm.Get_rank()
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                sbuf = array( 1, typecode, 1)
                rbuf = array(-1, typecode, 1)
                if rank == 0:
                    comm.send(sbuf.as_raw(), dest=1)
                elif rank == 1:
                    with self.assertRaises( Exception, msg='Message truncated' ):
                        rreq = comm.irecv(rbuf.as_raw())
                        rreq.wait()

    def testProcNull(self):
        comm = self.COMM
        #
        comm.sendrecv(None, MPI.PROC_NULL, 0,
                      None, MPI.PROC_NULL, 0)
        #
        comm.send (None, MPI.PROC_NULL)
        comm.isend (None, MPI.PROC_NULL).Wait()
        #
        comm.ssend(None, MPI.PROC_NULL)
        comm.issend(None, MPI.PROC_NULL).Wait()
        #
        buf = MPI.Alloc_mem(MPI.BSEND_OVERHEAD)
        MPI.Attach_buffer(buf)
        comm.bsend(None, MPI.PROC_NULL)
        comm.ibsend(None, MPI.PROC_NULL).Wait()
        MPI.Detach_buffer()
        MPI.Free_mem(buf)
        #
        comm.Rsend(None, MPI.PROC_NULL)
        comm.Irsend(None, MPI.PROC_NULL).Wait()
        #
        comm.recv (None, MPI.PROC_NULL)
        comm.irecv(None, MPI.PROC_NULL).Wait()

    def test_probe(self):
        comm = self.COMM.Dup()
        try:
            request = comm.issend(None, comm.rank, 123)
            self.assertTrue(request)
            status = MPI.Status()
            comm.probe(MPI.ANY_SOURCE, MPI.ANY_TAG, status)
            self.assertEqual(status.source, comm.rank)
            self.assertEqual(status.tag, 123)
            self.assertTrue(request)
            flag = request.Test()
            self.assertTrue(request)
            self.assertFalse(flag)
            comm.recv(None, comm.rank, 123)
            self.assertTrue(request)
            #flag = request.Test()
            flag = False
            while not flag:
                flag = request.Test()
            self.assertFalse(request)
            self.assertTrue(flag)
        finally:
            comm.Free()

    @unittest.skipMPI('MPICH1')
    @unittest.skipMPI('LAM/MPI')
    def test_probe_cancel(self):
        comm = self.COMM.Dup()
        try:
            request = comm.issend(None, comm.rank, 123)
            status = MPI.Status()
            comm.probe(MPI.ANY_SOURCE, MPI.ANY_TAG, status)
            self.assertEqual(status.source, comm.rank)
            self.assertEqual(status.tag, 123)
            request.Cancel()
            self.assertTrue(request)
            status = MPI.Status()
            request.Get_status(status)
            cancelled = status.Is_cancelled()
            if not cancelled:
                comm.recv(None, comm.rank, 123)
                request.Wait()
            else:
                request.Free()
        finally:
            comm.Free()

    def test_iprobe(self):
        comm = self.COMM.Dup()
        try:
            f = comm.iprobe()
            self.assertFalse(f)
            f = comm.iprobe(MPI.ANY_SOURCE)
            self.assertFalse(f)
            f = comm.iprobe(MPI.ANY_SOURCE, MPI.ANY_TAG)
            self.assertFalse(f)
            status = MPI.Status()
            f = comm.iprobe(MPI.ANY_SOURCE, MPI.ANY_TAG, status)
            self.assertFalse(f)
            self.assertEqual(status.source, MPI.ANY_SOURCE)
            self.assertEqual(status.tag,    MPI.ANY_TAG)
            self.assertEqual(status.error,  MPI.SUCCESS)
        finally:
            comm.Free()


class TestP2PBuf_sSelf(BaseTestP2PBuf_s, unittest.TestCase):
    COMM = MPI.COMM_SELF

@unittest.skip('necmpi')
class TestP2PBuf_sWorld(BaseTestP2PBuf_s, unittest.TestCase):
    COMM = MPI.COMM_WORLD

class TestP2PBuf_sSelfDup(TestP2PBuf_sSelf):
    def setUp(self):
        self.COMM = MPI.COMM_SELF.Dup()
    def tearDown(self):
        self.COMM.Free()

@unittest.skipMPI('openmpi(<1.4.0)', MPI.Query_thread() > MPI.THREAD_SINGLE)
@unittest.skip('necmpi')
class TestP2PBuf_sWorldDup(TestP2PBuf_sWorld):
    def setUp(self):
        self.COMM = MPI.COMM_WORLD.Dup()
    def tearDown(self):
        self.COMM.Free()


if __name__ == '__main__':
    unittest.main()
