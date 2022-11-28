from mpi4pyve import MPI
import mpiunittest as unittest
import nlcpy as vp
import numpy as np
import nlcpy_ndarray_wrapper

class BaseTestVAIBuf(object):

    COMM = MPI.COMM_NULL

    def testSendRecvbool(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        sbuf = vp.array([True, True, True], dtype='bool')
        rbuf = vp.array([False, False, False], dtype='bool')
        if size < 2: return
        if rank == 0:
            self.COMM.Send(sbuf, dest=1)
        elif rank == 1:
            self.COMM.Recv(rbuf, source=0)
            self.assertEqual(np.allclose(sbuf, rbuf),True)
        else :
            pass

    def testSendRecvMPIBOOL(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        sbuf = vp.array([True, True, True], dtype='bool')
        rbuf = vp.array([False, False, False], dtype='bool')
        if size < 2: return
        if rank == 0:
            self.COMM.Send([sbuf,MPI.BOOL], dest=1)
        elif rank == 1:
            self.COMM.Recv([rbuf,MPI.BOOL], source=0)
            self.assertEqual(np.allclose(sbuf, rbuf),True)
        else :
            pass

    def testPickledbool(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        sbuf = vp.array([True, True, True], dtype='bool')
        rbuf = vp.array([False, False, False], dtype='bool')
        if size < 2: return
        if rank == 0:
            self.COMM.send(sbuf, dest=1)
        elif rank == 1:
            rbuf = self.COMM.recv()
            self.assertEqual(np.allclose(sbuf, rbuf),True)
        else :
            pass
    
class TestVAIBufSelf(BaseTestVAIBuf, unittest.TestCase):
    COMM = MPI.COMM_SELF

class _TestVAIBufWorld(BaseTestVAIBuf, unittest.TestCase):
    COMM = MPI.COMM_WORLD

if __name__ == '__main__':
    unittest.main()
