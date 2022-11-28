from mpi4pyve import MPI
import mpiunittest as unittest
import nlcpy as vp
import numpy as np
import nlcpy_ndarray_wrapper

class BaseTestVAIBuf(object):

    COMM = MPI.COMM_NULL

    def testSendrecv(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        dest = (rank + 1) % size
        source = (rank - 1) % size
        if size < 1: return
        
        if rank == 0:
            sbuf = vp.arange(10)[2:]
            rbuf = vp.array(vp.zeros(8), dtype='int')
            self.COMM.Sendrecv(sbuf, 0, 0,
                               rbuf, 0, 0)
            self.assertEqual(np.allclose(sbuf, rbuf),True)
        else :
            pass

    def testSendRecv(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        sbuf = vp.arange(10)[2:]
        rbuf = vp.array(vp.zeros(8), dtype='int')
        if size < 2: return
        if rank == 0:
            self.COMM.Send(sbuf, dest=1)
        elif rank == 1:
            self.COMM.Recv(rbuf, source=0)
            self.assertEqual(np.allclose(sbuf, rbuf),True)
        else :
            pass
    
    def testPickledSendrecv(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        dest = (rank + 1) % size
        source = (rank - 1) % size
        if size < 1: return
        
        if rank == 0:
            sbuf = vp.arange(10)[2:]
            rbuf = vp.array(vp.zeros(8), dtype='int')
            rbuf = self.COMM.sendrecv(sbuf,dest=0,source=0)
            self.assertEqual(np.allclose(sbuf, rbuf),True)
        else :
            pass

    def testPickledSendRecv(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        sbuf = vp.arange(10)[2:]
        rbuf = vp.array(vp.zeros(8), dtype='int')
        if size < 2: return
        if rank == 0:
            self.COMM.send(sbuf, dest=1)
        elif rank == 1:
            rbuf = self.COMM.recv()
            self.assertEqual(np.allclose(sbuf, rbuf),True)
        else :
            pass

    def testVAIReadOnly(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        sbuf =  nlcpy_ndarray_wrapper.nlcpy_ndarray_wrapper(shape=(8))
        sbuf.fill(0)
        rbuf =  nlcpy_ndarray_wrapper.nlcpy_ndarray_wrapper(shape=(8))
        rbuf.fill(0)
        if size < 2: return
        if rank == 0:
            self.COMM.Send(sbuf, dest=1)
        elif rank == 1:
            self.COMM.Recv(rbuf, source=0)
            self.assertEqual(np.allclose(sbuf, rbuf),True)
        else :
            pass
        sbuf.set_read_only_flag(True)
        rbuf.set_read_only_flag(True)
        if rank == 0:
            #self.COMM.Send(sbuf, dest=1)
            pass
        elif rank == 1:
            with self.assertRaises(BufferError):
                self.COMM.Recv(rbuf, source=0)
        else :
            pass

class TestVAIBufSelf(BaseTestVAIBuf, unittest.TestCase):
    COMM = MPI.COMM_SELF

class TestVAIBufWorld(BaseTestVAIBuf, unittest.TestCase):
    COMM = MPI.COMM_WORLD

if __name__ == '__main__':
    unittest.main()
