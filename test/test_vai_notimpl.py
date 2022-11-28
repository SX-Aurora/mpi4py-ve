from mpi4pyve import MPI
import mpiunittest as unittest
import nlcpy
import numpy

class TestVAI_Notimpl(unittest.TestCase):

    def testRestriction(self):
        src = nlcpy.arange(10)
        dst = nlcpy.arange(10)
        np_src = numpy.arange(10)
        np_dst = numpy.arange(10)

        with self.assertRaises(NotImplementedError):
            MPI.Attach_buffer(src)
        
        comm = MPI.COMM_SELF
        with self.assertRaises(NotImplementedError):
            comm.Bsend(src, 0)
        with self.assertRaises(NotImplementedError):
            comm.Ibsend(src, 0)
        with self.assertRaises(NotImplementedError):
            comm.Bsend_init(src, 0)
        with self.assertRaises(NotImplementedError):
            comm.bsend(src, 0)
        with self.assertRaises(NotImplementedError):
            comm.ibsend(src, 0)

        op = MPI.SUM
        with self.assertRaises(NotImplementedError):
            op.Reduce_local(src, np_src)
        with self.assertRaises(NotImplementedError):
            op.Reduce_local(np_src, src)
  
        data_type = MPI.INT 
        with self.assertRaises(NotImplementedError):
            data_type.Pack_external('external32',src, np_src, 0)
        with self.assertRaises(NotImplementedError):
            data_type.Pack_external('external32',np_src, src, 0)
        with self.assertRaises(NotImplementedError):
            data_type.Unpack_external('external32',src, 0, np_src)
        with self.assertRaises(NotImplementedError):
            data_type.Unpack_external('external32',np_src, 0, src)

if __name__ == '__main__':
    unittest.main()
