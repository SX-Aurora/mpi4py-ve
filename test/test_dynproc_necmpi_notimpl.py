from mpi4pyve import MPI
import mpiunittest as unittest

class TestDPM(unittest.TestCase):

    def testRestriction(self):
        with self.assertRaises(NotImplementedError):
            MPI.Open_port()
        with self.assertRaises(NotImplementedError):
            MPI.Close_port(None)
        with self.assertRaises(NotImplementedError):
            MPI.Publish_name(None, None)
        with self.assertRaises(NotImplementedError):
            MPI.Lookup_name(None)
        with self.assertRaises(NotImplementedError):
            MPI.Unpublish_name(None, None)
        comm = MPI.COMM_SELF
        with self.assertRaises(NotImplementedError):
            comm.Accept(None)
        with self.assertRaises(NotImplementedError):
            comm.Connect(None)
        with self.assertRaises(NotImplementedError):
            comm.Join(None)

if __name__ == '__main__':
    unittest.main()
