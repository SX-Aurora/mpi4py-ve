from mpi4pyve import MPI
import mpiunittest as unittest
import nlcpy_only_arrayimpl as arrayimpl
#import arrayimpl

from functools import reduce
prod = lambda sequence,start=1: reduce(lambda x, y: x*y, sequence, start)

def maxvalue(a):
    try:
        typecode = a.typecode
    except AttributeError:
        typecode = a.dtype.char
    if typecode == ('f'):
        return 1e30
    elif typecode == ('d'):
        return 1e300
    else:
        return 2 ** (a.itemsize * 7) - 1


class BaseTestCCOBuf(object):

    COMM = MPI.COMM_NULL

    def test_barrier(self):
        self.COMM.barrier()

    def test_bcast(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for root in range(size):
                    if rank == root:
                        buf = array(root, typecode, root)
                    else:
                        buf = array(  -1, typecode, root)
                    self.COMM.bcast(buf.as_raw(), root=root)
                    for value in buf:
                        self.assertEqual(value, root)

    def test_gather(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for root in range(size):
                    sbuf = array(root, typecode, root+1)
                    #if rank == root:
                    #    rbuf = array(-1, typecode, (size,root+1))
                    #else:
                    #    rbuf = array([], typecode)
                    rbuf=self.COMM.gather(sbuf.as_raw(), 
                                     root=root)
                    #if rank == root:
                    #    for value in rbuf.flat:
                    #        self.assertEqual(value, root)

    def test_scatter(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for root in range(size):
                    if rank == root:
                        sbuf = array(root, typecode, (size, size))
                    else:
                        sbuf = array([], typecode)
                    rbuf=self.COMM.scatter(sbuf.as_raw(), 
                                      root=root)
                    #for value in rbuf:
                    #    self.assertEqual(value, root)

    def test_allgather(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for root in range(size):
                    sbuf = array(root, typecode, root+1)
                    #rbuf = array(  -1, typecode, (size, root+1))
                    rbuf= self.COMM.allgather(sbuf.as_raw())
                    #for value in rbuf.flat:
                    #    self.assertEqual(value, root)

    def test_alltoall(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for root in range(size):
                    sbuf = array(root, typecode, (size, root+1))
                    rbuf=self.COMM.alltoall(sbuf.as_raw())
                    #for value in rbuf.flat:
                    #    self.assertEqual(value, root)

    def assertAlmostEqual(self, first, second):
        num = float(float(second-first))
        den = float(second+first)/2 or 1.0
        if (abs(num/den) > 1e-2):
            raise self.failureException('%r != %r' % (first, second))

    def test_reduce(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for root in range(size):
                    for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
                        sbuf = array(range(size), typecode)
                        rbuf=self.COMM.reduce(sbuf.as_raw(),
                                         op, root)
                        max_val = maxvalue(rbuf)
                        for i, value in enumerate(rbuf):
                            if rank != root:
                                self.assertEqual(value, -1)
                                continue
                            if op == MPI.SUM:
                                if (i * size) < max_val:
                                    self.assertAlmostEqual(value, i*size)
                            elif op == MPI.PROD:
                                if (i ** size) < max_val:
                                    self.assertAlmostEqual(value, i**size)
                            elif op == MPI.MAX:
                                self.assertEqual(value, i)
                            elif op == MPI.MIN:
                                self.assertEqual(value, i)

    def test_allreduce(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for op in (MPI.SUM, MPI.MAX, MPI.MIN, MPI.PROD):
                    sbuf = array(range(size), typecode)
                    rbuf=self.COMM.allreduce(sbuf.as_raw(),
                                        op)
                    max_val = maxvalue(rbuf)
                    for i, value in enumerate(rbuf):
                        if op == MPI.SUM:
                            if (i * size) < max_val:
                                self.assertAlmostEqual(value, i*size)
                        elif op == MPI.PROD:
                            if (i ** size) < max_val:
                                self.assertAlmostEqual(value, i**size)
                        elif op == MPI.MAX:
                            self.assertEqual(value, i)
                        elif op == MPI.MIN:
                            self.assertEqual(value, i)

    def test_scan(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        # --
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
                    sbuf = array(range(size), typecode)
                    rbuf=self.COMM.scan(sbuf.as_raw(),
                                   op)
                    max_val = maxvalue(rbuf)
                    for i, value in enumerate(rbuf):
                        if op == MPI.SUM:
                            if (i * (rank + 1)) < max_val:
                                self.assertAlmostEqual(value, i * (rank + 1))
                        elif op == MPI.PROD:
                            if (i ** (rank + 1)) < max_val:
                                self.assertAlmostEqual(value, i ** (rank + 1))
                        elif op == MPI.MAX:
                            self.assertEqual(value, i)
                        elif op == MPI.MIN:
                            self.assertEqual(value, i)

    def test_exscan(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
                    sbuf = array(range(size), typecode)
                    try:
                        rbuf=self.COMM.exscan(sbuf.as_raw(),
                                         op)
                    except NotImplementedError:
                        self.skipTest('mpi-exscan')
                    if rank == 1:
                        for i, value in enumerate(rbuf):
                            self.assertEqual(value, i)
                    elif rank > 1:
                        max_val = maxvalue(rbuf)
                        for i, value in enumerate(rbuf):
                            if op == MPI.SUM:
                                if (i * rank) < max_val:
                                    self.assertAlmostEqual(value, i * rank)
                            elif op == MPI.PROD:
                                if (i ** rank) < max_val:
                                    self.assertAlmostEqual(value, i ** rank)
                            elif op == MPI.MAX:
                                self.assertEqual(value, i)
                            elif op == MPI.MIN:
                                self.assertEqual(value, i)

class BaseTestCCOBufInplace(object):

    def test_gather(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for root in range(size):
                    count = root+3
                    if rank == root:
                        sbuf = MPI.IN_PLACE
                        buf = array(-1, typecode, (size, count))
                        #buf.flat[(rank*count):((rank+1)*count)] = \
                        #    array(root, typecode, count)
                        s, e = rank*count, (rank+1)*count
                        for i in range(s, e): buf.flat[i] = root
                        #rbuf = buf.as_raw()
                    else:
                        buf = array(root, typecode, count)
                        sbuf = buf.as_raw()
                        #rbuf = None
                    rbuf=self.COMM.gather(sbuf, root=root)
                    for value in buf.flat:
                        self.assertEqual(value, root)

    @unittest.skipMPI('msmpi(==10.0.0)')
    def test_scatter(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for root in range(size):
                    for count in range(1, 10):
                        if rank == root:
                            buf = array(root, typecode, (size, count))
                            sbuf = buf.as_raw()
                        #    rbuf = MPI.IN_PLACE
                        else:
                            buf = array(-1, typecode, count)
                            sbuf = None
                        #    rbuf = buf.as_raw()
                        rbuf=self.COMM.scatter(sbuf, root=root)
                        #for value in buf.flat:
                        #    self.assertEqual(value, root)

    def test_allgather(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for count in range(1, 10):
                    #s, e = rank*count, (rank+1)*count
                    #for i in range(s, e): buf.flat[i] = count
                    buf=self.COMM.allgather(MPI.IN_PLACE)
                    #for value in buf.flat:
                    #    self.assertEqual(value, count)

    def assertAlmostEqual(self, first, second):
        num = float(float(second-first))
        den = float(second+first)/2 or 1.0
        if (abs(num/den) > 1e-2):
            raise self.failureException('%r != %r' % (first, second))

    def test_reduce(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for root in range(size):
                    for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
                        count = size
                        if rank == root:
                            buf  = array(range(size), typecode)
                            sbuf = MPI.IN_PLACE
                            #rbuf = buf.as_raw()
                        else:
                            buf  = array(range(size), typecode)
                            buf2 = array(range(size), typecode)
                            sbuf = buf.as_raw()
                            #rbuf = buf2.as_raw()
                        rbuf=self.COMM.reduce(sbuf, op, root)
                        if rank == root:
                            max_val = maxvalue(buf)
                            for i, value in enumerate(buf):
                                if op == MPI.SUM:
                                    if (i * size) < max_val:
                                        self.assertAlmostEqual(value, i*size)
                                elif op == MPI.PROD:
                                    if (i ** size) < max_val:
                                        self.assertAlmostEqual(value, i**size)
                                elif op == MPI.MAX:
                                    self.assertEqual(value, i)
                                elif op == MPI.MIN:
                                    self.assertEqual(value, i)

    def test_allreduce(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for op in (MPI.SUM, MPI.MAX, MPI.MIN, MPI.PROD):
                    buf = array(range(size), typecode)
                    sbuf = MPI.IN_PLACE
                    buf=self.COMM.allreduce(sbuf, op)
                    #max_val = maxvalue(buf)
                    #for i, value in enumerate(buf):
                    #    if op == MPI.SUM:
                    #        if (i * size) < max_val:
                    #            self.assertAlmostEqual(value, i*size)
                    #    elif op == MPI.PROD:
                    #        if (i ** size) < max_val:
                    #            self.assertAlmostEqual(value, i**size)
                    #    elif op == MPI.MAX:
                    #        self.assertEqual(value, i)
                    #    elif op == MPI.MIN:
                    #        self.assertEqual(value, i)


    @unittest.skipMPI('openmpi(<=1.8.4)')
    def test_scan(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        # --
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
                    #buf = array(range(size), typecode)
                    buf=self.COMM.scan(MPI.IN_PLACE,
                                   op)
                    #max_val = maxvalue(buf)
                    #for i, value in enumerate(buf):
                    #    if op == MPI.SUM:
                    #        if (i * (rank + 1)) < max_val:
                    #            self.assertAlmostEqual(value, i * (rank + 1))
                    #    elif op == MPI.PROD:
                    #        if (i ** (rank + 1)) < max_val:
                    #            self.assertAlmostEqual(value, i ** (rank + 1))
                    #    elif op == MPI.MAX:
                    #        self.assertEqual(value, i)
                    #    elif op == MPI.MIN:
                    #        self.assertEqual(value, i)

    @unittest.skipMPI('openmpi(<=1.8.4)')
    @unittest.skipMPI('msmpi(<=4.2.0)')
    def test_exscan(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
                    #buf = array(range(size), typecode)
                    try:
                        buf=self.COMM.exscan(MPI.IN_PLACE,
                                         op)
                    except NotImplementedError:
                        self.skipTest('mpi-exscan')
                    if rank == 1:
                        for i, value in enumerate(buf):
                            self.assertEqual(value, i)
                    elif rank > 1:
                        max_val = maxvalue(buf)
                        for i, value in enumerate(buf):
                            if op == MPI.SUM:
                                if (i * rank) < max_val:
                                    self.assertAlmostEqual(value, i * rank)
                            elif op == MPI.PROD:
                                if (i ** rank) < max_val:
                                    self.assertAlmostEqual(value, i ** rank)
                            elif op == MPI.MAX:
                                self.assertEqual(value, i)
                            elif op == MPI.MIN:
                                self.assertEqual(value, i)

class TestReduceLocal(unittest.TestCase):

    @unittest.skip('necmpi')
    def test_reduce_local(self):
        for array in arrayimpl.ArrayTypes:
            for typecode in arrayimpl.TypeMap:
                for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
                    size = 5
                    sbuf = array(range(1,size+1), typecode)
                    rbuf = array(range(0,size+0), typecode)
                    try:
                        op.reduce_local(sbuf.as_raw(), rbuf.as_raw())
                    except NotImplementedError:
                        self.skipTest('mpi-op-reduce_local')
                    for i, value in enumerate(rbuf):
                        self.assertEqual(sbuf[i], i+1)
                        if op == MPI.SUM:
                            self.assertAlmostEqual(value, i+(i+1))
                        elif op == MPI.PROD:
                            self.assertAlmostEqual(value, i*(i+1))
                        elif op == MPI.MAX:
                            self.assertEqual(value, i+1)
                        elif op == MPI.MIN:
                            self.assertEqual(value, i)
        for array in arrayimpl.ArrayTypes:
            for op in (MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN):
                sbuf = array(range(3), "i")
                rbuf = array(range(3), "i")
                def f(): op.reduce_local(sbuf.as_raw(),
                                         rbuf.as_raw())
                self.assertRaises(ValueError, f)
                def f(): op.reduce_local([sbuf.as_raw(), 1, MPI.INT],
                                         [rbuf.as_raw(), 1, MPI.SHORT])
                self.assertRaises(ValueError, f)


class TestCCOBufSelf(BaseTestCCOBuf, unittest.TestCase):
    COMM = MPI.COMM_SELF

@unittest.skip('necmpi')
class TestCCOBufWorld(BaseTestCCOBuf, unittest.TestCase):
    COMM = MPI.COMM_WORLD

@unittest.skipMPI('MPICH1')
@unittest.skipMPI('LAM/MPI')
@unittest.skipIf(MPI.IN_PLACE == MPI.BOTTOM, 'mpi-in-place')
class TestCCOBufInplaceSelf(BaseTestCCOBufInplace, unittest.TestCase):
    COMM = MPI.COMM_SELF

@unittest.skipMPI('MPICH1')
@unittest.skipMPI('LAM/MPI')
@unittest.skipIf(MPI.IN_PLACE == MPI.BOTTOM, 'mpi-in-place')
@unittest.skip('necmpi')
class TestCCOBufInplaceWorld(BaseTestCCOBufInplace, unittest.TestCase):
    COMM = MPI.COMM_WORLD

class TestCCOBufSelfDup(TestCCOBufSelf):
    def setUp(self):
        self.COMM = MPI.COMM_SELF.Dup()
    def tearDown(self):
        self.COMM.Free()

@unittest.skipMPI('openmpi(<1.4.0)', MPI.Query_thread() > MPI.THREAD_SINGLE)
class TestCCOBufWorldDup(TestCCOBufWorld):
    def setUp(self):
        self.COMM = MPI.COMM_WORLD.Dup()
    def tearDown(self):
        self.COMM.Free()


if __name__ == '__main__':
    unittest.main()
