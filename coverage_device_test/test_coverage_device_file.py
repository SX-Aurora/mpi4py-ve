import unittest  # NOQA
from unittest import TestCase  # NOQA
from parameterized import parameterized  # NOQA
from mpi4pyve import MPI  # NOQA
from mpi4pyve import util  # NOQA
from numpy.testing import (
    assert_equal,
)
import sys, os, tempfile  # NOQA
import  coverage_device_util  # NOQA
from coverage_device_util import (
    _patterns, _get_array, _get_type, _get_sbuf, _get_rbuf, vp, np, IS_MULTI_HOST
)


class TestCoverageDeviceFile(unittest.TestCase):

    COMM = MPI.COMM_WORLD
    FILE = MPI.FILE_NULL

    prefix = 'mpi4pyve'
    tmpname = './tmp'

    def setUp(self):
        comm = self.COMM
        fname = None
        if comm.Get_rank() == 0:
            if not os.path.exists(self.tmpname):
                try:
                    os.mkdir(self.tmpname)
                except OSError as e:
                    if e.errno != 17:  # not File exists
                        raise
                    pass
            fd, fname = tempfile.mkstemp(prefix=self.prefix, dir=self.tmpname)
            os.close(fd)
        fname = comm.bcast(fname, 0)
        amode = MPI.MODE_RDWR | MPI.MODE_CREATE
        amode |= MPI.MODE_DELETE_ON_CLOSE
        amode |= MPI.MODE_UNIQUE_OPEN
        info = MPI.INFO_NULL
        try:
            self.FILE = MPI.File.Open(comm, fname, amode, info)
        except Exception:
            if comm.Get_rank() == 0:
                os.remove(fname)
            raise

    def tearDown(self):
        if self.FILE:
            self.FILE.Close()
        self.COMM.Barrier()

    @parameterized.expand(_patterns)
    def test_ReadWriteAt(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()

        actual = _get_sbuf(np, shape, dtype, order)
        comm.Bcast(actual, root=0)

        x = dev1.array(actual, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype, order, fromdev=dev1)
        nbytes = max(x.nbytes, y.nbytes)
        fh = self.FILE
        fh.Set_size(0)
        fh.Set_view(0, _get_type(dtype))

        fh.Write_at(rank * nbytes * 10, x)
        fh.Sync()
        comm.Barrier()
        fh.Sync()
        fh.Read_at(rank * nbytes * 10, y)

        assert_equal(_get_array(y), actual)

    @parameterized.expand(_patterns)
    def test_ReadWriteAtAll(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()

        actual = _get_sbuf(np, shape, dtype, order)
        comm.Bcast(actual, root=0)

        x = dev1.array(actual, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype, order, fromdev=dev1)
        fh = self.FILE
        nbytes = max(x.nbytes, y.nbytes)
        fh.Set_size(0)
        fh.Set_view(0, _get_type(dtype))

        fh.Write_at_all(rank * nbytes * 10, x)
        fh.Sync()
        comm.Barrier()
        fh.Sync()
        fh.Read_at_all(rank * nbytes * 10, y)

        assert_equal(_get_array(y), actual)

    @parameterized.expand(_patterns)
    def test_IReadIWriteAt(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()

        actual = _get_sbuf(np, shape, dtype, order)
        comm.Bcast(actual, root=0)

        x = dev1.array(actual, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype, order, fromdev=dev1)
        nbytes = max(x.nbytes, y.nbytes)
        fh = self.FILE
        fh.Set_size(0)
        fh.Set_view(0, _get_type(dtype))

        fh.Iwrite_at(rank * nbytes * 10, x).Wait()
        fh.Sync()
        comm.Barrier()
        fh.Sync()
        fh.Iread_at(rank * nbytes * 10, y).Wait()

        assert_equal(_get_array(y), actual)

    @parameterized.expand(_patterns)
    def test_IReadIWriteAtAll(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
 
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()

        actual = _get_sbuf(np, shape, dtype, order)
        comm.Bcast(actual, root=0)

        x = dev1.array(actual, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype, order, fromdev=dev1)
        nbytes = max(x.nbytes, y.nbytes)
        fh = self.FILE
        fh.Set_size(0)
        fh.Set_view(0, _get_type(dtype))
        fh.Iwrite_at_all(rank * nbytes * 10, x).Wait()
        fh.Sync()
        comm.Barrier()
        fh.Sync()
        fh.Iread_at_all(rank * nbytes * 10, y).Wait()

        assert_equal(_get_array(y), actual)

    @parameterized.expand(_patterns)
    def test_ReadWrite(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()

        actual = _get_sbuf(np, shape, dtype, order)
        comm.Bcast(actual, root=0)

        x = dev1.array(actual, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype, order, fromdev=dev1)
        nbytes = max(x.nbytes, y.nbytes)
        fh = self.FILE
        fh.Set_size(0)
        fh.Set_view(0, _get_type(dtype))
        fh.Seek(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Write(x)
        fh.Sync()
        comm.Barrier()
        fh.Sync()
        fh.Seek(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Read(y)

        assert_equal(_get_array(y), actual)

    @parameterized.expand(_patterns)
    def test_ReadWriteAll(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()

        actual = _get_sbuf(np, shape, dtype, order)
        comm.Bcast(actual, root=0)

        x = dev1.array(actual, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype, order, fromdev=dev1)
        nbytes = max(x.nbytes, y.nbytes)
        fh = self.FILE
        fh.Set_size(0)
        fh.Set_view(0, _get_type(dtype))
        fh.Seek(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Write_all(x)
        fh.Sync()
        comm.Barrier()
        fh.Sync()
        fh.Seek(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Read_all(y)

        assert_equal(_get_array(y), actual)

    @parameterized.expand(_patterns)
    def test_IreadIwrite(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()

        actual = _get_sbuf(np, shape, dtype, order)
        comm.Bcast(actual, root=0)

        x = dev1.array(actual, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype, order, fromdev=dev1)
        nbytes = max(x.nbytes, y.nbytes)
        fh = self.FILE
        fh.Set_size(0)
        fh.Set_view(0, _get_type(dtype))
        fh.Seek(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Iwrite(x).Wait()
        fh.Sync()
        comm.Barrier()
        fh.Sync()
        fh.Seek(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Iread(y).Wait()

        assert_equal(_get_array(y), actual)

    @parameterized.expand(_patterns)
    def test_IreadIwriteAll(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()

        actual = _get_sbuf(np, shape, dtype, order)
        comm.Bcast(actual, root=0)

        x = dev1.array(actual, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype, order, fromdev=dev1)
        nbytes = max(x.nbytes, y.nbytes)
        fh = self.FILE
        fh.Set_size(0)
        fh.Set_view(0, _get_type(dtype))
        fh.Seek(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Iwrite_all(x).Wait()
        fh.Sync()
        comm.Barrier()
        fh.Sync()
        fh.Seek(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Iread_all(y).Wait()

        assert_equal(_get_array(y), actual)

    @parameterized.expand(_patterns)
    @unittest.skipIf(IS_MULTI_HOST, 'necmpi-multi-host')
    def test_ReadWriteShared(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()

        actual = _get_sbuf(np, shape, dtype, order)
        comm.Bcast(actual, root=0)

        x = dev1.array(actual, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype, order, fromdev=dev1)
        nbytes = max(x.nbytes, y.nbytes)
        fh = self.FILE
        fh.Set_size(0)
        fh.Set_view(0, _get_type(dtype))
        fh.Seek_shared(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Write_shared(x)
        fh.Sync()
        comm.Barrier()
        fh.Sync()
        fh.Seek_shared(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Read_shared(y)

        assert_equal(_get_array(y), actual)

    @parameterized.expand(_patterns)
    @unittest.skipIf(IS_MULTI_HOST, 'necmpi-multi-host')
    def test_IreadIwriteShared(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()

        actual = _get_sbuf(np, shape, dtype, order)
        comm.Bcast(actual, root=0)

        x = dev1.array(actual, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype, order, fromdev=dev1)
        nbytes = max(x.nbytes, y.nbytes)
        fh = self.FILE
        fh.Set_size(0)
        fh.Set_view(0, _get_type(dtype))
        fh.Seek_shared(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Iwrite_shared(x).Wait()
        fh.Sync()
        comm.Barrier()
        fh.Sync()
        fh.Seek_shared(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Iread_shared(y).Wait()

        assert_equal(_get_array(y), actual)

    @parameterized.expand(_patterns)
    @unittest.skipIf(IS_MULTI_HOST, 'necmpi-multi-host')
    def test_ReadWriteOrderd(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()

        actual = _get_sbuf(np, shape, dtype, order)
        comm.Bcast(actual, root=0)

        x = dev1.array(actual, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype, order, fromdev=dev1)
        nbytes = max(x.nbytes, y.nbytes)
        fh = self.FILE
        fh.Set_size(0)
        fh.Set_view(0, _get_type(dtype))
        fh.Seek_shared(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Write_ordered(x)
        fh.Sync()
        comm.Barrier()
        fh.Sync()
        fh.Seek_shared(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Read_ordered(y)

        assert_equal(_get_array(y), actual)

    @parameterized.expand(_patterns)
    def test_ReadWriteAtAllBegin(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()

        actual = _get_sbuf(np, shape, dtype, order)
        comm.Bcast(actual, root=0)

        x = dev1.array(actual, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype, order, fromdev=dev1)
        nbytes = max(x.nbytes, y.nbytes)
        fh = self.FILE
        fh.Set_size(0)
        fh.Set_view(0, _get_type(dtype))
        fh.Write_at_all_begin(rank * nbytes * 10, x)
        fh.Write_at_all_end(x)
        fh.Sync()
        comm.Barrier()
        fh.Sync()
        fh.Read_at_all_begin(rank * nbytes * 10, y)
        fh.Read_at_all_end(y)

        assert_equal(_get_array(y), actual)

    @parameterized.expand(_patterns)
    def test_ReadWriteAllBegin(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np) or
                                (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()

        actual = _get_sbuf(np, shape, dtype, order)
        comm.Bcast(actual, root=0)

        x = dev1.array(actual, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype, order, fromdev=dev1)
        nbytes = max(x.nbytes, y.nbytes)
        fh = self.FILE
        fh.Set_size(0)
        fh.Set_view(0, _get_type(dtype))
        fh.Seek(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Write_all_begin(x)
        fh.Write_all_end(x)
        fh.Sync()
        comm.Barrier()
        fh.Sync()
        fh.Seek(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Read_all_begin(y)
        fh.Read_all_end(y)

        assert_equal(_get_array(y), actual)

    @parameterized.expand(_patterns)
    @unittest.skipIf(IS_MULTI_HOST, 'necmpi-multi-host')
    def test_ReadWriteOrderdBegin(self, dev1, dev2, shape, dtype, order):
        if dtype == 'bool' and ((dev1 is vp and dev2 is np)
                                or (dev1 is np and dev2 is vp)):
            self.skipTest('Booleans case in vp-to-np is not testable')
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()

        actual = _get_sbuf(np, shape, dtype, order)
        comm.Bcast(actual, root=0)

        x = dev1.array(actual, dtype=dtype, order=order)
        y = _get_rbuf(dev2, shape, dtype, order, fromdev=dev1)
        nbytes = max(x.nbytes, y.nbytes)
        fh = self.FILE
        fh.Set_size(0)
        fh.Set_view(0, _get_type(dtype))
        fh.Seek_shared(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Write_ordered_begin(x)
        fh.Write_ordered_end(x)
        fh.Sync()
        comm.Barrier()
        fh.Sync()
        fh.Seek_shared(rank * nbytes * 10, MPI.SEEK_SET)
        fh.Read_ordered_begin(y)
        fh.Read_ordered_end(y)

        assert_equal(_get_array(y), actual)


if __name__ == '__main__':
    unittest.main()
