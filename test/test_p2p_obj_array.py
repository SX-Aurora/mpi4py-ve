from mpi4pyve import MPI
import nlcpy as vp
import numpy as np
import mpiunittest as unittest
import sys


pypy_lt_53 = (hasattr(sys, 'pypy_version_info') and
              sys.pypy_version_info < (5, 3))

def allocate(n):
    if pypy_lt_53:
        try:
            import array
            return array.array('B', [0]) * n
        except ImportError:
            return None
    return bytearray(n)

vp_array=vp.array([1,2,3])
messages = [vp_array,]

class BaseTestP2PObj(object):

    COMM = MPI.COMM_NULL

    def testSendAndRecv(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for smess in messages:
            self.COMM.send(smess,  MPI.PROC_NULL)
            rmess = self.COMM.recv(None, MPI.PROC_NULL, 0)
            self.assertEqual(rmess, None)
        if size == 1: return
        for smess in messages:
            if rank == 0:
                self.COMM.send(smess,  rank+1, 0)
                rmess = smess
            elif rank == size - 1:
                rmess = self.COMM.recv(None, rank-1, 0)
            else:
                rmess = self.COMM.recv(None, rank-1, 0)
                self.COMM.send(rmess,  rank+1, 0)
            self.assertTrue(np.array_equal(smess, rmess))

    def testISendAndRecv(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        buf = None
        for smess in messages:
            req = self.COMM.isend(smess,  MPI.PROC_NULL)
            self.assertTrue(req)
            req.Wait()
            self.assertFalse(req)
            rmess = self.COMM.recv(buf, MPI.PROC_NULL, 0)
            self.assertEqual(rmess, None)
        for smess in messages:
            req = self.COMM.isend(smess,  rank, 0)
            self.assertTrue(req)
            rmess = self.COMM.recv(buf, rank, 0)
            self.assertTrue(req)
            #flag = req.Test()
            flag = False
            while not flag:
                flag = req.Test()
            self.assertTrue(flag)
            self.assertFalse(req)
            self.assertTrue(np.array_equal(smess, rmess))
        for smess in messages:
            dst = (rank+1)%size
            src = (rank-1)%size
            req = self.COMM.isend(smess,  dst, 0)
            self.assertTrue(req)
            rmess = self.COMM.recv(buf,  src, 0)
            req.Wait()
            self.assertFalse(req)
            self.assertTrue(np.array_equal(smess, rmess))

    def testIRecvAndSend(self):
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        for smess in messages:
            req = comm.irecv(0, MPI.PROC_NULL)
            self.assertTrue(req)
            comm.send(smess,  MPI.PROC_NULL)
            rmess = req.wait()
            self.assertFalse(req)
            self.assertEqual(rmess, None)
        for smess in messages:
            buf = allocate(512)
            req = comm.irecv(buf, rank, 0)
            self.assertTrue(req)
            flag, rmess = req.test()
            self.assertTrue(req)
            self.assertFalse(flag)
            self.assertEqual(rmess, None)
            comm.send(smess, rank, 0)
            self.assertTrue(req)
            flag, rmess = req.test()
            while not flag: flag, rmess = req.test()
            self.assertTrue(flag)
            self.assertFalse(req)
            self.assertTrue(np.array_equal(smess, rmess))
        tmp = allocate(1024)
        for buf in (None, 1024, tmp):
            for smess in messages:
                dst = (rank+1)%size
                src = (rank-1)%size
                req = comm.irecv(buf, src, 0)
                self.assertTrue(req)
                comm.send(smess, dst, 0)
                rmess = req.wait()
                self.assertFalse(req)
                self.assertTrue(np.array_equal(smess, rmess))
        for smess in messages:
            src = dst = rank
            rreq1 = comm.irecv(None, src, 1)
            rreq2 = comm.irecv(None, src, 2)
            rreq3 = comm.irecv(None, src, 3)
            rreqs = [rreq1, rreq2, rreq3]
            for i in range(len(rreqs)):
                self.assertTrue(rreqs[i])
                comm.send(smess, dst, i+1)
                index, obj = MPI.Request.waitany(rreqs)
                self.assertEqual(index, i)
                self.assertTrue(np.array_equal(smess, obj))
                self.assertFalse(rreqs[index])
            index, obj = MPI.Request.waitany(rreqs)
            self.assertEqual(index, MPI.UNDEFINED)
            self.assertEqual(obj, None)
        for smess in messages:
            src = dst = rank
            rreq1 = comm.irecv(None, src, 1)
            rreq2 = comm.irecv(None, src, 2)
            rreq3 = comm.irecv(None, src, 3)
            rreqs = [rreq1, rreq2, rreq3]
            index, flag, obj = MPI.Request.testany(rreqs)
            self.assertEqual(index, MPI.UNDEFINED)
            self.assertEqual(flag, False)
            self.assertEqual(obj, None)
            for i in range(len(rreqs)):
                self.assertTrue(rreqs[i])
                comm.send(smess, dst, i+1)
                index, flag, obj = MPI.Request.testany(rreqs)
                while not flag:
                    index, flag, obj = MPI.Request.testany(rreqs)
                self.assertEqual(index, i)
                self.assertEqual(flag, True)
                self.assertTrue(np.array_equal(obj, smess))
                self.assertFalse(rreqs[i])
            index, flag, obj = MPI.Request.testany(rreqs)
            self.assertEqual(index, MPI.UNDEFINED)
            self.assertEqual(flag, True)
            self.assertEqual(obj, None)

    def testIRecvAndISend(self):
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        tmp = allocate(512)
        for smess in messages:
            dst = (rank+1)%size
            src = (rank-1)%size
            rreq = comm.irecv(None, src, 0)
            self.assertTrue(rreq)
            sreq = comm.isend(smess, dst, 0)
            self.assertTrue(sreq)
            index1, mess1 = MPI.Request.waitany([sreq,rreq])
            self.assertTrue(index1 in (0, 1))
            if index1 == 0:
                self.assertFalse(sreq)
                self.assertTrue (rreq)
                self.assertEqual(mess1, None)
            else:
                self.assertTrue (sreq)
                self.assertFalse(rreq)
                self.assertTrue(np.array_equal(mess1, rmess))
            index2, mess2 = MPI.Request.waitany([sreq,rreq])
            self.assertTrue(index2 in (0, 1))
            self.assertNotEqual(index2, index1)
            self.assertFalse(sreq)
            self.assertFalse(rreq)
            if index2 == 0:
                self.assertEqual(mess2, None)
            else:
                self.assertTrue(np.array_equal(mess2, smess))
        for smess in messages:
            dst = (rank+1)%size
            src = (rank-1)%size
            rreq = comm.irecv(None, src, 0)
            self.assertTrue(rreq)
            sreq = comm.isend(smess, dst, 0)
            self.assertTrue(sreq)
            index1, flag1, mess1 = MPI.Request.testany([sreq,rreq])
            while not flag1:
                index1, flag1, mess1 = MPI.Request.testany([sreq,rreq])
            self.assertTrue(index1 in (0, 1))
            if index1 == 0:
                self.assertFalse(sreq)
                self.assertTrue (rreq)
                self.assertEqual(mess1, None)
            else:
                self.assertTrue (sreq)
                self.assertFalse(rreq)
                self.assertTrue(np.array_equal(mess1, smess))
            index2, flag2, mess2 = MPI.Request.testany([sreq,rreq])
            while not flag2:
                index2, flag2, mess2 = MPI.Request.testany([sreq,rreq])
            self.assertTrue(index2 in (0, 1))
            self.assertNotEqual(index2, index1)
            self.assertFalse(sreq)
            self.assertFalse(rreq)
            if index2 == 0:
                self.assertEqual(mess2, None)
            else:
                self.assertTrue(np.array_equal(mess2, smess))
        for buf in (None, 512, tmp):
            for smess in messages:
                dst = (rank+1)%size
                src = (rank-1)%size
                rreq = comm.irecv(buf, src, 0)
                self.assertTrue(rreq)
                sreq = comm.isend(smess, dst, 0)
                self.assertTrue(sreq)
                dummy, rmess = MPI.Request.waitall([sreq,rreq], [])
                self.assertFalse(sreq)
                self.assertFalse(rreq)
                self.assertEqual(dummy, None)
                self.assertTrue(np.array_equal(rmess, smess))
        for buf in (None, 512, tmp):
            for smess in messages:
                src = dst = rank
                rreq = comm.irecv(buf, src, 1)
                flag, msg = MPI.Request.testall([rreq])
                self.assertEqual(flag, False)
                self.assertEqual(msg, None)
                sreq = comm.isend(smess, dst, 1)
                while True:
                    flag, msg = MPI.Request.testall([sreq,rreq], [])
                    if not flag:
                        self.assertEqual(msg, None)
                        continue
                    (dummy, rmess) = msg
                    self.assertEqual(dummy, None)
                    self.assertTrue(np.array_equal(rmess, smess))
                    break

    def testManyISendAndRecv(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for smess in messages:
            reqs = []
            for k in range(6):
                r = self.COMM.isend(smess,  rank, 0)
                reqs.append(r)
            flag = MPI.Request.Testall(reqs)
            if not flag:
                index, flag = MPI.Request.Testany(reqs)
                indices = MPI.Request.Testsome(reqs)
                if indices is None:
                    count = MPI.UNDEFINED
                    indices = []
                else:
                    count = len(indices)
                self.assertTrue(count in  [0, MPI.UNDEFINED])
            for k in range(3):
                rmess = self.COMM.recv(None, rank, 0)
                self.assertTrue(np.array_equal(rmess, smess))
            flag = MPI.Request.Testall(reqs)
            if not flag:
                index, flag = MPI.Request.Testany(reqs)
                self.assertEqual(index,  0)
                self.assertTrue(flag)
                indices = MPI.Request.Testsome(reqs)
                if indices is None:
                    count = MPI.UNDEFINED
                    indices = []
                else:
                    count = len(indices)
                self.assertTrue(count >= 2)
                indices = list(indices)
                indices.sort()
                self.assertTrue(indices[:2] == [1, 2])
            for k in range(3):
                rmess = self.COMM.recv(None, rank, 0)
                self.assertTrue(np.array_equal(rmess, smess))
            flag = MPI.Request.Testall(reqs)
            self.assertTrue(flag)
        for smess in messages:
            reqs = []
            for k in range(6):
                r = self.COMM.isend(smess,  rank, 0)
                reqs.append(r)
            for k in range(3):
                rmess = self.COMM.recv(None, rank, 0)
                self.assertTrue(np.array_equal(rmess, smess)) 
            index = MPI.Request.Waitany(reqs)
            self.assertTrue(index == 0)
            self.assertTrue(flag)
            indices1 = MPI.Request.Waitsome(reqs)
            if indices1 is None:
                count1 = MPI.UNDEFINED
                indices1 = []
            else:
                count1 = len(indices1)
            for k in range(3):
                rmess = self.COMM.recv(None, rank, 0)
                self.assertTrue(np.array_equal(rmess, smess))
            indices2 = MPI.Request.Waitsome(reqs)
            if indices2 is None:
                count2 = MPI.UNDEFINED
                indices2 = []
            else:
                count2 = len(indices2)
            if count1 == MPI.UNDEFINED: count1 = 0
            if count2 == MPI.UNDEFINED: count2 = 0
            self.assertEqual(6, 1+count1+count2)
            indices = [0]+list(indices1)+list(indices2)
            indices.sort()
            self.assertEqual(indices, list(range(6)))
            MPI.Request.Waitall(reqs)

    def testSSendAndRecv(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for smess in messages:
            self.COMM.ssend(smess,  MPI.PROC_NULL)
            rmess = self.COMM.recv(None, MPI.PROC_NULL, 0)
            self.assertEqual(rmess, None)
        if size == 1: return
        for smess in messages:
            if rank == 0:
                self.COMM.ssend(smess,  rank+1, 0)
                rmess = smess
            elif rank == size - 1:
                rmess = self.COMM.recv(None, rank-1, 0)
            else:
                rmess = self.COMM.recv(None, rank-1, 0)
                self.COMM.ssend(rmess,  rank+1, 0)
            self.assertTrue(np.array_equal(rmess, smess))

    def testISSendAndRecv(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for smess in messages:
            req = self.COMM.issend(smess,  MPI.PROC_NULL)
            self.assertTrue(req)
            req.Wait()
            self.assertFalse(req)
            rmess = self.COMM.recv(None, MPI.PROC_NULL, 0)
            self.assertEqual(rmess, None)
        for smess in messages:
            req = self.COMM.issend(smess,  rank, 0)
            self.assertTrue(req)
            flag = req.Test()
            self.assertFalse(flag)
            self.assertTrue(req)
            rmess = self.COMM.recv(None, rank, 0)
            self.assertTrue(req)
            #flag = req.Test()
            flag = False
            while not flag:
                flag = req.Test()
            self.assertTrue(flag)
            self.assertFalse(req)
            self.assertTrue(np.array_equal(rmess, smess))
        for smess in messages:
            dst = (rank+1)%size
            src = (rank-1)%size
            req = self.COMM.issend(smess,  dst, 0)
            self.assertTrue(req)
            rmess = self.COMM.recv(None,  src, 0)
            req.Wait()
            self.assertFalse(req)
            self.assertTrue(np.array_equal(rmess, smess))

    def testIRecvAndBSend(self):
        comm = self.COMM
        rank = comm.Get_rank()
        buf = MPI.Alloc_mem((1<<16)+MPI.BSEND_OVERHEAD)
        MPI.Attach_buffer(buf)
        try:
            for smess in messages:
                src = dst = rank
                req1 = comm.irecv(None, src, 1)
                req2 = comm.irecv(None, src, 2)
                req3 = comm.irecv(None, src, 3)
                comm.bsend(smess, dst, 3)
                comm.bsend(smess, dst, 2)
                comm.bsend(smess, dst, 1)
                self.assertTrue(np.array_equal(smess, req3.wait()))
                self.assertTrue(np.array_equal(smess, req2.wait()))
                self.assertTrue(np.array_equal(smess, req1.wait()))
                comm.bsend(smess, MPI.PROC_NULL, 3)
        finally:
            MPI.Detach_buffer()
            MPI.Free_mem(buf)

    def testIRecvAndIBSend(self):
        comm = self.COMM
        rank = comm.Get_rank()
        buf = MPI.Alloc_mem((1<<16)+MPI.BSEND_OVERHEAD)
        MPI.Attach_buffer(buf)
        try:
            for smess in messages:
                src = dst = rank
                req1 = comm.irecv(None, src, 1)
                req2 = comm.irecv(None, src, 2)
                req3 = comm.irecv(None, src, 3)
                req4 = comm.ibsend(smess, dst, 3)
                req5 = comm.ibsend(smess, dst, 2)
                req6 = comm.ibsend(smess, dst, 1)
                MPI.Request.waitall([req4, req5, req6])
                self.assertTrue(np.array_equal(smess, req3.wait()))
                self.assertTrue(np.array_equal(smess, req2.wait()))
                self.assertTrue(np.array_equal(smess, req1.wait()))
                comm.ibsend(smess, MPI.PROC_NULL, 3).wait()
        finally:
            MPI.Detach_buffer()
            MPI.Free_mem(buf)

    def testIRecvAndSSend(self):
        comm = self.COMM
        rank = comm.Get_rank()
        for smess in messages:
            src = dst = rank
            req1 = comm.irecv(None, src, 1)
            req2 = comm.irecv(None, src, 2)
            req3 = comm.irecv(None, src, 3)
            comm.ssend(smess, dst, 3)
            comm.ssend(smess, dst, 2)
            comm.ssend(smess, dst, 1)
            self.assertTrue(np.array_equal(smess, req3.wait()))
            self.assertTrue(np.array_equal(smess, req2.wait()))
            self.assertTrue(np.array_equal(smess, req1.wait()))
            comm.ssend(smess, MPI.PROC_NULL, 3)

    def testIRecvAndISSend(self):
        comm = self.COMM
        rank = comm.Get_rank()
        for smess in messages:
            src = dst = rank
            req1 = comm.irecv(None, src, 1)
            req2 = comm.irecv(None, src, 2)
            req3 = comm.irecv(None, src, 3)
            req4 = comm.issend(smess, dst, 3)
            req5 = comm.issend(smess, dst, 2)
            req6 = comm.issend(smess, dst, 1)
            MPI.Request.waitall([req4, req5, req6])
            self.assertTrue(np.array_equal(smess, req3.wait()))
            self.assertTrue(np.array_equal(smess, req2.wait()))
            self.assertTrue(np.array_equal(smess, req1.wait()))
            comm.issend(smess, MPI.PROC_NULL, 3).wait()

    def testSendrecv(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for smess in messages:
            dest = (rank + 1) % size
            source = (rank - 1) % size
            rmess = self.COMM.sendrecv(smess, dest,   0,
                                       None,  source, 0)
            continue
            self.assertTrue(np.array_equal(rmess, smess))
            rmess = self.COMM.sendrecv(None,  dest,   0,
                                       None,  source, 0)
            self.assertEqual(rmess, None)
            rmess = self.COMM.sendrecv(smess,  MPI.PROC_NULL, 0,
                                       None,   MPI.PROC_NULL, 0)
            self.assertEqual(rmess, None)

    def testSendrecv_recvbuf_nosetting(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        for smess in messages:
            if rank == 0:
                comm.send(smess , dest=1)
                rmess = comm.recv()
                comm.Barrier()
            elif rank == 1:
                rmess = comm.sendrecv(smess, dest=0)
                comm.Barrier()
            else:
                rmess = smess
                comm.Barrier()
            self.assertTrue(np.array_equal(rmess, smess))

    def testSendAndRecv_recvbuf_nosetting(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        for smess in messages:
            if rank == 0:
                comm.send(smess, dest=1)
                rmess = smess
                comm.Barrier()
            elif rank == 1:
                rmess = comm.recv()
                comm.Barrier()
            else:
                rmess = smess
                comm.Barrier()
            self.assertTrue(np.array_equal(smess, rmess))

    def testSendAndIRecv_recvbuf_nosetting(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        for smess in messages:
            if rank == 0:
                comm.send(smess , dest=1)
                rmess = smess
                comm.Barrier()
            elif rank == 1:
                rreq = comm.irecv()
                rmess = rreq.wait()
                comm.Barrier()
            else:
                rmess = smess
                comm.Barrier()
            self.assertTrue(np.array_equal(smess, rmess ))

    def testSendAndIRecv_test(self):
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        for smess in messages:
            comm.send(smess , dest=rank)
            rreq = comm.irecv(source=rank)
            rmess = rreq.test()[1]
            self.assertTrue(np.array_equal(smess, rmess ))

    def testSendAndIRecv_waitall(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        for smess in messages:
            if rank == 0:
                comm.send(smess , dest=1)
                rmess = smess
                comm.Barrier()
            elif rank == 1:
                rreq = comm.irecv()
                rmess = MPI.Request.waitall([rreq])[0]
                comm.Barrier()
            else:
                rmess = smess
                comm.Barrier()
            self.assertTrue(np.array_equal(smess, rmess ))

    def testSendAndIRecv_waitall_many(self):
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        for smess in messages:
            rng = range(1, 4)
            sbufs = sbufs = [smess*n for n in rng]
            for sbuf in sbufs:
                comm.send(sbuf, dest=rank)
            rreqs = [comm.irecv(source=rank) for n in rng]
            ret = MPI.Request.waitall(rreqs)
            comm.Barrier()
            for n, rbuf in enumerate(ret):
                self.assertTrue(np.array_equal(rbuf, sbufs[n]))

    def testSendAndIRecv_testall(self):
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        for smess in messages:
            comm.send(smess , dest=rank)
            rreq = comm.irecv(source=rank)
            rmess = MPI.Request.testall([rreq])[1][0]
            self.assertTrue(np.array_equal(smess, rmess ))

            rng = range(1, 4)
            sbufs = [smess*n for n in rng]
            for sbuf in sbufs:
                comm.send(sbuf, dest=rank)
            rreqs = [comm.irecv(source=rank) for n in rng]
            ret = MPI.Request.testall(rreqs)
            comm.Barrier()
            if ret[0]:
                for n, rbuf in enumerate(ret[1]):
                    self.assertTrue(np.array_equal(rbuf, sbufs[n]))

    def testSendAndIRecv_waitany(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        for smess in messages:
            if rank == 0:
                comm.send(smess , dest=1)
                rmess = smess
                comm.Barrier()
            elif rank == 1:
                rreq = comm.irecv()
                rmess = MPI.Request.waitany([rreq])[1]
                comm.Barrier()
            else:
                rmess = smess
                comm.Barrier()
            self.assertTrue(np.array_equal(smess, rmess ))

    def testSendAndIRecv_waitany_many(self):
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        for smess in messages:
            rng = range(1, 4)
            sbufs = sbufs = [smess*n for n in rng]
            for sbuf in sbufs:
                comm.send(sbuf, dest=rank)
            rreqs = [comm.irecv(source=rank) for n in rng]
            ret = MPI.Request.waitany(rreqs)
            comm.Barrier()
            self.assertTrue(np.array_equal(ret[1], sbufs[ret[0]]))

    def testSendAndIRecv_testany(self):
        comm = self.COMM
        size = comm.Get_size()
        rank = comm.Get_rank()
        for smess in messages:
            comm.send(smess , dest=rank)
            rreq = comm.irecv(source=rank)
            rmess = MPI.Request.testany([rreq])[2]
            self.assertTrue(np.array_equal(smess, rmess ))

            rng = range(1, 4)
            sbufs = [smess*n for n in rng]
            for sbuf in sbufs:
                comm.send(sbuf, dest=rank)
            rreqs = [comm.irecv(source=rank) for n in rng]
            ret = MPI.Request.testany(rreqs)
            comm.Barrier()
            if ret[1]:
                self.assertTrue(np.array_equal(ret[2], sbufs[ret[0]]))

    @unittest.skip('necmpi')
    def testMixed(self):
        comm = self.COMM
        rank = comm.Get_rank()
        #
        sreq = comm.Isend([None, 0, 'B'], rank)
        obj = comm.recv(None, rank)
        sreq.Wait()
        self.assertTrue(obj is None)
        for smess in messages:
            buf = MPI.pickle.dumps(smess)
            sreq = comm.Isend([buf, 'B'], rank)
            rmess = comm.recv(None, rank)
            sreq.Wait()
            self.assertTrue(np.array_equal(rmess, smess))
        #
        sreq = comm.Isend([None, 0, 'B'], rank)
        rreq = comm.irecv(None, rank)
        sreq.Wait()
        obj = rreq.wait()
        self.assertTrue(obj is None)
        for smess in messages:
            buf = MPI.pickle.dumps(smess)
            sreq = comm.Isend([buf, 'B'], rank)
            rreq = comm.irecv(None, rank)
            sreq.Wait()
            rmess = rreq.wait()
            self.assertTrue(np.array_equal(rmess, smess))
    def testPingPong01(self):
        size = self.COMM.Get_size()
        rank = self.COMM.Get_rank()
        for smess in messages:
            self.COMM.send(smess, MPI.PROC_NULL)
            rmess = self.COMM.recv(None, MPI.PROC_NULL, 0)
            self.assertEqual(rmess, None)
        if size == 1: return
        smess = None
        if rank == 0:
            self.COMM.send(smess,  rank+1, 0)
            rmess = self.COMM.recv(None, rank+1, 0)
        elif rank == 1:
            rmess = self.COMM.recv(None, rank-1, 0)
            self.COMM.send(smess,  rank-1, 0)
        else:
            rmess = smess
        self.assertEqual(rmess, smess)
        for smess in messages:
            if rank == 0:
                self.COMM.send(smess,  rank+1, 0)
                rmess = self.COMM.recv(None, rank+1, 0)
            elif rank == 1:
                rmess = self.COMM.recv(None, rank-1, 0)
                self.COMM.send(smess,  rank-1, 0)
            else:
                rmess = smess
            self.assertTrue(np.array_equal(rmess, smess))

    @unittest.skipMPI('MPICH1')
    def testProbe(self):
        comm = self.COMM.Dup()
        try:
            status = MPI.Status()
            flag = comm.iprobe(MPI.ANY_SOURCE, MPI.ANY_TAG, status)
            self.assertFalse(flag)
            for smess in messages:
                request = comm.issend(smess, comm.rank, 123)
                self.assertTrue(request)
                #flag = comm.iprobe(MPI.ANY_SOURCE, MPI.ANY_TAG, status)
                while not comm.iprobe(MPI.ANY_SOURCE, MPI.ANY_TAG, status): pass
                self.assertEqual(status.source, comm.rank)
                self.assertEqual(status.tag, 123)
                #self.assertTrue(flag)
                comm.probe(MPI.ANY_SOURCE, MPI.ANY_TAG, status)
                self.assertEqual(status.source, comm.rank)
                self.assertEqual(status.tag, 123)
                self.assertTrue(request)
                flag, obj = request.test()
                self.assertTrue(request)
                self.assertFalse(flag)
                self.assertEqual(obj, None)
                obj = comm.recv(None, comm.rank, 123)
                self.assertTrue(np.array_equal(obj, smess))
                self.assertTrue(request)
                #flag, obj = request.test()
                obj = request.wait()
                self.assertFalse(request)
                #self.assertTrue(flag)
                self.assertEqual(obj, None)
        finally:
            comm.Free()


class TestP2PObjSelf(BaseTestP2PObj, unittest.TestCase):
    COMM = MPI.COMM_SELF

class TestP2PObjWorld(BaseTestP2PObj, unittest.TestCase):
    COMM = MPI.COMM_WORLD

class TestP2PObjSelfDup(TestP2PObjSelf):
    def setUp(self):
        self.COMM = MPI.COMM_SELF.Dup()
    def tearDown(self):
        self.COMM.Free()

@unittest.skipMPI('openmpi(<1.4.0)', MPI.Query_thread() > MPI.THREAD_SINGLE)
class TestP2PObjWorldDup(TestP2PObjWorld):
    def setUp(self):
        self.COMM = MPI.COMM_WORLD.Dup()
    def tearDown(self):
        self.COMM.Free()



if __name__ == '__main__':
    unittest.main()
