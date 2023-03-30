from mpi4pyve import MPI
import sys, os, tempfile

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def get_fh():
    fh = MPI.FILE_NULL
    fname = None
    if comm.Get_rank() == 0:
        fd, fname = tempfile.mkstemp(prefix='mpi4pyve-')
        os.close(fd)
    fname = comm.bcast(fname, 0)
    amode  = MPI.MODE_RDWR | MPI.MODE_CREATE
    amode |= MPI.MODE_DELETE_ON_CLOSE
    info = MPI.INFO_NULL
    try:
        fh = MPI.File.Open(comm, fname, amode, info)
        return fh
    except Exception:
        if comm.Get_rank() == 0:
            os.remove(fname)
        raise
