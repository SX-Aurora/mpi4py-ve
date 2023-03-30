from mpi4pyve import MPI
from mpi4pyve import util
import numpy as np
import nlcpy as vp
import argparse

def send_recv_helper(buf, comm):
    max_count = 2 ** 31
    begin = 0
    remain = buf.size
    while remain > 0:
        part = buf[begin:min(begin + remain, begin + max_count - 1)]
        if rank == 0:
            comm.Send([part, MPI.DOUBLE], 1)
            comm.Recv([part, MPI.DOUBLE], 1)
        elif rank == 1:
            comm.Recv([part, MPI.DOUBLE], 0)
            comm.Send([part, MPI.DOUBLE], 0)
        begin += part.size
        remain -= part.size

parser = argparse.ArgumentParser()
parser.add_argument('-dev1', type=str, required=True, choices=['vh', 've'])
parser.add_argument('-dev2', type=str, required=True, choices=['vh', 've'])
parser.add_argument('-n', type=int, required=False, default=20)
parser.add_argument('-loop_count', type=int, required=False, default=10)
args = parser.parse_args()

# set module
if args.dev1 == 'vh':
    dev1 = np
elif args.dev1 == 've':
    dev1 = vp
else:
    raise ValueError
if args.dev2 == 'vh':
    dev2 = np
elif args.dev2 == 've':
    dev2 = vp
else:
    raise ValueError

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
if rank == 0: print(vars(args))
if size != 2:
    raise ValueError

loop_count = args.loop_count
if rank == 0:
    print('| Data Size (B) | Avg Transfer Time (s) | Bandwidth (GB/s) |')
    print('|---------------|-----------------------|------------------|')

for n in range(0, args.n):
    nelem = 1 << n
    if rank == 0:
        buf = dev1.arange(nelem, dtype='f8')
    else:
        buf = dev2.empty(nelem, dtype='f8')
    if isinstance(buf, vp.ndarray):
        buf.venode.synchronize()
    comm.Barrier()
    t0 = MPI.Wtime()
    for i in range(loop_count):
        send_recv_helper(buf, comm)
    comm.Barrier()
    t1 = MPI.Wtime()
    elapsed_time = t1 - t0
    avg_transfer_time = elapsed_time / (2.0 * loop_count)
    bandwidth = buf.nbytes / (1024 ** 3) / avg_transfer_time
    if rank == 0:
        print('|{:>15d}|{:23.9f}|{:18.9f}|'.format(buf.nbytes, avg_transfer_time, bandwidth))
    if not np.array_equal(np.asarray(buf), np.arange(nelem, dtype='f8')):
        print("Result mismatch (rank = {})".format(rank))
        MPI.Finalize()
        exit()

if rank == 0:
    print('|---------------|-----------------------|------------------|')
MPI.Finalize()
