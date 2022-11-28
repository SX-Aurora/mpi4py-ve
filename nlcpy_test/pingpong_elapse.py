from mpi4pyve import MPI
import nlcpy as vp
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

tag1 = 10
tag2 = 20
loop_count = 50

comm.barrier()
status = MPI.Status()

for N in range(0,28):
    A = 1 << N
    V = vp.zeros(A, dtype=float)

    # Warm-up
    for i in range(5):
        if rank == 0:
            comm.Send([V, MPI.DOUBLE], 1, tag1)
            comm.Recv([V, MPI.DOUBLE], 1, tag2, status=status)
        elif rank == 1:
            comm.Recv([V, MPI.DOUBLE], 0, tag1, status=status)
            comm.Send([V, MPI.DOUBLE], 0, tag2)

    comm.barrier()

    t0 = MPI.Wtime()
    for i in range(loop_count):
        if rank == 0:
            comm.Send([V, MPI.DOUBLE], 1, tag1)
            comm.Recv([V, MPI.DOUBLE], 1, tag2, status=status)
        elif rank == 1:
            comm.Recv([V, MPI.DOUBLE], 0, tag1, status=status)
            comm.Send([V, MPI.DOUBLE], 0, tag2)
    t1 = MPI.Wtime()
    elapsed_time = t1 - t0
    num_B = 8*A
    B_in_GB = 1 << 30
    num_GB = num_B / B_in_GB
    avg_time_per_transfer = elapsed_time / (2.0 * loop_count)
    if rank == 0:
        print('Transfer size (B): {:>10d}, Transfer Time (s): {:15.9f}, Bandwidth (GB/s): {:15.9f}'.format(num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer))
