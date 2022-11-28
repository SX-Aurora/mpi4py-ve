from mpi4pyve import MPI
from mpi4pyve import util
import argparse
import math

DTYPE = 'float32'
MPI_DTYPE = MPI.FLOAT
ROOT = 0

class Grid:
    def __init__(self, nproc, order, cart_comm, row_comm, col_comm, row_pos,
                 col_pos, rank):
        self.nproc = nproc
        self.order = order
        self.cart_comm = cart_comm
        self.row_comm = row_comm
        self.col_comm = col_comm
        self.row_pos = row_pos
        self.col_pos = col_pos
        self.rank = rank

    def __str__(self):
        return "Grid Process <row_pos:{}, col_pol:{}, rank:{}>".format(
            self.row_pos, self.col_pos, self.rank)


def create_grid_process(nproc):
    dims = MPI.Compute_dims(nproc, 2)
    if dims[0] != dims[1]:
        raise ValueError('the number of process is not a perfect square')
    cart_comm = comm.Create_cart(dims, periods=[True, True], reorder=True)
    row_comm = cart_comm.Sub([0, 1])
    col_comm = cart_comm.Sub([1, 0])
    coords = cart_comm.coords
    grid = Grid(nproc, dims[0], cart_comm, row_comm, col_comm, coords[0], coords[1],
                cart_comm.Get_rank())
    return grid

def create_block_datatype(sizes, subsizes, grid, itemsize):
    assert subsizes[0] * grid.order == sizes[0]
    assert subsizes[1] * grid.order == sizes[1]
    starts = [0, 0]
    block_type = MPI.Datatype(MPI_DTYPE).Create_subarray(
        sizes, subsizes, starts, order=MPI.ORDER_C)
    resized_type = MPI.Datatype(block_type).Create_resized(
        0, subsizes[1] * itemsize)
    resized_type.Commit()
    return resized_type

def scatter_matrix(root_mat, local_mat, grid, n_d, block_type):
    sendcount = [1 for i in range(grid.nproc)]
    recvcount = local_mat.size
    displs = []
    offset = 0
    for i in range(grid.order):
        offset = i * grid.order * n_d
        for j in range(grid.order):
            displs.append(offset)
            offset += 1
    sendbuf = [root_mat, sendcount, displs, block_type]
    recvbuf = [local_mat, recvcount, MPI_DTYPE]
    grid.cart_comm.Scatterv(sendbuf, recvbuf, root=ROOT)

def gather_matrix(root_mat, local_mat, grid, n_d, block_type):
    recvcount = [1 for i in range(grid.nproc)]
    sendcount = local_mat.size
    displs = []
    offset = 0
    for i in range(grid.order):
        offset = i * grid.order * n_d
        for j in range(grid.order):
            displs.append(offset)
            offset += 1
    sendbuf = [local_mat, sendcount, MPI_DTYPE]
    recvbuf = [root_mat, recvcount, displs, block_type]
    grid.cart_comm.Gatherv(sendbuf, recvbuf, root=ROOT)

def matmul(local_A, local_B, local_C, grid):
    for i in range(grid.order - 1):
        peer_send = (grid.col_pos + i + 1) % grid.order
        grid.row_comm.Isend(local_A[grid.col_pos], peer_send)
    A_recvreqs = [None for i in range(grid.order)]
    for i in range(grid.order - 1):
        peer_recv = (grid.col_pos - i - 1 + grid.order) % grid.order
        req = grid.row_comm.Irecv(local_A[peer_recv], peer_recv)
        A_recvreqs[peer_recv] = req
    for i in range(grid.order - 1):
        peer_send = (grid.row_pos + i + 1) % grid.order
        grid.col_comm.Isend(local_B[grid.row_pos], peer_send)
    B_recvreqs = [None for i in range(grid.order)]
    for i in range(grid.order - 1):
        peer_recv = (grid.row_pos - i - 1 + grid.order) % grid.order
        req = grid.col_comm.Irecv(local_B[peer_recv], peer_recv)
        B_recvreqs[peer_recv] = req
    for i in range(grid.order):
        idx = i
        if A_recvreqs[idx]: A_recvreqs[idx].wait()
        if B_recvreqs[idx]: B_recvreqs[idx].wait()
        local_C += local_A[idx] @ local_B[idx]

def scaling(flops):
    units = [
        [1e12, 'TFLOPS'], [1e9, 'GFLOPS'], [1e6, 'MFLOPS'],
        [1e3, 'KFLOPS'], [1, 'FLOPS']]
    for scale, unit in units:
        if flops >= scale:
            break
    return unit, flops / scale

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dev', type=str, required=True, choices=['vh', 've'],
                        help='Execution device')
    parser.add_argument('-dtype', type=str, required=True, choices=['float', 'double'],
                        help='Execution data type')
    parser.add_argument('-n', type=int, required=False, default=10,
                        help='the number of row and col')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0: print(vars(args))

    # parse arguments
    if args.dev == 'vh':
        import numpy as np
        dev = np
    elif args.dev == 've':
        import nlcpy as vp
        dev = vp
    if args.dtype == 'float':
        DTYPE = 'f4'
        MPI_DTYPE = MPI.FLOAT
    elif args.dtype == 'double':
        DTYPE = 'f8'
        MPI_DTYPE = MPI.DOUBLE
    else:
        raise ValueError
    n = args.n

    # create grid process
    grid = create_grid_process(nproc)

    # create matrix A, B, C
    rng = dev.random.default_rng()
    if grid.rank == ROOT:
        A = rng.random((n, n), dtype=DTYPE)
        B = rng.random((n, n), dtype=DTYPE)
        C = dev.zeros((n, n), dtype=DTYPE)
    else:
        A = None
        B = None
        C = None
    if n % grid.order != 0:
        raise ValueError('n is not evenly divisible by sqrt(nproc)')

    # create local matrix for computation
    n_d = n // grid.order
    local_A = [dev.zeros((n_d, n_d), dtype=DTYPE) for _ in range(grid.order)]
    local_B = [dev.zeros((n_d, n_d), dtype=DTYPE) for _ in range(grid.order)]
    local_C = dev.zeros((n_d, n_d), dtype=DTYPE)
    block_type = create_block_datatype(
        (n, n), (n_d, n_d), grid, dev.dtype(DTYPE).itemsize)
    scatter_matrix(A, local_A[grid.row_pos], grid, n_d, block_type)
    scatter_matrix(B, local_B[grid.col_pos], grid, n_d, block_type)

    # execute matmul
    if dev.__name__ == 'nlcpy':
        dev.request.flush()
    grid.cart_comm.Barrier()
    t0 = MPI.Wtime()
    matmul(local_A, local_B, local_C, grid)
    if dev.__name__ == 'nlcpy':
        dev.request.flush()
    grid.cart_comm.Barrier()
    t1 = MPI.Wtime()

    # result check and show perf
    gather_matrix(C, local_C, grid, n_d, block_type)
    if grid.rank == 0:
        elapsed = t1 - t0
        flops = 2 * n ** 3 / elapsed
        print("elapsed: {} [sec], {}: {}".format(elapsed, *scaling(flops)))
        exp = A @ B
        norm = dev.linalg.norm(C)
        if dev.all(((C - exp) / norm) < 1e-4):
            print("result OK")
        else:
            print("result NG")
