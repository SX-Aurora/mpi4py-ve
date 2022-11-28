from mpi4pyve import MPI
from mpi4pyve import util
from matplotlib import pyplot as plt
import argparse

NX = 100               # The number of grid points in X-direction.
NY = 100               # The number of grid points in Y-direction.
NZ = 100               # The number of grid points in Z-direction.
DT = 0.001             # The time step interval.
CHECK_INTERVAL = 1000  # The number of time steps for checking convergence.
LX = NX * 1e-3
LY = NY * 1e-3
LZ = NZ * 1e-3
T0 = 20.0
T1 = 60.0
T2 = 80.0
HC = 398.0 / (8960.0 * 385.0)
DTYPE = 'float32'
TOLERANCE = 1e-2

def initialize(grid, dev):
    grid.fill(T0)
    grid[:, :, 0] = T1 * dev.sin(
        dev.linspace(0, dev.pi, grid.shape[0]))[:, None]
    grid[:, 0, :] = T2 * dev.sin(
        dev.linspace(0, dev.pi, grid.shape[0]))[:, None]

def get_count_and_displs(rank, size):
    offset = 0
    count = []
    displs = []
    for r in range(size):
        lz_s = NZ * r // size
        lz_e = NZ * (r + 1) // size
        count.append(lz_e - lz_s + 2)
        displs.append(lz_s)
    return count, displs

def create_stencil_kernel(grid_work1, grid_work2, coef, vp):
    kernels = []
    dgrid1, dgrid2 = vp.sca.create_descriptor((grid_work1, grid_work2))
    # input: grid_work1, output: grid_work2
    desc = ((dgrid1[0, 0, -1] + dgrid1[0, 0, 1]) * coef[0] +
            (dgrid1[0, -1, 0] + dgrid1[0, 1, 0]) * coef[1] +
            (dgrid1[-1, 0, 0] + dgrid1[1, 0, 0]) * coef[2] +
            dgrid1[0, 0, 0] * coef[3])
    kernels.append(vp.sca.create_kernel(desc, desc_o=dgrid2[0, 0, 0]))
    # input: grid_work2, output: grid_work1
    desc = ((dgrid2[0, 0, -1] + dgrid2[0, 0, 1]) * coef[0] +
            (dgrid2[0, -1, 0] + dgrid2[0, 1, 0]) * coef[1] +
            (dgrid2[-1, 0, 0] + dgrid2[1, 0, 0]) * coef[2] +
            dgrid2[0, 0, 0] * coef[3])
    kernels.append(vp.sca.create_kernel(desc, desc_o=dgrid1[0, 0, 0]))
    return kernels

def execute_naive(grid_in, grid_out, coef):
    grid_out[1:-1, 1:-1, 1:-1] = (
        (grid_in[1:-1, 1:-1, 0:-2] + grid_in[1:-1, 1:-1, 2:]) * coef[0] +
        (grid_in[1:-1, 0:-2, 1:-1] + grid_in[1:-1, 2:,   1:-1]) * coef[1] +
        (grid_in[0:-2, 1:-1, 1:-1] + grid_in[2:,   1:-1, 1:-1]) * coef[2] +
        grid_in[1:-1, 1:-1, 1:-1] * coef[3])
    return grid_out

def exchange_data(grid, comm, rank, size):
    """ Exchange local boundary data
        '-' indicates xy planar.
            grid_root: -----------
            * transfer to upper process
            rank0    : -----
                           ^
                           |
            rank1    :    -----
                              ^
                              |
            rank2    :       -----
            * transfer to lower process
            rank0    : -----
                          |
                          v
            rank1    :    -----
                             |
                             v
            rank2    :       -----
    """
    if size == 1: return
    # transfer to upper process
    if rank == 0:
        peer_src = rank + 1
        peer_dst = MPI.PROC_NULL
    elif rank == size - 1:
        peer_src = MPI.PROC_NULL
        peer_dst = rank - 1
    else:
        peer_src = rank + 1
        peer_dst = rank - 1
    comm.Sendrecv(grid[1], dest=peer_dst, recvbuf=grid[-1], source=peer_src)
    # transfer to lower process
    if rank == 0:
        peer_src = MPI.PROC_NULL
        peer_dst = rank + 1
    elif rank == size - 1:
        peer_src = rank - 1
        peer_dst = MPI.PROC_NULL
    else:
        peer_src = rank - 1
        peer_dst = rank + 1
    comm.Sendrecv(grid[-2], dest=peer_dst, recvbuf=grid[0], source=peer_src)

def get_l2_norm(grid_work1, grid_work2, comm, dev):
    norm_local = dev.power(
        grid_work1[1:-1, 1:-1, 1:-1] - grid_work2[1:-1, 1:-1, 1:-1], 2).sum()
    l2_norm = dev.zeros_like(norm_local)
    comm.Allreduce(norm_local, l2_norm, op=MPI.SUM)
    l2_norm = dev.sqrt(l2_norm)
    return float(l2_norm)

def scatter_to_local_grid(grid_root, grid_local, count, displs, comm, rank, size):
    if size == 1:
        grid_local[...] = grid_root
    else:
        if rank == 0:
            begin = displs[0]
            end = begin + count[0]
            grid_local[...] = grid_root[begin:end]
        for r in range(1, size):
            if rank == 0:
                begin = displs[r]
                end = begin + count[r]
                comm.Send(grid_root[begin:end], dest=r)
            elif rank == r:
                comm.Recv(grid_local, source=0)

def gather_from_local_grid(grid_root, grid_local, count, displs, comm, rank, size):
    if size == 1:
        grid_root[...] = grid_local
    else:
        if rank == 0:
            begin = displs[0]
            end = begin + count[0]
            grid_root[begin:end] = grid_local
        for r in range(1, size):
            if rank == 0:
                begin = displs[r]
                end = begin + count[r]
                comm.Recv(grid_root[begin:end], source=r)
            elif rank == r:
                comm.Send(grid_local, dest=0)

def draw(fig, ax, xx, yy, grid, z, t):
    ax.set_xlabel("x[m]")
    ax.set_ylabel("y[m]")
    ax.set_title("z = {:4.3f} [m], timestep = {:>10d}".format(z, t))
    c = ax.pcolormesh(xx, yy, grid, cmap='coolwarm', vmin=0, vmax=100)
    return c

def thermal(dev):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    dx = LX / (NX + 1)
    dy = LY / (NY + 1)
    dz = LZ / (NZ + 1)
    coef = [
        (HC * DT) / (dx * dx),
        (HC * DT) / (dy * dy),
        (HC * DT) / (dz * dz),
        1.0 - 2.0 * HC * DT * (1 / (dx * dx) + 1 / (dy * dy) + 1 / (dz * dz)),
    ]
    mx = NX + 2
    my = NY + 2
    mz = NZ + 2
    # create base grid on root process
    if rank == 0:
        x = dev.linspace(0, LX, mx)
        y = dev.linspace(0, LY, my)
        z = dev.linspace(0, LZ, mz)
        zz, yy, xx = dev.meshgrid(z, y, x, indexing='ij')
        grid_root = dev.empty((mz, my, mx), dtype=DTYPE)
        initialize(grid_root, dev)
    else:
        grid_root = None
    # draw initial grid
    if rank == 0:
        fig, axes = plt.subplots(3, 2, figsize=(9, 9), constrained_layout=True)
        zstep = dev.linspace(0, mz, 5, dtype=int)[1:-1]
        for i, ax in enumerate(axes[:, 0]):
            zind = int(zstep[i])
            c = draw(fig, ax, xx[zind, :, :], yy[zind, :, :],
                     grid_root[zind, :, :], float(LZ * zind / mz), 0)
    # create local grid
    count, displs = get_count_and_displs(rank, size)
    lz_d = count[rank]
    grid_work1 = dev.empty((lz_d, my, mx), dtype=DTYPE)
    grid_work2 = dev.empty((lz_d, my, mx), dtype=DTYPE)
    scatter_to_local_grid(grid_root, grid_work1, count, displs, comm, rank, size)
    grid_work2[...] = grid_work1

    if dev.__name__ == 'nlcpy':
        # create stencil kernels
        kernels = create_stencil_kernel(grid_work1, grid_work2, coef, dev)

    # execute difference method
    comm.Barrier()
    t0 = MPI.Wtime()
    loop_cnt = 0
    while True:
        if dev.__name__ == 'nlcpy':
            grid = kernels[loop_cnt % 2].execute()
        else:
            grid = execute_naive(
                grid_work1 if loop_cnt % 2 == 0 else grid_work2,
                grid_work2 if loop_cnt % 2 == 0 else grid_work1,
                coef)
        exchange_data(grid, comm, rank, size)
        if loop_cnt % CHECK_INTERVAL == 0:  # check convergence
            l2_norm = get_l2_norm(grid_work1, grid_work2, comm, dev)
            if rank == 0: print("loop_cnt: {:>10d}, l2_norm: {:>12.6f}".format(loop_cnt, l2_norm))
            if l2_norm < TOLERANCE:
                break
        loop_cnt += 1
    comm.Barrier()
    t1 = MPI.Wtime()
    if rank == 0:
        print("elapsed:", t1 - t0)

    gather_from_local_grid(grid_root, grid, count, displs, comm, rank, size)
    # draw latest grid and save figure
    if rank == 0:
        for i, ax in enumerate(axes[:, 1]):
            zind = int(zstep[i])
            c = draw(fig, ax, xx[zind, :, :], yy[zind, :, :],
                     grid_root[zind, :, :], float(LZ * zind / mz), loop_cnt)
        fig.colorbar(c, ax=axes[:, 1], location='bottom', label='T[$^{\circ}$C]')
        plt.savefig('img_thermal_{}.png'.format(dev.__name__))

    if dev.__name__ == 'nlcpy':
        # destroy stencil kernels
        for kern in kernels:
            vp.sca.destroy_kernel(kern)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dev', type=str, required=True, choices=['vh', 've'],
                        help='Execution device')
    args = parser.parse_args()

    # set module
    if args.dev == 'vh':
        import numpy as np
        dev = np
    elif args.dev == 've':
        import nlcpy as vp
        dev = vp
    else:
        raise ValueError

    thermal(dev)
