from mpi4py import MPI
import numpy as np

import sys

from pySDC.implementations.datatype_classes.complex_mesh import mesh

def main():

    # set MPI communicator
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    left_rank = (rank - 1) % size
    right_rank = (rank + 1) % size

    if len(sys.argv) == 2:
        N = int(sys.argv[-1])
    else:
        N = 10

    buf_send = mesh(init=N, val=0.0 + (rank + 1) * 1j)
    sum = mesh(init=N, val=0.0 + 0.0 * 1j)

    Emat = np.zeros((size, size))

    for p in range(size):
        Emat[p, :] = 10 ** (-p)

    # if rank == 0:
    #     print(Emat)

    t0 = MPI.Wtime()

    for p in range(size):
        col_index = (rank + p + 1) % size
        req = comm.isend(buf_send, dest=left_rank, tag=(rank + p) % size)
        buf_recv = comm.recv(source=right_rank, tag=col_index)

        sum += Emat[rank, col_index] * buf_recv
        req.wait()
        buf_send = buf_recv

    t1 = MPI.Wtime()

    print(sum.values)

    tmax = comm.reduce(t1 - t0, op=MPI.MAX)

    if rank == 0:
        print(tmax)

    MPI.Finalize()


if __name__ == "__main__":
    main()
