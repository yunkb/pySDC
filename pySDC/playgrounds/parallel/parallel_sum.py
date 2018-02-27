from mpi4py import MPI
import numpy as np

import sys

from pySDC.implementations.datatype_classes.complex_mesh import mesh


def main():

    # set MPI communicator
    comm = MPI.COMM_WORLD

    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if len(sys.argv) == 3:
        color = int(world_rank / int(sys.argv[2]))
    else:
        color = int(world_rank / 1)

    space_comm = comm.Split(color=color)
    space_rank = space_comm.Get_rank()
    space_size = space_comm.Get_size()

    if len(sys.argv) == 3:
        color = int(world_rank % int(sys.argv[2]))
    else:
        color = int(world_rank / world_size)

    time_comm = comm.Split(color=color)
    time_rank = time_comm.Get_rank()
    time_size = time_comm.Get_size()

    print("IDs (world, space, time):  %i / %i -- %i / %i -- %i / %i" % (world_rank, world_size, space_rank, space_size,
                                                                        time_rank, time_size))

    left_time_rank = (time_rank - 1) % time_size
    right_time_rank = (time_rank + 1) % time_size

    if len(sys.argv) >= 2:
        N = int(2 ** int(sys.argv[1]) / space_size)
    else:
        N = int(2 ** 10 / space_size)

    if time_rank == 0:
        print('ID %i -- Number of DoFs in space: %i' % (space_rank, N))

    buf_send = mesh(init=N, val=0.0 + (space_rank + 1) * 1j)
    sum = mesh(init=N, val=0.0 + 0.0 * 1j)

    Emat = np.zeros((time_size, time_size))

    for p in range(time_size):
        Emat[p, :] = 10 ** (-p)

    # if rank == 0:
    #     print(Emat)

    t0 = MPI.Wtime()

    for p in range(time_size):
        col_index = (time_rank + p + 1) % time_size
        req = time_comm.isend(buf_send, dest=left_time_rank, tag=(time_rank + p) % time_size)
        buf_recv = time_comm.recv(source=right_time_rank, tag=col_index)

        sum += Emat[time_rank, col_index] * buf_recv
        req.wait()
        buf_send = buf_recv

    t1 = MPI.Wtime()

    print("ID (space/time) %i/%i: Sum = %s" % (space_rank, time_rank, sum.values))

    tmax = comm.reduce(t1 - t0, op=MPI.MAX, root=0)

    if world_rank == 0:
        print(tmax)

    MPI.Finalize()


if __name__ == "__main__":
    main()
