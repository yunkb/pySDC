from mpi4py import MPI
import numpy as np

from pySDC.implementations.problem_classes.HeatEquation_2D_FD_periodic import heat2d_periodic
from pySDC.implementations.problem_classes.HeatEquation_2D_FD import heat2d
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI
from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI
from pySDC.implementations.controller_classes.allinclusive_classic_MPI import allinclusive_classic_MPI
from pySDC.implementations.controller_classes.allinclusive_multigrid_MPI import allinclusive_multigrid_MPI

from pySDC.helpers.stats_helper import filter_stats, sort_stats


def main():
    """
    A simple test program to do PFASST runs for the 2D heat equation
    """

    # set MPI communicator
    comm = MPI.COMM_WORLD

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.125
    level_params['nsweeps'] = 1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['spread'] = False

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 1.0  # diffusion coefficient
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = [(127, 127), (63, 63)]  # number of degrees of freedom for each level

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 2
    space_transfer_params['periodic'] = False
    # space_transfer_params['finter'] = True

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['predict'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heat2d  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['dtype_u'] = mesh  # pass data type for u
    description['dtype_f'] = mesh  # pass data type for f
    description['sweeper_class'] = generic_implicit  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    # instantiate controller
    controller = allinclusive_multigrid_MPI(controller_params=controller_params, description=description, comm=comm)
    # controller = allinclusive_classic_MPI(controller_params=controller_params, description=description, comm=comm)

    # get initial values on finest level
    P = controller.S.levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)
    err = abs(uex - uend)

    rank = comm.Get_rank()
    size = comm.Get_size()

    timing_run = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')
    tmax = comm.reduce(timing_run[0][1], op=MPI.MAX, root=0)
    if rank == 0:
        print(tmax)

    timing_comm = filter_stats(stats, type='timing_comm')
    timing_sort = sort_stats(timing_comm, sortby='time')
    sum_time_comm = sum([item[1] for item in timing_sort])
    tcomm_max = comm.reduce(sum_time_comm, op=MPI.MAX, root=0)
    print(rank, sum_time_comm, tcomm_max)

    if False:

        # filter statistics by type (number of iterations)
        filtered_stats = filter_stats(stats, type='niter')

        # convert filtered statistics to list of iterations count, sorted by process
        iter_counts = sort_stats(filtered_stats, sortby='time')

        # compute and print statistics
        for item in iter_counts:
            out = 'Number of iterations for time %4.2f: %2i' % item
            print(out)

        niters = np.array([item[1] for item in iter_counts])
        out = '   Mean number of iterations: %4.2f' % np.mean(niters)
        print(out)
        out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
        print(out)
        out = '   Position of max/min number of iterations: %2i -- %2i' % \
              (int(np.argmax(niters)), int(np.argmin(niters)))
        print(out)
        out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
        print(out)

        print('CFL number: %4.2f' % (level_params['dt'] * problem_params['nu'] /
                                     (1.0 / problem_params['nvars'][0][0])**2))
        print('Error: %8.4e' % err)


if __name__ == "__main__":
    main()
