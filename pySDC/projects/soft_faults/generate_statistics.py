import numpy as np

from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI

from pySDC.projects.soft_faults.implicit_sweeper_faults import implicit_sweeper_faults
from pySDC.projects.soft_faults.FaultHooks import fault_hook

from pySDC.helpers.stats_helper import filter_stats, sort_stats


def diffusion_setup():
    """
    Setup routine for diffusion test
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.25
    level_params['nsweeps'] = 1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['spread'] = True

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = 127  # number of degrees of freedom for each level

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heat1d  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['dtype_u'] = mesh  # pass data type for u
    description['dtype_f'] = mesh  # pass data type for f
    description['sweeper_class'] = implicit_sweeper_faults  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters

    return description, controller_params


def run_clean_simulations(type=None, f=None):
    """
    A simple code to run fault-free simulations

    Args:
        type (str): setup type
        f: file handler
    """

    if type == 'diffusion':
        description, controller_params = diffusion_setup()
        # set time parameters
        t0 = 0.0
        Tend = 1.0
    else:
        raise ValueError('No valis setup type provided, aborting..')

    out = '\nCLEAN RUN: Working with %s setup..' % type
    f.write(out + '\n')
    print(out)

    controller_params['hook_class'] = fault_hook
    description['sweeper_params']['bitflip_probability'] = 0.0

    # instantiate controller
    controller = allinclusive_multigrid_nonMPI(num_procs=1, controller_params=controller_params,
                                               description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # this is where the iteration is happening
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by type (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts = sort_stats(filtered_stats, sortby='time')

    niters = np.array([item[1] for item in iter_counts])

    out = '   Mean number of iterations: %4.2f' % np.mean(niters)
    f.write(out + '\n')
    print(out)
    out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
    f.write(out + '\n')
    print(out)
    out = '   Max/min number of iterations: %2i -- %2i' % (max(niters), min(niters))
    f.write(out + '\n')
    print(out)
    out = '   Position of max/min number of iterations: %2i -- %2i' % (int(np.argmax(niters)), int(np.argmin(niters)))
    f.write(out + '\n')
    print(out)
    out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
    f.write(out + '\n')
    print(out)


def run_faulty_simulations(type=None, f=None):
    """
    A simple program to run faulty simulations

    Args:
        type (str): setup type
        f: file handler
    """

    if type == 'diffusion':
        description, controller_params = diffusion_setup()
        # set time parameters
        t0 = 0.0
        Tend = 1.0
    else:
        raise ValueError('No valis setup type provided, aborting..')

    out = '\nFAULTY RUN: Working with %s setup..' % type
    f.write(out + '\n')
    print(out)

    controller_params['hook_class'] = fault_hook
    description['sweeper_params']['allow_multiple_faults_per_iteration'] = False
    description['sweeper_params']['allow_multiple_faults_per_run'] = False
    description['sweeper_params']['allow_fault_correction'] = True
    description['sweeper_params']['detector_threshold'] = 1E-12
    description['sweeper_params']['bitflip_probability'] = 1.0

    # instantiate controller
    controller = allinclusive_multigrid_nonMPI(num_procs=1, controller_params=controller_params,
                                               description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    nruns = 3
    results = []
    for nr in range(nruns):

        # this is where the iteration is happening
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        results.append(stats)

    return results


def process_statistics(type=None, results=None):

    for stats in results:

        # Some black magic to extract fault stats ouf of monstrous stats object
        fault_stats = sort_stats(filter_stats(stats, type='fault_stats'), sortby='type')[0][1]
        # Some black magic to extract number of iterations ouf of monstrous stats object
        niter_stats = sort_stats(filter_stats(stats, type='niter'), sortby='time')
        niters = np.array([item[1] for item in niter_stats])

        # Example output of what we now can do
        print('Number of faults in u + f: %s + %s = %s' %
              (fault_stats.nfaults_injected_u, fault_stats.nfaults_injected_f,
               fault_stats.nfaults_injected_u + fault_stats.nfaults_injected_f))
        print('Mean number of iterations for this run: %s' % np.mean(niters))
        print()
    pass


def main():

    f = open('generate_statistics.txt', 'w')

    type = 'diffusion'
    run_clean_simulations(type=type, f=f)
    results = run_faulty_simulations(type=type, f=f)
    process_statistics(type=type, results=results)

    f.close()


if __name__ == "__main__":
    main()
