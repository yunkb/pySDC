import numpy as np
import dill

from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from pySDC.implementations.problem_classes.GeneralizedFisher_1D_FD_implicit import generalized_fisher
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
    sweeper_params['detector_threshold'] = 1E-10

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


def reaction_setup():
    """
    Setup routine for diffusion-reaction test with Newton solver
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
    sweeper_params['detector_threshold'] = 1E-10

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 1.0
    problem_params['lambda0'] = 2.0
    problem_params['newton_maxiter'] = 20
    problem_params['newton_tol'] = 1E-10
    problem_params['interval'] = (-5, 5)
    problem_params['nvars'] = 127

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = generalized_fisher  # pass problem class
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
        Tend = 0.25
    elif type == 'reaction':
        description, controller_params = reaction_setup()
        # set time parameters
        t0 = 0.0
        Tend = 0.25
    else:
        raise ValueError('No valis setup type provided, aborting..')

    out = '\nCLEAN RUN: Working with %s setup..' % type
    f.write(out + '\n')
    print(out)

    # instantiate controller
    controller = allinclusive_multigrid_nonMPI(num_procs=1, controller_params=controller_params,
                                               description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # this is where the iteration is happening
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    uex = P.u_exact(Tend)

    # filter statistics by type (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts = sort_stats(filtered_stats, sortby='time')

    print('After %s iterations, I got an error of %6.4e for a clean run!' % (iter_counts[0][1], abs(uex - uend)))

    return iter_counts[0][1]


def run_faulty_simulations(type=None, niters=None, f=None):
    """
    A simple program to run faulty simulations

    Args:
        type (str): setup type
        niters (int): number of iterations in clean run
        f: file handler
    """

    if type == 'diffusion':
        description, controller_params = diffusion_setup()
        # set time parameters
        t0 = 0.0
        Tend = 0.25
    elif type == 'reaction':
        description, controller_params = reaction_setup()
        # set time parameters
        t0 = 0.0
        Tend = 0.25
    else:
        raise ValueError('No valid setup type provided, aborting..')

    out = '\nFAULTY RUN: Working with %s setup..' % type
    f.write(out + '\n')
    print(out)

    filehandle_injections = open('dump_injections_' + type + '.txt', 'w')
    controller_params['hook_class'] = fault_hook
    description['sweeper_params']['allow_fault_correction'] = False
    description['sweeper_params']['dump_injections_filehandle'] = filehandle_injections
    description['sweeper_params']['niters'] = niters

    # instantiate controller
    controller = allinclusive_multigrid_nonMPI(num_procs=1, controller_params=controller_params,
                                               description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    nruns = 10
    results = []
    for nr in range(nruns):

        # this is where the iteration is happening
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        results.append(stats)

    filehandle_injections.close()

    dill.dump(results, open("results_" + type + ".pkl", "wb"))


def process_statistics(type=None, results=None):

    minlen = 1000
    nruns = 0
    for stats in results:
        residuals = sort_stats(filter_stats(stats, type='residual_post_iteration'), sortby='iter')
        minlen = min(minlen, len(residuals))
        nruns += 1

    minres = np.zeros(minlen)
    minres[:] = 1000
    maxres = np.zeros(minlen)
    meanres = np.zeros(minlen)
    for stats in results:

        # Some black magic to extract fault stats ouf of monstrous stats object
        fault_stats = sort_stats(filter_stats(stats, type='fault_stats'), sortby='type')[0][1]
        # Some black magic to extract number of iterations ouf of monstrous stats object
        residuals = sort_stats(filter_stats(stats, type='residual_post_iteration'), sortby='iter')
        for i in range(minlen):
            minres[i] = min(minres[i], residuals[i][1])
            maxres[i] = max(maxres[i], residuals[i][1])
            meanres[i] += residuals[i][1]

        # Example output of what we now can do
        print(fault_stats.nfaults_injected_u, fault_stats.nfaults_injected_f, fault_stats.nfaults_detected,
              fault_stats.nfalse_positives, fault_stats.nfalse_positives_in_correction,
              fault_stats.nfaults_missed, fault_stats.nclean_steps)

        # print('Mean number of iterations for this run: %s' % np.mean(niters))
        # print()

    meanres /= nruns
    print(meanres)
    # print(minres)
    # print(maxres)


def main():

    f = open('generate_statistics.txt', 'w')

    type = 'diffusion'
    niters = run_clean_simulations(type=type, f=f)
    run_faulty_simulations(type=type, niters=niters, f=f)
    results = dill.load(open("results_" + type + ".pkl", "rb"))
    process_statistics(type=type, results=results)

    type = 'reaction'
    niters = run_clean_simulations(type=type, f=f)
    run_faulty_simulations(type=type, niters=niters, f=f)
    results = dill.load(open("results_" + type + ".pkl", "rb"))
    process_statistics(type=type, results=results)

    f.close()


if __name__ == "__main__":
    main()
