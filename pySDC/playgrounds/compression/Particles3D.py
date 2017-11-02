import numpy as np
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI
from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.datatype_classes.particles import particles, fields
from pySDC.tutorial.step_3.HookClass_Particles import particle_hook

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    """
    A simple test program to run IMEX SDC for a single time step of the penning trap example
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-08
    level_params['dt'] = 1.0 / 16

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 4

    # initialize problem parameters for the Penning trap
    problem_params = dict()
    problem_params['omega_E'] = 4.9  # E-field frequency
    problem_params['omega_B'] = 25.0  # B-field frequency
    problem_params['u0'] = np.array([[10, 0, 0], [100, 0, 100], [1], [1]])  # initial center of positions
    problem_params['nparts'] = 2**17  # number of particles in the trap
    problem_params['sig'] = 0.1  # smoothing parameter for the forces

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 1

    # initialize controller parameters
    controller_params = dict()
    controller_params['hook_class'] = particle_hook  # specialized hook class for more statistics and output

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = penningtrap
    description['problem_params'] = problem_params
    description['dtype_u'] = particles
    description['dtype_f'] = fields
    description['sweeper_class'] = boris_2nd_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller (no controller parameters used here)
    controller = allinclusive_classic_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = level_params['dt']

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_init()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(uinit.pos.values[0::3], uinit.pos.values[1::3], uinit.pos.values[2::3])
    # plt.show()

    with open('particle_dump_128k', 'w') as f:
        np.savez(f, [uinit.pos.values, uinit.vel.values])

    with open('particle_dump_128k', 'w') as f:
        np.savez(f, [uend.pos.values, uend.vel.values])


if __name__ == "__main__":
    main()
