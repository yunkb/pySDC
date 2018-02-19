import numpy as np

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

from pySDC.helpers.pysdc_helper import FrozenClass


class _fault_stats(FrozenClass):
    def __init__(self):
        self.nfaults_called = 0
        self.nfaults_injected_u = 0
        self.nfaults_injected_f = 0
        self.nfaults_detected = 0
        self.ncorrection_attempts = 0
        self.nfaults_missed = 0
        self.nfalse_positives = 0

        self._freeze()


class implicit_sweeper_faults(generic_implicit):
    """
    LU sweeper using LU decomposition of the Q matrix for the base integrator, special type of generic implicit sweeper

    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        if 'allow_multiple_faults_per_iteration' not in params:
            params['allow_multiple_faults_per_iteration'] = True
        if 'allow_multiple_faults_per_run' not in params:
            params['allow_multiple_faults_per_run'] = True
        assert not(params['allow_multiple_faults_per_iteration'] and not params['allow_multiple_faults_per_run']), \
            'ERROR: multiple faults per iteration allowed, but not per run'

        if 'bitflip_probability' not in params:
            params['bitflip_probability'] = 1.0
        assert 0.0 <= params['bitflip_probability'] <= 1.0, \
            'ERROR: bitflip probability has to be between 0 and 1, got %s' % params['bitflip_probability']

        # call parent's initialization routine
        super(implicit_sweeper_faults, self).__init__(params)

        self.fault_stats = _fault_stats()

        self.fault_detected = False
        self.fault_at_u = False
        self.fault_at_f = False
        self.fault_injected_iteration = False
        self.fault_injected_run = False

    def reset_fault_stats(self):
        """
        Helper method to reset all fault related stats and flags. Will be called after the run in post-processing.
        """

        self.fault_stats = _fault_stats()
        self.fault_detected = False
        self.fault_at_u = False
        self.fault_at_f = False
        self.fault_injected_iteration = False
        self.fault_injected_run = False

    def set_fault(self):
        """
        Routine to check if bitflip should be done based on input, history and probabilities
        """

        if (not self.fault_injected_iteration or self.params.allow_multiple_faults_per_iteration) and \
                (not self.fault_injected_run or self.params.allow_multiple_faults_per_run):

            bitflip = np.random.rand(1)

            if bitflip < self.params.bitflip_probability:

                u_or_f = np.random.randint(2)
                if u_or_f == 0:
                    self.fault_at_u = True
                else:
                    self.fault_at_f = True

    def inject_fault(self, type=None, target=None):
        """
        Main method to inject a fault if set_faults() as decided this is necessary

        Args:
            type (str): string describing whether u of f should be affected
            target: data to be modified
        """

        if type == 'u':

            # do something to target = u here!
            target.values[19] = -1000

            self.fault_stats.nfaults_injected_u += 1
            # Reset indicator for where fault occured
            self.fault_at_u = False
        elif type == 'f':

            # do something to target = f here!
            target.values[19] = -1000

            self.fault_stats.nfaults_injected_f += 1
            # Reset indicator for where fault occured
            self.fault_at_f = False
        else:
            print('ERROR: wrong fault type specified, got %s' %type)
            exit()
        self.fault_injected_run = True
        self.fault_injected_iteration = True

    def detect_fault(self):

        if self.fault_injected_iteration:
            self.fault_stats.nfaults_detected += 1
        else:
            self.fault_stats.nfalse_positives += 1
        pass

    def correct_fault(self):

        self.fault_stats.ncorrection_attempts += 1
        pass

    def update_nodes(self):
        """
            Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

            Returns:
                None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QdF(u^k) + tau

        # get QF(u^k)
        integral = self.integrate()
        for m in range(M):

            # get -QdF(u^k)_m
            for j in range(M + 1):
                integral[m] -= L.dt * self.QI[m + 1, j] * L.f[j]

            # add initial value
            integral[m] += L.u[0]
            # add tau if associated
            if L.tau is not None:
                integral[m] += L.tau[m]

        # do the sweep
        for m in range(0, M):

            self.set_fault()

            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(m + 1):
                rhs += L.dt * self.QI[m + 1, j] * L.f[j]

            if self.fault_at_u:

                # implicit solve with prefactor stemming from the diagonal of Qd
                L.u[m + 1] = P.solve_system(rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1],
                                            L.time + L.dt * self.coll.nodes[m])

                self.inject_fault(type='u', target=L.u[m + 1])

                # update function values
                L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

            elif self.fault_at_f:

                # implicit solve with prefactor stemming from the diagonal of Qd
                L.u[m + 1] = P.solve_system(rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1],
                                            L.time + L.dt * self.coll.nodes[m])

                # update function values
                L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

                self.inject_fault(type='f', target=L.f[m + 1])

            else:

                # implicit solve with prefactor stemming from the diagonal of Qd
                L.u[m + 1] = P.solve_system(rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1],
                                            L.time + L.dt * self.coll.nodes[m])

                # update function values
                L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

            self.detect_fault()

            if self.fault_detected:
                self.correct_fault()

        self.fault_injected_iteration = False

        # indicate presence of new values at this level
        L.status.updated = True

        return None
