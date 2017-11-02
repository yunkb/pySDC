from __future__ import division
from pySDC.core.Hooks import hooks

import numpy as np


class dump(hooks):
    """
    Hook class to add output of error
    """

    def post_iteration(self, step, level_number):
        """
        Default routine called after each iteration
        Args:
            step: the current step
            level_number: the current level number
        """

        super(dump, self).post_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        with open(P.params.filename, 'w') as f:
            np.savez(f, [L.u[m].values for m in range(1, L.sweep.coll.num_nodes + 1)])
