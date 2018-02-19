from __future__ import division
from pySDC.core.Hooks import hooks


class fault_hook(hooks):

    def pre_run(self, step, level_number):

        super(fault_hook, self).pre_run(step, level_number)

        L = step.levels[level_number]

        L.sweep.reset_fault_stats()

    def post_run(self, step, level_number):

        super(fault_hook, self).post_run(step, level_number)

        L = step.levels[level_number]

        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=step.status.iter,
                          sweep=L.status.sweep, type='fault_stats', value=L.sweep.fault_stats)
