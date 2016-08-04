from __future__ import division
import numpy as np

from pySDC.Transfer import transfer
from pySDC.datatype_classes.fenics_mesh import fenics_mesh,rhs_fenics_mesh

import dolfin as df

class mesh_to_mesh_fenics(transfer):
    """
    Custon transfer class, implements Transfer.py

    This implementation can restrict and prolong between fenics meshes

    Attributes:
        fine: reference to the fine level
        coarse: reference to the coarse level
        init_f: number of variables on the fine level (whatever init represents there)
        init_c: number of variables on the coarse level (whatever init represents there)
    """

    def __init__(self,fine_level,coarse_level,params):
        """
        Initialization routine

        Args:
            fine_level: fine level connected with the transfer operations (passed to parent)
            coarse_level: coarse level connected with the transfer operations (passed to parent)
            params: parameters for the transfer operators
        """

        # invoke super initialization
        super(mesh_to_mesh_fenics,self).__init__(fine_level,coarse_level,params)

        pass

    def restrict_space(self,F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """

        if isinstance(F,fenics_mesh):
            u_coarse = fenics_mesh(self.init_c)
            u_coarse.values = df.interpolate(F.values,u_coarse.V)
        elif isinstance(F,rhs_fenics_mesh):
            u_coarse = rhs_fenics_mesh(self.init_c)
            u_coarse.impl.values = df.interpolate(F.impl.values,u_coarse.impl.V)
            u_coarse.expl.values = df.interpolate(F.expl.values,u_coarse.expl.V)

        return u_coarse


    def project_space(self,F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """

        if isinstance(F,fenics_mesh):
            u_coarse = fenics_mesh(self.init_c)
            u_coarse.values = df.project(F.values,u_coarse.V)
        elif isinstance(F,rhs_fenics_mesh):
            u_coarse = rhs_fenics_mesh(self.init_c)
            u_coarse.impl.values = df.project(F.impl.values,u_coarse.impl.V)
            u_coarse.expl.values = df.project(F.expl.values,u_coarse.expl.V)

        return u_coarse

    def prolong_space(self,G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """

        if isinstance(G,fenics_mesh):
            u_fine = fenics_mesh(self.init_f)
            u_fine.values = df.interpolate(G.values,u_fine.V)
        elif isinstance(G,rhs_fenics_mesh):
            u_fine = rhs_fenics_mesh(self.init_f)
            u_fine.impl.values = df.interpolate(G.impl.values,u_fine.impl.V)
            u_fine.expl.values = df.interpolate(G.expl.values,u_fine.expl.V)

        return u_fine

    def restrict(self):
        """
        Space-time restriction routine

        The routine applies the spatial restriction operator to teh fine values on the fine nodes, then reevaluates f
        on the coarse level. This is used for the first part of the FAS correction tau via integration. The second part
        is the integral over the fine values, restricted to the coarse level. Finally, possible tau corrections on the
        fine level are restricted as well.
        """

        # get data for easier access
        F = self.fine
        G = self.coarse

        PF = F.prob
        PG = G.prob

        SF = F.sweep
        SG = G.sweep

        # only of the level is unlocked at least by prediction
        assert F.status.unlocked
        # can only do space-restriction so far
        assert np.array_equal(SF.coll.nodes, SG.coll.nodes)

        # restrict fine values in space, reevaluate f on coarse level
        G.u[0] = self.project_space(F.u[0])
        G.f[0] = PG.eval_f(G.u[0], G.time)
        for m in range(1, SG.coll.num_nodes + 1):
            G.u[m] = self.project_space(F.u[m])
            G.f[m] = PG.eval_f(G.u[m], G.time + G.dt * SG.coll.nodes[m - 1])

        # build coarse level tau correction part
        tauG = G.sweep.integrate()
        for m in range(SG.coll.num_nodes):
            tauG[m] = PG.apply_mass_matrix(G.u[m+1]) - tauG[m]

        # build fine level tau correction part
        tauF = F.sweep.integrate()

        # restrict fine level tau correction part
        tauFG = []
        for m in range(SG.coll.num_nodes):
            tauFG.append(self.restrict_space(PF.apply_mass_matrix(F.u[m+1]) - tauF[m]))
            # tauFG.append(self.restrict_space(tauF[m]))

        # build tau correction, also restrict possible tau correction from fine
        for m in range(SG.coll.num_nodes):
            G.tau[m] =  tauG[m] - tauFG[m]
            # G.tau[m] = tauFG[m] - tauG[m]

            if F.tau is not None:
                G.tau[m] += self.restrict_space(F.tau[m])

        # save u and rhs evaluations for interpolation
        for m in range(SG.coll.num_nodes + 1):
            G.uold[m] = PG.dtype_u(G.u[m])
            G.fold[m] = PG.dtype_f(G.f[m])

        # TODO: WTF? Do I need this?
        G.u[0] = self.restrict_space(PF.apply_mass_matrix(F.u[0]))

        # works as a predictor
        G.status.unlocked = True

        return None

    def prolong(self):
        """
        Space-time prolongation routine

        This routine applies the spatial prolongation routine to the difference between the computed and the restricted
        values on the coarse level and then adds this difference to the fine values as coarse correction.
        """

        # get data for easier access
        F = self.fine
        G = self.coarse

        PF = F.prob

        SF = F.sweep
        SG = G.sweep

        # only of the level is unlocked at least by prediction or restriction
        assert G.status.unlocked
        # can only do space-restriction so far
        assert np.array_equal(SF.coll.nodes,SG.coll.nodes)

        # build coarse correction
        # need to restrict F.u[0] again here, since it might have changed in PFASST

        # G.uold[0] = self.project_space(F.u[0])
        #
        # F.u[0] += self.prolong_space(G.u[0] - G.uold[0])
        # F.f[0] = PF.eval_f(F.u[0],F.time)

        for m in range(1,SF.coll.num_nodes+1):
            F.u[m] += self.prolong_space(G.u[m] - G.uold[m])
            F.f[m] = PF.eval_f(F.u[m],F.time+F.dt*SF.coll.nodes[m-1])

        return None
