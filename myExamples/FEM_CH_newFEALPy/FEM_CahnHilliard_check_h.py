#!/anaconda3/envs/FEALPy/bin python3.9
# -*- coding: utf-8 -*-
# ---
# @File: FEM_CahnHilliard_check_h.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Time: 2025/2/6
# ---

__doc__ = """
The fealpy-FEM program for Cahn-Hilliard equation.
The ref: 2019 (JCP YangZhiguo) An unconditionally energy-stable scheme based on an implicit auxiliary energy variable for 
            incompressible two-phase flows with different densities involving only precomputable coefficient matrices.pdf
"""

from matplotlib import pyplot as plt
from fealpy.utils import timer
from fealpy import logger
logger.setLevel('WARNING')
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from CahnHilliard2DData_newFEALPy import CahnHilliardData0
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.typing import TensorLike, Index, _S
from fealpy.fem.integrator import LinearInt, OpInt, CellInt, CoefLike, enable_cache
from fealpy.fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    ScalarMassIntegrator,
    DirichletBC
)
from fealpy.solver import cg
from fealpy.old.timeintegratoralg import UniformTimeLine

bm.set_backend('numpy')
# bm.set_backend('pytorch')
# # bm.set_default_device('cuda')
# bm.set_default_device('cpu')


class FEMCahnHilliardModel2d(LinearInt, OpInt, CellInt):
    """这里本质上是作为 FEALPy 中的一个 'integrator'-类 进行定义的,
    只是这个类的名字没有取为 'ScalarDiffusionIntegrator' 这样的名字.
    """
    def __init__(self, pde, mesh, p, timeline):
        super().__init__()
        self.pde, self.mesh, self.p = pde, mesh, p
        self.q = p + 3
        self.timeline, self.dt = timeline, timeline.dt
        self.space = LagrangeFESpace(mesh, p=p)
        self.uh = self.space.function()
        self.wh = self.space.function()
        self.idxNeuEdge = self.set_Neumann_edge()

        # |---
        bform = BilinearForm(self.space)
        bform.add_integrator(ScalarDiffusionIntegrator())
        self.stiffM = bform.assembly()  # stiff-matrix

        mform = BilinearForm(self.space)
        mform.add_integrator(ScalarMassIntegrator())
        self.massM = mform.assembly()  # mass-matrix

    @enable_cache
    def fetch(self, space: _FS):
        mesh = getattr(space, 'mesh', None)

        index: Index = _S
        cm = mesh.entity_measure('cell', index=index)
        fm = mesh.entity_measure('face', index=index)
        q = self.q

        # |--- cell-based phi
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()  # bcs:(NQ,GD+1),  ws:(NQ,)
        pp = mesh.bc_to_point(bcs)  # (NC,NQ,GD)
        phi = space.basis(bcs, index=index)  # (1,NQ,ldof)
        gphi = space.grad_basis(bcs, index=index)  # (NC,NQ,ldof,GD)

        # |--- face-based phi
        indexF = self.idxNeuEdge
        qfF = mesh.quadrature_formula(q, 'edge')
        bcsF, wsF = qfF.get_quadrature_points_and_weights()
        ppF = mesh.bc_to_point(bcsF, index=indexF)  # (NE,NQ_E,GD)
        phiF = space.cell_basis_on_edge(bcsF, eindex=indexF)  # (NE,NQ_E,ldof)
        gphiF = space.cell_grad_basis_on_edge(bcsF, eindex=indexF)  # (NE,NQ_E,ldof,GD)

        return phi, gphi, cm, pp, bcs, ws, phiF, gphiF, fm, ppF, bcsF, wsF

    def setCoefficient_T1stOrder(self, dt_minimum=None):
        pde = self.pde
        dt_min = self.dt if dt_minimum is None else dt_minimum
        m = pde.m
        epsilon = pde.epsilon

        s = bm.sqrt(4 * epsilon / (m * dt_min))
        alpha = 1. / (2 * epsilon) * (-s + bm.sqrt(abs(s ** 2 - 4 * epsilon / (m * self.dt))))
        return s, alpha

    def setCoefficient_T2ndOrder(self, dt_minimum=None):
        pde = self.pde
        dt_min = self.dt if dt_minimum is None else dt_minimum
        m = pde.m
        epsilon = pde.epsilon

        s = bm.sqrt(4 * (3/2) * epsilon / (m * dt_min))
        alpha = 1. / (2 * epsilon) * (-s + bm.sqrt(abs(s ** 2 - 4 * (3/2) * epsilon / (m * self.dt))))
        return s, alpha

    def set_Neumann_edge(self, idxNeuEdge=None):
        if idxNeuEdge is not None:
            return idxNeuEdge
        mesh = self.mesh
        edge2cell = mesh.edge_to_cell()
        bdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # the bool vars, to get the boundary edges

        isNeuEdge = bdEdge  # here, we first set all the boundary edges are Neu edges
        idxNeuEdge, = bm.nonzero(isNeuEdge)  # (NE_Dir,)
        return idxNeuEdge

    def set_Dirichlet_edge(self, idxDirEdge=None):
        if idxDirEdge is not None:
            return idxDirEdge
        mesh = self.mesh
        edge2cell = mesh.edge_to_cell()
        isBdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # (NE,), the bool vars, to get the boundary edges

        isDirEdge = isBdEdge  # here, we set all the boundary edges are Dir edges
        idxDirEdge, = bm.nonzero(isDirEdge)  # (NE_Dir,)
        return idxDirEdge

    def uh_grad_value_at_faces(self, vh, f_bcs, cellidx, localidx):
        cell2dof = self.space.dof.cell_to_dof()
        f_gphi = self.space.edge_grad_basis(f_bcs, cellidx, localidx)  # (NE,NQ,cldof,GD)
        val = bm.einsum('ik, ijkm->jim', vh[cell2dof[cellidx]], f_gphi)  # (NQ,NE,GD)
        return val

    def grad_free_energy_at_faces(self, uh, f_bcs, idxBdEdge, cellidx, localidx):
        """
        1. Compute the grad of free energy at FACE Gauss-integration points (barycentric coordinates).
        2. In this function, the free energy has NO coefficients.
        -------
        :param uh:
        :param f_bcs: f_bcs.shape: (NQ,(GD-1)+1)
        :return:
        """

        uh_val = self.space.value(uh, f_bcs)[..., idxBdEdge]  # (NQ,NBE)
        guh_val = self.uh_grad_value_at_faces(uh, f_bcs, cellidx, localidx)  # (NQ,NBE,GD)

        guh_val[..., 0] = 3 * uh_val ** 2 * guh_val[..., 0] - guh_val[..., 0]
        guh_val[..., 1] = 3 * uh_val ** 2 * guh_val[..., 1] - guh_val[..., 1]
        return guh_val  # (NQ,NBE,2)

    def grad_free_energy_at_cells(self, uh, c_bcs):
        """
        1. Compute the grad of free energy at CELL Gauss-integration points (barycentric coordinates).
        2. In this function, the free energy has NO coefficients.
        -------
        :param uh:
        :param c_bcs: c_bcs.shape: (NQ,GD+1)
        :return:
        """

        uh_val = self.space.value(uh, c_bcs)  # (NQ,NC)
        guh_val = self.space.grad_value(uh, c_bcs)  # (NQ,NC,2)

        guh_val[..., 0] = 3 * uh_val ** 2 * guh_val[..., 0] - guh_val[..., 0]
        guh_val[..., 1] = 3 * uh_val ** 2 * guh_val[..., 1] - guh_val[..., 1]
        return guh_val  # (NQ,NC,2)

    def CH_Solver_T1stOrder(self):
        """这里其实是当做 'integrator'-类中的 'assembly()' 函数定义的.
        """
        pde, dt, timeline, GD = self.pde, self.dt, self.timeline, self.mesh.GD
        dt_min = pde.dt_min if hasattr(pde, 'dt_min') else dt
        s, alpha = self.setCoefficient_T1stOrder(dt_minimum=dt_min)
        m, epsilon, eta = pde.m, pde.epsilon, pde.eta

        print('    # #################################### #')
        print('      Time 1st-order scheme')
        print('    # #################################### #')
        print('    # ------------ parameters ------------ #')
        print('    s = %.4e,  alpha = %.4e,  m = %.4e,  epsilon = %.4e,  eta = %.4e' % (s, alpha, m, epsilon, eta))
        print('    t0 = %.4e,  T = %.4e, dt = %.4e' % (timeline.T0, timeline.T1, dt))
        print(' ')

        idxNeuEdge = self.idxNeuEdge
        nBd = self.mesh.face_unit_normal(index=idxNeuEdge)  # (NBE,2)
        NeuCellIdx = self.mesh.edge2cell[idxNeuEdge, 0]
        NeuLocalIdx = self.mesh.edge2cell[idxNeuEdge, 2]
        neu_face_measure = self.mesh.entity_measure('face', index=idxNeuEdge)  # (Nneu,2)
        phi, gphi, cm, pp, bcs, ws, phiF, gphiF, fm, ppF, bcsF, wsF = self.fetch(self.space)

        # # time-looping
        print('    # ------------ begin the time-looping ------------ #')
        for nt in range(timeline.NL):
            currt_t = timeline.current_time_level()
            next_t = timeline.next_time_level()

            if nt % max([int(timeline.NL / 10), 1]) == 0:
                print('    currt_t = %.4e' % currt_t)
            if nt == 0:
                # the initial value setting
                u0_c = pde.solution(pp, pde.t0)  # (NC,NQC)
                gu0_c = pde.gradient(pp, pde.t0)  # (NC,NQC,2)
                u0_f = pde.solution(ppF, pde.t0)  # (NBE,NQF)
                gu0_f = pde.gradient(ppF, pde.t0)  # (NBE,NQF,2)
                grad_free_energy_c = epsilon / eta ** 2 * (3 * bm.repeat(u0_c[..., bm.newaxis], GD, axis=-1) ** 2 * gu0_c - gu0_c)
                grad_free_energy_f = epsilon / eta ** 2 * (3 * bm.repeat(u0_f[..., bm.newaxis], GD, axis=-1) ** 2 * gu0_f - gu0_f)

                uh_val = u0_c.copy()  # (NC,NQC)
                guh_val_c = gu0_c.copy()  # (NC,NQC,2)
                guh_val_f = gu0_f.copy()  # (NBE,NQF,2)
                del u0_c, gu0_c, u0_f, gu0_f
            else:
                0



def plot_mesh(mesh):
    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    mesh.find_node(axes, showindex=True)
    mesh.find_cell(axes, showindex=True)
    mesh.find_edge(axes, showindex=True)

    # 设置标题和轴标签
    axes.set_title('Triangle Mesh')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_aspect('equal')
    plt.show()


def main():
    t0, T, nt = 0., 0.02, 2000
    p = 1  # the polynomial order
    NN = 4
    maxit = 4
    timeline = UniformTimeLine(0, T, nt)
    dt = timeline.dt
    print(f"t0={t0},  T={T},  dt={dt:.4e}")
    tmr = timer()
    next(tmr)

    pde = CahnHilliardData0(t0=t0, T=T)
    pdePars = {'m': 1e-3, 's': 1, 'alpha': 1, 'epsilon': 1e-3, 'eta': 1e-1}  # value of parameters
    pde.setPDEParameters(pdePars)
    mesh = TriangleMesh.from_box([0, 1, 0, 1], NN, NN)
    # plot_mesh(mesh)

    errorType = ['$|| u - u_h||_{\\Omega,0}$']
    errorMatrix = bm.zeros((1, maxit), dtype=bm.float64)
    tmr.send('网格 和 pde 生成时间')

    for i in range(maxit):
        space = LagrangeFESpace(mesh, p=p)
        tmr.send(f'第{i}次空间时间')
        ch_fem = FEMCahnHilliardModel2d(pde, mesh, p, timeline)
        ch_fem.CH_Solver_T1stOrder()




if __name__ == '__main__':
    main()

