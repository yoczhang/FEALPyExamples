#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: CocurrentFlowModel2d.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Feb 16, 2022
# ---

__doc__ = """
The fealpy-FEM program for Variable-Coefficient coupled Cahn-Hilliard-Navier-Stokes equation, 
add the solver for \\xi.
  |___ Co-current flow problem.
"""


import numpy as np
from scipy.sparse import csr_matrix, spdiags, eye, bmat
# from fealpy.quadrature import FEMeshIntegralAlg
from scipy.sparse.linalg import spsolve
from fealpy.boundarycondition import DirichletBC
# from fealpy.decorator import timer
# from fealpy.functionspace import LagrangeFiniteElementSpace
# from sym_diff_basis import compute_basis
from FEM_CH_NS_Model2d import FEM_CH_NS_Model2d


class CocurrentFlowModel2d(FEM_CH_NS_Model2d):
    """
        注意:
            本程序所参考的数值格式是按照 $D(u)=\nabla u + \nabla u^T$ 来写的,
            如果要调整为 $D(u)=(\nabla u + \nabla u^T)/2$, 需要调整的地方太多了,
            所以为了保证和数值格式的一致性, 本程序也是严格按照 $D(u)=\nabla u + \nabla u^T$ 来编写的.
        """

    def __init__(self, pde, mesh, p, dt):
        super(CocurrentFlowModel2d, self).__init__(pde, mesh, p, dt)
        self.wh_part0 = self.space.function()
        self.wh_part1 = self.space.function()
        self.uh_part0 = self.space.function()
        self.uh_part1 = self.space.function()
        self.uh_last_part0 = self.space.function()
        self.uh_last_part1 = self.space.function()
        self.ph_part0 = self.space.function()
        self.ph_part1 = self.space.function()
        self.vel0_part0 = self.vspace.function()
        self.vel1_part0 = self.vspace.function()
        self.vel0_part1 = self.vspace.function()
        self.vel1_part1 = self.vspace.function()
        self.auxVel0_part0 = self.vspace.function()
        self.auxVel1_part0 = self.vspace.function()
        self.auxVel0_part1 = self.vspace.function()
        self.auxVel1_part1 = self.vspace.function()

        self.grad_mu_val = 0.  # 此项在 `update_mu_and_Xi()` 中更新.
        self.rho_bar_n = 0  # 此项在 `decoupled_NS_addXi_Solver_T1stOrder()` 中更新: 为了获得第 n 时间层的取值 (在 `update_mu_and_Xi()` 会用到).
        self.nu_bar_n = 0  # 此项在 `decoupled_NS_addXi_Solver_T1stOrder()` 中更新: 为了获得第 n 时间层的取值 (在 `update_mu_and_Xi()` 会用到).
        self.R_n = 0.  # 此项在 `update_mu_and_Xi()` 中更新.
        self.C0 = 1.  # 此项在 `update_mu_and_Xi()` 中, 以保证 E_n = \int H(\phi) + C0 > 0.
        self.Xi = 1.  # 此项在 `update_mu_and_Xi()` 中更新.
        self.s, self.alpha = self.set_CH_Coeff(dt_minimum=self.dt_min)

        if hasattr(self, 'idxNotPeriodicEdge') is False:
            self.idxPeriodicEdge0, self.idxPeriodicEdge1, self.idxNotPeriodicEdge = self.set_periodic_edge()

        # |--- setting periodic dofs
        self.periodicDof0, self.periodicDof1, self.notPeriodicDof = self.set_boundaryDofs(self.dof)
        #   |___ the phi_h- and p_h-related variables periodic dofs (using p-order polynomial)
        self.vPeriodicDof0, self.vPeriodicDof1, self.vNotPeriodicDof = self.set_boundaryDofs(self.vdof)
        #   |___ the velocity-related variables periodic dofs (using (p+1)-order polynomial)

        # |--- CH: setting algebraic system for periodic boundary condition
        self.auxM_CH = self.StiffMatrix + (self.alpha + self.s / self.pde.epsilon) * self.MassMatrix  # csr_matrix
        self.auxPeriodicM_CH = None
        self.orgM_CH = self.StiffMatrix - self.alpha * self.MassMatrix  # csr_matrix
        self.orgPeriodicM_CH = None

        # |--- NS: setting algebraic system for periodic boundary condition
        self.plsm = 1. / min(self.pde.rho0, self.pde.rho1) * self.StiffMatrix
        self.pPeriodicM_NS = None

        self.auxVLM = 1. / self.dt * self.vel_MM
        self.vAuxPeriodicM_NS = None

        self.VLM = 1. / self.dt * self.vel_MM + max(self.pde.nu0 / self.pde.rho0, self.pde.nu1 / self.pde.rho1) * self.vel_SM
        self.vOrgPeriodicM_NS = None




    def set_periodic_edge(self):
        """
        :return: idxPeriodicEdge0, 表示区域网格 `左侧` 的边
                 idxPeriodicEdge1, 表示区域网格 `右侧` 的边
                 idxNotPeriodicEdge, 表示区域网格 `上下两侧` 的边
        """
        mesh = self.mesh
        idxBdEdge = self.bdIndx

        mid_coor = mesh.entity_barycenter('edge')  # (NE,2)
        bd_mid = mid_coor[idxBdEdge, :]

        isPeriodicEdge0 = np.abs(bd_mid[:, 0] - 0.0) < 1e-8
        isPeriodicEdge1 = np.abs(bd_mid[:, 0] - 1.0) < 1e-8
        notPeriodicEdge = ~(isPeriodicEdge0 + isPeriodicEdge1)
        idxPeriodicEdge0 = idxBdEdge[isPeriodicEdge0]  # (NE_Peri,)
        idxPeriodicEdge1 = idxBdEdge[isPeriodicEdge1]  # (NE_Peri,)

        # |--- 检验 idxPeriodicEdge0 与 idxPeriodicEdge1 是否是一一对应的
        y_0 = mid_coor[idxPeriodicEdge0, 1]
        y_1 = mid_coor[idxPeriodicEdge1, 1]
        if np.allclose(np.sort(y_0), np.sort(y_1)) is False:
            raise ValueError("`idxPeriodicEdge0` and `idxPeriodicEdge1` are not 1-to-1.")
        idxNotPeriodicEdge = idxBdEdge[notPeriodicEdge]
        return idxPeriodicEdge0, idxPeriodicEdge1, idxNotPeriodicEdge

    def set_boundaryDofs(self, dof):
        """
        :param dof: dof, CH using the 'p'-order element
                |___ vdof, NS (velocity) using the 'p+1'-order element
        :return: periodicDof0, 表示区域网格 `左侧` 的边的自由度
                 periodicDof1, 表示区域网格 `右侧` 的边的自由度
                 np.unique(notPeriodicDof), 表示区域网格 `上下两侧` 的边的自由度
        """

        edge2dof = dof.edge_to_dof()
        periodicDof0 = edge2dof[self.idxPeriodicEdge0, :].flatten()
        periodicDof1 = edge2dof[self.idxPeriodicEdge1, :].flatten()
        notPeriodicDof = edge2dof[self.idxNotPeriodicEdge, :].flatten()

        # |--- 下面将左右两边的周期边界上的自由度一一对应起来
        periodicDof0 = np.setdiff1d(periodicDof0, notPeriodicDof)  # 返回的结果会去掉重复的项, 并自动从小到大排序
        periodicDof1 = np.setdiff1d(periodicDof1, notPeriodicDof)  # 返回的结果会去掉重复的项, 并自动从小到大排序
        #     |___ 注意, 上面这种处理方式直接将 `长方形` 区域的 `四个角点` 当做 Dirichlet-dof,
        #         |___ 即, 下面的 periodicDof0, periodicDof1 中是不包含区域的 `四个角点` 的.

        ip = dof.interpolation_points()  # 插值点, 也就是自由度所在的坐标
        ip_coory = ip[periodicDof0, 1]  # 得到 y 的坐标
        argsort = np.argsort(ip_coory)  # ip_coory 从小到大排序时, 返回原来的索引位置
        periodicDof0 = periodicDof0[argsort]  # 重新排列自由度
        ip_coory = ip[periodicDof1, 1]  # 得到 y 的坐标
        argsort = np.argsort(ip_coory)  # ip_coory 从小到大排序时, 返回原来的索引位置
        periodicDof1 = periodicDof1[argsort]  # 重新排列自由度

        if np.allclose(ip[periodicDof0, 1], ip[periodicDof1, 1]) is False:
            raise ValueError("`periodicDof0` and `periodicDof1` are not 1-to-1.")
        return periodicDof0, periodicDof1, np.unique(notPeriodicDof)

    def set_NS_Dirichlet_edge(self, idxDirEdge=None):
        if hasattr(self, 'idxNotPeriodicEdge'):
            idxDirEdge = self.idxNotPeriodicEdge
        else:
            self.idxPeriodicEdge0, self.idxPeriodicEdge1, self.idxNotPeriodicEdge = self.set_periodic_edge()
            idxDirEdge = self.idxNotPeriodicEdge
        return idxDirEdge

    def set_CH_Neumann_edge(self, idxNeuEdge=None):
        if hasattr(self, 'idxNotPeriodicEdge'):
            idxNeuEdge = self.idxNotPeriodicEdge
        else:
            self.idxPeriodicEdge0, self.idxPeriodicEdge1, self.idxNotPeriodicEdge = self.set_periodic_edge()
            idxNeuEdge = self.idxNotPeriodicEdge
        return idxNeuEdge

    def set_periodicAlgebraicSystem(self, dof, rhsVec0, rhsVec1, lhsM=None):
        if dof.p == self.p:
            periodicDof0, periodicDof1 = self.periodicDof0, self.periodicDof1
            #   |___ the phi_h- and p_h-related variables periodic dofs (using p-order polynomial)
        else:
            periodicDof0, periodicDof1 = self.vPeriodicDof0, self.vPeriodicDof1
            #   |___ the velocity-related variables periodic dofs (using (p+1)-order polynomial)

        NPDof = len(periodicDof0)
        NglobalDof = len(rhsVec0)
        IM = identity(NglobalDof, dtype=np.int_)
        if lhsM is None:
            lhsM = IM.copy()

        oneVec = np.ones((NPDof,), dtype=np.int_)
        operatorM = IM + csr_matrix((oneVec, (periodicDof0, periodicDof1)), shape=(NglobalDof, NglobalDof), dtype=np.int_)
        lhsM = operatorM @ lhsM
        rhsVec0 = operatorM @ rhsVec0
        rhsVec1 = operatorM @ rhsVec1
        #    |___ operatorM@lhsM: lhsM 中第 periodicDof1 行加到 periodicDof0 上; rhsVec0, rhsVec1 同理.

        oneVec = np.ones((NglobalDof,), dtype=np.int_)
        oneVec[periodicDof1] = np.int_(0)
        operatorM = spdiags(oneVec, 0, NglobalDof, NglobalDof)
        lhsM = operatorM @ lhsM
        rhsVec0 = operatorM @ rhsVec0
        rhsVec1 = operatorM @ rhsVec1
        #    |___ rowM0@lhsM: lhsM 中第 periodicDof1 行所有元素设置为 0; rhsVec0, rhsVec1 同理.

        oneVec = np.ones((2 * NPDof,), dtype=np.int_)
        oneVec[NPDof:] = np.int_(-1)
        operatorM = csr_matrix((oneVec, (np.hstack([periodicDof1, periodicDof1]), np.hstack([periodicDof0, periodicDof1]))),
                               shape=(NglobalDof, NglobalDof), dtype=np.int_)
        lhsM += operatorM
        #    |___ lhsM 中第 periodicDof1 行: 第 periodicDof0 列为 1, 第 periodicDof1 列为 -1.

        return rhsVec0, rhsVec1, lhsM

