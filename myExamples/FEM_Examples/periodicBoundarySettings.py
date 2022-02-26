#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: periodicBoundarySettings.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Feb 24, 2022
# ---

import numpy as np
from scipy.sparse import csr_matrix, spdiags, identity, eye, bmat
# from fealpy.quadrature import FEMeshIntegralAlg
from scipy.sparse.linalg import spsolve


class periodicBoundarySettings:
    def __init__(self, mesh, dof, set_periodic_edge_func):
        self.mesh = mesh
        self.dof = dof
        self.idxPeriodicEdge0, self.idxPeriodicEdge1, self.idxNotPeriodicEdge = self.set_periodic_edge(set_periodic_edge_func)

    # |--- periodic boundary and dofs settings
    def set_periodic_edge(self, set_periodic_edge_func):
        """
            :return: idxPeriodicEdge0, 表示区域网格 `左侧` 的边
                     idxPeriodicEdge1, 表示区域网格 `右侧` 的边
                     idxNotPeriodicEdge, 表示区域网格 `上下两侧` 的边
            """
        mesh = self.mesh
        idxPeriodicEdge0, idxPeriodicEdge1, idxNotPeriodicEdge = set_periodic_edge_func(mesh)
        return idxPeriodicEdge0, idxPeriodicEdge1, idxNotPeriodicEdge

    def set_boundaryDofs(self):
        """
            :param dof: dof, CH using the 'p'-order element
                    |___ vdof, NS (velocity) using the 'p+1'-order element
            :return: periodicDof0, 表示区域网格 `左侧` 的边的自由度
                     periodicDof1, 表示区域网格 `右侧` 的边的自由度
                     np.unique(notPeriodicDof), 表示区域网格 `上下两侧` 的边的自由度
        """

        dof = self.dof
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

    def set_periodicAlgebraicSystem(self, periodicDof0, periodicDof1, rhsVec0, rhsVec1=None, lhsM=None):

        NPDof = len(periodicDof0)
        NglobalDof = len(rhsVec0)
        IM = identity(NglobalDof, dtype=np.int_)
        rhsVec1_flag = True
        lhsM_flag = True
        if rhsVec1 is None:
            rhsVec1_flag = False
            rhsVec1 = rhsVec0.copy()
        if lhsM is None:
            lhsM_flag = False
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

        if (rhsVec1_flag is not True) and (lhsM_flag is True):
            return rhsVec0, lhsM
        elif (rhsVec1_flag is True) and (lhsM_flag is not True):
            return rhsVec0, rhsVec1
        else:
            return rhsVec0, rhsVec1, lhsM


