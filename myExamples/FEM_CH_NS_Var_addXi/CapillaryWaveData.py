#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: CapillaryWaveData.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Jan 28, 2022
# ---

import numpy as np
from fealpy.decorator import cartesian
from fealpy.mesh.TriangleMesh import TriangleMesh
from numpy import pi, sin, cos, exp, tanh
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh.HalfEdgeMesh2d import HalfEdgeMesh2d


class CapillaryWaveSolution:
    def __init__(self, t0, T):
        self.t0 = t0
        self.T = T
        self.haveTrueSolution = True
        self.box = self.box_settings()
        self.mesh = self.customized_mesh()

    def setPDEParameters(self, parameters):
        for k, v in parameters.items():
            self.__dict__[k] = v
        return None

    def box_settings(self):
        box = [0, 1, -1, 1]
        return box

    def customized_mesh(self):
        # box = self.box
        # mesh = MF.boxmesh2d(box, nx=10, ny=20, meshtype='tri')
        # mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)  # 三角形网格的单边数据结构

        cell = np.array([[0, 1, 3], [4, 3, 1], [4, 1, 5], [2, 5, 1],
                         [6, 3, 7], [4, 7, 3], [4, 5, 7], [8, 7, 5],
                         [6, 7, 9], [10, 9, 7], [10, 7, 11], [8, 11, 7],
                         [12, 9, 13], [10, 13, 9], [10, 11, 13], [14, 13, 11]], dtype=np.int)
        node = np.array([[0, -1], [0.5, -1], [1, -1],
                         [0, -0.5], [0.5, -0.5], [1, -0.5],
                         [0, 0], [0.5, 0], [1, 0],
                         [0, 0.5], [0.5, 0.5], [1, 0.5],
                         [0, 1], [0.5, 1], [1, 1]], dtype=np.float)
        mesh = TriangleMesh(node, cell)
        mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)
        mesh.uniform_refine(2)

        cm = 1.
        tt = 0.2
        while cm > 0.01 / 2:
            bc = mesh.cell_barycenter()
            NC = mesh.number_of_cells()
            cellstart = mesh.ds.cellstart
            isMarkedCell = np.zeros(NC + cellstart, dtype=np.bool_)
            isMarkedCell[cellstart:] = abs(bc[:, 1] - 0.) < tt
            mesh.refine_triangle_rg(isMarkedCell)
            cm = np.sqrt(np.min(mesh.entity_measure('cell')))
            if tt > 0.025:
                tt = tt / 2.
        return mesh

    def time_mesh(self, dt):
        n = int(np.ceil((self.T - self.t0) / dt))
        dt = (self.T - self.t0) / n
        return np.linspace(self.t0, self.T, num=n + 1), dt

    @cartesian
    def initial_CH(self, p, eta=5.e-3):
        x = p[..., 0]
        y = p[..., 1]
        Lw = 1.
        Kw = 2 * pi / Lw
        H0 = 0.01

        eta = self.eta if hasattr(self, 'eta') else eta

        u = tanh((y - H0*cos(Kw*x))/(np.sqrt(2)*eta))
        return u
