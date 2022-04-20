#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: CoCurrentFlowData.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Feb 16, 2022
# ---

import numpy as np
from scipy.io import loadmat
from fealpy.decorator import cartesian
from fealpy.mesh.TriangleMesh import TriangleMesh
from numpy import pi, sin, cos, exp, tanh
# from CH_NS_Data import CH_NS_Data_truesolution
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh.HalfEdgeMesh2d import HalfEdgeMesh2d
import matplotlib  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
matplotlib.use("TkAgg")  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
import matplotlib.pyplot as plt


class CoCurrentFlowTrueSolution:
    def __init__(self, t0, T, K):
        self.t0 = t0
        self.T = T
        self.K = K  # for the rhs of NS-equation
        self.haveTrueSolution = True
        self.box = self.box_settings()
        self.mesh = self.customized_mesh()
        self.domainVolume = 0

    def setPDEParameters(self, parameters):
        for k, v in parameters.items():
            self.__dict__[k] = v
        return None

    def box_settings(self):
        box = [0, 1, -1, 1]
        self.domainVolume = (box[1] - box[0]) * (box[3] - box[2])
        return box

    def customized_mesh(self):
        print('\n')
        print('# --------------------- in CoCurrentFlowData code --------------------- #')
        nodename = 'CoCurrentMeshNode.npy'
        cellname = 'CoCurrentMeshCell.npy'
        node = np.load('./CoCurrentFlowMesh/' + nodename)
        cell = np.load('./CoCurrentFlowMesh/' + nodename)
        mesh = TriangleMesh(node, cell)

        print('Mesh-cell-name = %s,  ||  Number-of-mesh-cells = %d' % (cellname, mesh.number_of_cells()))
        print('# --------------------------------------------------------------------- #')
        return mesh

    # def customized_mesh(self):
    #     box = self.box
    #     mesh = MF.boxmesh2d(box, nx=10, ny=20, meshtype='tri')
    #     return mesh

    def time_mesh(self, dt):
        n = int(np.ceil((self.T - self.t0) / dt))
        dt = (self.T - self.t0) / n
        return np.linspace(self.t0, self.T, num=n + 1), dt

    @cartesian
    def solution_CH(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        u = 0 * x
        return u

    @cartesian
    def gradient_CH(self, p, t):
        val = np.zeros(p.shape, dtype=np.float64)
        return val  # val.shape == p.shape

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

    @cartesian
    def neumann_CH(self, p, t, n):
        """
        Neumann boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        t: the time
        n: (NE, 2)

        grad*n : (NQ, NE)
        """

        val = 0. * p[..., 0]  # (NQ,NE)
        return val

    @cartesian
    def laplace_neumann_CH(self, p, t, n):
        """
        Laplace Neumann boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        t: the time
        n: (NE, 2)

        grad(laplace u)*n : (NQ, NE)
        """

        val = 0. * p[..., 0]  # (NQ,NE)
        return val

    @cartesian
    def source_CH(self, p, t):

        val = 0. * p[..., 0]
        return val

    # |--- the Navier-Stokes data
    @cartesian
    def velocity_NS(self, p, t):
        val = np.zeros(p.shape, dtype=np.float)
        return val

    @cartesian
    def grad_velocity0_NS(self, p, t):
        val = np.zeros(p.shape, dtype=np.float)
        return val

    @cartesian
    def grad_velocity1_NS(self, p, t):
        val = np.zeros(p.shape, dtype=np.float)
        return val

    @cartesian
    def pressure_NS(self, p, t):
        val = 0. * p[..., 0]
        return val

    @cartesian
    def source_NS(self, p, t, rho_bar_n):
        """

        :param p: (NQ,NC,GD) the physical Gauss points
        :param t:
        :param rho_bar_n: (NQ,NC) the approximation of rho
        :return:
        """

        val = np.ones(p.shape, dtype=np.float)  # (NQ,NC,2)
        # val[..., 0] = val[..., 0] * rho_bar_n * grav[0]
        # val[..., 1] = val[..., 1] * rho_bar_n * grav[1]
        return val

    @cartesian
    def dirichlet_NS(self, p, t):
        return np.zeros(p.shape, dtype=np.float)


