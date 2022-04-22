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
    def __init__(self, t0, T, K=-0.01, r0=0.5, r1=1., nu0=None, nu1=None):
        self.t0 = t0
        self.T = T
        self.K = K  # for the rhs of NS-equation
        self.r0 = r0
        self.r1 = r1
        self.nu0 = nu0
        self.nu1 = nu1
        self.haveTrueSolution = True
        self.box = self.box_settings()
        self.mesh = self.customized_mesh()
        self.domainVolume = 0
        self.domain0_cellIdx, self.domain1_cellIdx = self.get_domain_cellIdx()
        self.aver_vel = self.get_aver_vel()

    def setPDEParameters(self, parameters):
        for k, v in parameters.items():
            self.__dict__[k] = v
        return None

    def box_settings(self):
        box = [0, 0.8, -1, 1]
        self.domainVolume = (box[1] - box[0]) * (box[3] - box[2])
        return box

    def customized_mesh(self):
        print('\n')
        print('# --------------------- in CoCurrentFlowData code --------------------- #')
        nodename = 'CoCurrentMeshNode.npy'
        cellname = 'CoCurrentMeshCell.npy'
        node = np.load('./CoCurrentFlowMesh/' + nodename)
        cell = np.load('./CoCurrentFlowMesh/' + cellname)
        mesh = TriangleMesh(node, cell)
        print('Mesh-cell-name = %s,  ||  Number-of-mesh-cells = %d' % (cellname, mesh.number_of_cells()))
        print('# --------------------------------------------------------------------- #')
        return mesh

    def time_mesh(self, dt):
        n = int(np.ceil((self.T - self.t0) / dt))
        dt = (self.T - self.t0) / n
        return np.linspace(self.t0, self.T, num=n + 1), dt

    def get_domain_cellIdx(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        allCellIdx = np.arange(NC)
        r0 = self.r0
        bc = mesh.entity_barycenter('cell')  # (NC,2) barycenter of cells

        domain0_cellIdx, = np.nonzero(abs(bc[..., 1]) < r0)
        domain1_cellIdx = np.setdiff1d(allCellIdx, domain0_cellIdx)
        return domain0_cellIdx, domain1_cellIdx

    @cartesian
    def solution_CH(self, p, eta):
        x = p[..., 0]
        y = p[..., 1]

        u = -tanh((abs(y) - self.r0)/(np.sqrt(2)*eta))
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
    def get_aver_vel(self):
        K = self.K
        r0 = self.r0
        r1 = self.r1
        aver_vel = K * r1 ** 2 / (8 * r0) * (r0 ** 4 / r1 ** 4 * (nu1 / nu0 - 1) + 1)
        return aver_vel


    @cartesian
    def velocity_NS(self, p, t):
        domain0_cellIdx = self.domain0_cellIdx
        domain1_cellIdx = self.domain1_cellIdx
        # nu0 = self.nu0 if hasattr(self, 'nu0') else ValueError("In 'CoCurrentFlowData' no 'nu0' attribute.")
        # nu1 = self.nu1 if hasattr(self, 'nu1') else ValueError("In 'CoCurrentFlowData' no 'nu1' attribute.")
        nu0 = self.nu0
        nu1 = self.nu1
        r0 = self.r0
        r1 = self.r1
        K = self.K
        aver_vel = self.aver_vel

        x = p[..., 0]  # (NQ,NC)
        y = p[..., 1]  # (NQ,NC)

        def domain0_vel(y):
            vel = aver_vel * 2 * (1 - (r0/r1)**2 + nu1/nu0*(r0**2 - y**2)/r1**2) / (r0**4/r1**4 * (nu1/nu0 - 1) + 1)
            return vel

        def domain1_vel(y):
            vel = aver_vel * 2 * (1 - y**2/r1**2) / (r0**4/r1**4 * (nu1/nu0 - 1) + 1)
            return vel

        val = np.zeros(p.shape, dtype=np.float)
        val[:, domain0_cellIdx, 0] = domain0_vel(y[:, domain0_cellIdx])
        val[:, domain1_cellIdx, 0] = domain1_vel(y[:, domain1_cellIdx])
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


