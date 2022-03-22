#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: CocurrentFlowData.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Feb 16, 2022
# ---

import numpy as np
from fealpy.decorator import cartesian
from fealpy.mesh.TriangleMesh import TriangleMesh
from numpy import pi, sin, cos, exp
# from CH_NS_Data import CH_NS_Data_truesolution
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh.HalfEdgeMesh2d import HalfEdgeMesh2d


class CocurrentFlowTrueSolution:
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
        # node = np.load('WaveMeshNode1.npy')
        # cell = np.load('WaveMeshCell1.npy')
        print('\n')
        print('# --------------------- in CapillaryWaveData code --------------------- #')
        nodename = 'WaveMeshNode1.npy'
        cellname = 'WaveMeshCell1.npy'
        # nodename = 'WaveMeshNode_mat2.npy'
        # cellname = 'WaveMeshCell_mat2.npy'
        node = np.load('./CapillaryWaveMesh/' + nodename)  # WaveMeshNode_mat1 是新构造的网格
        cell = np.load('./CapillaryWaveMesh/' + cellname)
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
        val[..., 0] = val[..., 0] * rho_bar_n * grav[0]
        val[..., 1] = val[..., 1] * rho_bar_n * grav[1]
        return val

    @cartesian
    def dirichlet_NS(self, p, t):
        return np.zeros(p.shape, dtype=np.float)


class CocurrentFlowPhycialTest:
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
        box = [0, 2, -1, 1]
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
        # |--- 下面做一下网格结构的转换, 因为目前 HalfEdgeMesh2d 对 p>1 时有 bug
        mesh = TriangleMesh(mesh.node, mesh.entity('cell'))
        return mesh

    def time_mesh(self, dt):
        n = int(np.ceil((self.T - self.t0) / dt))
        dt = (self.T - self.t0) / n
        return np.linspace(self.t0, self.T, num=n + 1), dt

    # # --- the Cahn-Hilliard data
    @cartesian
    def solution_CH(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        u = sin(t) * cos(pi * x) * cos(pi * y)
        return u

    @cartesian
    def initdata_CH(self, p):
        return self.solution_CH(p, 0)

    @cartesian
    def gradient_CH(self, p, t):
        """ The gradient of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi * sin(t) * sin(pi * x) * cos(pi * y)
        val[..., 1] = -pi * sin(t) * sin(pi * y) * cos(pi * x)
        return val  # val.shape == p.shape

    @cartesian
    def laplace_CH(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = -2 * pi ** 2 * sin(t) * cos(pi * x) * cos(pi * y)
        return val

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

        grad = self.gradient_CH(p, t)  # (NQ, NE, 2)
        val = np.sum(grad * n, axis=-1)  # (NQ, NE)
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
        x = p[..., 0]
        y = p[..., 1]

        grad_laplace = np.zeros(p.shape, dtype=np.float64)  # (NQ, NE, 2)
        grad_laplace[..., 0] = 2 * pi ** 3 * sin(t) * sin(pi * x) * cos(pi * y)  # (NQ, NE, 2)
        grad_laplace[..., 1] = 2 * pi ** 3 * sin(t) * sin(pi * y) * cos(pi * x)  # (NQ, NE, 2)
        val = np.sum(grad_laplace * n, axis=-1)  # (NQ, NE)
        return val

    @cartesian
    def source_CH(self, p, t, m, epsilon, eta):
        x = p[..., 0]
        y = p[..., 1]

        val = -m*(-4*epsilon*pi**4*sin(t)*cos(pi*x)*cos(pi*y) - 2*epsilon*pi**2*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*cos(pi*x)*cos(pi*y)/eta**2 + 6*epsilon*pi**2*sin(t)**3*sin(pi*x)**2*cos(pi*x)*cos(pi*y)**3/eta**2 + 6*epsilon*pi**2*sin(t)**3*sin(pi*y)**2*cos(pi*x)**3*cos(pi*y)/eta**2 - 4*epsilon*pi**2*sin(t)**3*cos(pi*x)**3*cos(pi*y)**3/eta**2) - pi*sin(t)**2*sin(pi*x)**2*cos(pi*y)**2 + pi*sin(t)**2*sin(pi*y)**2*cos(pi*x)**2 + cos(t)*cos(pi*x)*cos(pi*y)
        return val

    # # --- the Navier-Stokes data
    @cartesian
    def velocity_NS(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        # pi = np.pi
        # cos = np.cos
        # sin = np.sin
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = sin(pi*x)*cos(pi*y)*sin(t)
        val[..., 1] = -cos(pi*x)*sin(pi*y)*sin(t)
        return val

    @cartesian
    def grad_velocity0_NS(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = pi*sin(t)*cos(pi*x)*cos(pi*y)
        val[..., 1] = -pi*sin(t)*sin(pi*x)*sin(pi*y)
        return val

    @cartesian
    def grad_velocity1_NS(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = pi*sin(t)*sin(pi*x)*sin(pi*y)
        val[..., 1] = -pi*sin(t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def neumann_0_NS(self, p, t, n):
        """
        Neumann boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        t: the time
        n: (NE, 2)

        grad*n : (NQ, NE)
        """
        grad = self.grad_velocity0_NS(p, t)  # (NQ, NE, 2)
        val = np.sum(grad * n, axis=-1)  # (NQ, NE)
        return val

    @cartesian
    def neumann_1_NS(self, p, t, n):
        """
        Neumann boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        t: the time
        n: (NE, 2)

        grad*n : (NQ, NE)
        """
        grad = self.grad_velocity1_NS(p, t)  # (NQ, NE, 2)
        val = np.sum(grad * n, axis=-1)  # (NQ, NE)
        return val

    @cartesian
    def pressure_NS(self, p, t):
        x = p[..., 0]
        y = p[..., 1]

        val = sin(pi*x)*sin(pi*y)*cos(t)
        return val

    @cartesian
    def source_NS(self, p, t, epsilon, eta, m, rho0, rho1, nu0, nu1, stressC):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        # --- the velocity stress term: stressC*0.5*(\nabla u + (\nabla u)^T)
        # --- stressC may take 1.0 or 0.5
        val[..., 0] = m*pi*(-rho0/2 + rho1/2)*(-2*epsilon*pi**3*sin(t)*sin(pi*x)*cos(pi*y) - epsilon*pi*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*sin(pi*x)*cos(pi*y)/eta**2 - 2*epsilon*pi*sin(t)**3*sin(pi*x)*cos(pi*x)**2*cos(pi*y)**3/eta**2)*sin(t)*cos(pi*x)*cos(pi*y) - m*pi*(-rho0/2 + rho1/2)*(-2*epsilon*pi**3*sin(t)*sin(pi*y)*cos(pi*x) - epsilon*pi*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*sin(pi*y)*cos(pi*x)/eta**2 - 2*epsilon*pi*sin(t)**3*sin(pi*y)*cos(pi*x)**3*cos(pi*y)**2/eta**2)*sin(t)*sin(pi*x)*sin(pi*y) + 2*pi**2*stressC*(nu0/2 - nu1/2)*sin(t)**2*sin(pi*x)*cos(pi*x)*cos(pi*y)**2 + 2*pi**2*stressC*(nu0/2 + nu1/2 + (nu0/2 - nu1/2)*sin(t)*cos(pi*x)*cos(pi*y))*sin(t)*sin(pi*x)*cos(pi*y) + pi*sin(pi*y)*cos(t)*cos(pi*x) + (pi*sin(t)**2*sin(pi*x)*sin(pi*y)**2*cos(pi*x) + pi*sin(t)**2*sin(pi*x)*cos(pi*x)*cos(pi*y)**2)*(rho0/2 + rho1/2 + (rho0/2 - rho1/2)*sin(t)*cos(pi*x)*cos(pi*y)) + (rho0/2 + rho1/2 + (rho0/2 - rho1/2)*sin(t)*cos(pi*x)*cos(pi*y))*sin(pi*x)*cos(t)*cos(pi*y) + (-2*epsilon*pi**3*sin(t)*sin(pi*x)*cos(pi*y) - epsilon*pi*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*sin(pi*x)*cos(pi*y)/eta**2 - 2*epsilon*pi*sin(t)**3*sin(pi*x)*cos(pi*x)**2*cos(pi*y)**3/eta**2)*sin(t)*cos(pi*x)*cos(pi*y)
        val[..., 1] = m*pi*(-rho0/2 + rho1/2)*(-2*epsilon*pi**3*sin(t)*sin(pi*x)*cos(pi*y) - epsilon*pi*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*sin(pi*x)*cos(pi*y)/eta**2 - 2*epsilon*pi*sin(t)**3*sin(pi*x)*cos(pi*x)**2*cos(pi*y)**3/eta**2)*sin(t)*sin(pi*x)*sin(pi*y) - m*pi*(-rho0/2 + rho1/2)*(-2*epsilon*pi**3*sin(t)*sin(pi*y)*cos(pi*x) - epsilon*pi*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*sin(pi*y)*cos(pi*x)/eta**2 - 2*epsilon*pi*sin(t)**3*sin(pi*y)*cos(pi*x)**3*cos(pi*y)**2/eta**2)*sin(t)*cos(pi*x)*cos(pi*y) - 2*pi**2*stressC*(nu0/2 - nu1/2)*sin(t)**2*sin(pi*y)*cos(pi*x)**2*cos(pi*y) - 2*pi**2*stressC*(nu0/2 + nu1/2 + (nu0/2 - nu1/2)*sin(t)*cos(pi*x)*cos(pi*y))*sin(t)*sin(pi*y)*cos(pi*x) + pi*sin(pi*x)*cos(t)*cos(pi*y) + (pi*sin(t)**2*sin(pi*x)**2*sin(pi*y)*cos(pi*y) + pi*sin(t)**2*sin(pi*y)*cos(pi*x)**2*cos(pi*y))*(rho0/2 + rho1/2 + (rho0/2 - rho1/2)*sin(t)*cos(pi*x)*cos(pi*y)) - (rho0/2 + rho1/2 + (rho0/2 - rho1/2)*sin(t)*cos(pi*x)*cos(pi*y))*sin(pi*y)*cos(t)*cos(pi*x) + (-2*epsilon*pi**3*sin(t)*sin(pi*y)*cos(pi*x) - epsilon*pi*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*sin(pi*y)*cos(pi*x)/eta**2 - 2*epsilon*pi*sin(t)**3*sin(pi*y)*cos(pi*x)**3*cos(pi*y)**2/eta**2)*sin(t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def dirichlet_NS(self, p, t):
        return self.velocity_NS(p, t)

    @cartesian
    def velocityInitialValue_NS(self, p):
        return self.velocity_NS(p, 0)

    @cartesian
    def pressureInitialValue_NS(self, p):
        return self.pressure_NS(p, 0)

