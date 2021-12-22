#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: CH_NS_Data.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Aug 19, 2021
# ---

import numpy as np
from fealpy.decorator import cartesian
from fealpy.mesh.TriangleMesh import TriangleMesh
from numpy import pi, sin, cos, exp


class CH_NS_Data_truesolution:
    def __init__(self, t0, T):
        self.t0 = t0
        self.T = T
        self.haveTrueSolution = True

    def setPDEParameters(self, parameters):
        for k, v in parameters.items():
            self.__dict__[k] = v
        return None

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

        val = -m*(-4*epsilon*pi**4*sin(t)*cos(pi*x)*cos(pi*y) - 2*epsilon*pi**2*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*cos(pi*x)*cos(pi*y)/eta**2 + 6*epsilon*pi**2*sin(t)**3*sin(pi*x)**2*cos(pi*x)*cos(pi*y)**3/eta**2 + 6*epsilon*pi**2*sin(t)**3*sin(pi*y)**2*cos(pi*x)**3*cos(pi*y)/eta**2 - 4*epsilon*pi**2*sin(t)**3*cos(pi*x)**3*cos(pi*y)**3/eta**2) - pi*x**2*y*exp(-t)*sin(t)*sin(pi*x)*cos(pi*y) + pi*x*y**2*exp(-t)*sin(t)*sin(pi*y)*cos(pi*x) + cos(t)*cos(pi*x)*cos(pi*y)
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
        val[..., 0] = x * x * y * np.exp(-t)
        val[..., 1] = -x * y * y * np.exp(-t)
        return val

    @cartesian
    def grad_velocity0_NS(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 2 * x * y * np.exp(-t)
        val[..., 1] = x ** 2 * np.exp(-t)
        return val

    @cartesian
    def grad_velocity1_NS(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = -y ** 2 * np.exp(-t)
        val[..., 1] = -2 * x * y * np.exp(-t)
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
        # pi = np.pi
        val = (x * y - 1 / 4) * np.exp(-t)
        return val

    # @cartesian
    # def NS_nolinearTerm_0_NS(self, p, t):
    #     # x = p[..., 0]
    #     # y = p[..., 1]
    #     u0 = self.velocity_NS(p, t)[..., 0]
    #     u1 = self.velocity_NS(p, t)[..., 1]
    #     u0x = self.grad_velocity0_NS(p, t)[..., 0]
    #     u0y = self.grad_velocity0_NS(p, t)[..., 1]
    #     u1x = self.grad_velocity1_NS(p, t)[..., 0]
    #     u1y = self.grad_velocity1_NS(p, t)[..., 1]
    #     return u0 * u0x + u1 * u0y

    # @cartesian
    # def NS_nolinearTerm_1_NS(self, p, t):
    #     # x = p[..., 0]
    #     # y = p[..., 1]
    #     u0 = self.velocity_NS(p, t)[..., 0]
    #     u1 = self.velocity_NS(p, t)[..., 1]
    #     u0x = self.grad_velocity0_NS(p, t)[..., 0]
    #     u0y = self.grad_velocity0_NS(p, t)[..., 1]
    #     u1x = self.grad_velocity1_NS(p, t)[..., 0]
    #     u1y = self.grad_velocity1_NS(p, t)[..., 1]
    #     return u0 * u1x + u1 * u1y

    @cartesian
    def source_NS(self, p, t, nu, epsilon, eta):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = -2*nu*y*exp(-t) + x**3*y**2*exp(-2*t) - x**2*y*exp(-t) + y*exp(-t) + (-2*epsilon*pi**3*sin(t)*sin(pi*x)*cos(pi*y) - epsilon*pi*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*sin(pi*x)*cos(pi*y)/eta**2 - 2*epsilon*pi*sin(t)**3*sin(pi*x)*cos(pi*x)**2*cos(pi*y)**3/eta**2)*sin(t)*cos(pi*x)*cos(pi*y)
        val[..., 1] = 2*nu*x*exp(-t) + x**2*y**3*exp(-2*t) + x*y**2*exp(-t) + x*exp(-t) + (-2*epsilon*pi**3*sin(t)*sin(pi*y)*cos(pi*x) - epsilon*pi*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*sin(pi*y)*cos(pi*x)/eta**2 - 2*epsilon*pi*sin(t)**3*sin(pi*y)*cos(pi*x)**3*cos(pi*y)**2/eta**2)*sin(t)*cos(pi*x)*cos(pi*y)
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

    @cartesian
    def scalar_zero_fun_NS(self, p):
        x = p[..., 0]
        val = 0 * x
        return val


class CH_NS_Data_truesolution1:
    def __init__(self, t0, T):
        self.t0 = t0
        self.T = T
        self.haveTrueSolution = True
        self.box = self.box_settings()

    def setPDEParameters(self, parameters):
        for k, v in parameters.items():
            self.__dict__[k] = v
        return None

    def box_settings(self):
        box = [0, 2, -1, 1]
        return box

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
        # pi = np.pi
        val = sin(pi*x)*sin(pi*y)*cos(t)
        return val

    @cartesian
    def source_NS(self, p, t, nu, epsilon, eta):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 2*nu*pi**2*sin(t)*sin(pi*x)*cos(pi*y) + pi*sin(t)**2*sin(pi*x)*sin(pi*y)**2*cos(pi*x) + pi*sin(t)**2*sin(pi*x)*cos(pi*x)*cos(pi*y)**2 + pi*sin(pi*y)*cos(t)*cos(pi*x) + (-2*epsilon*pi**3*sin(t)*sin(pi*x)*cos(pi*y) - epsilon*pi*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*sin(pi*x)*cos(pi*y)/eta**2 - 2*epsilon*pi*sin(t)**3*sin(pi*x)*cos(pi*x)**2*cos(pi*y)**3/eta**2)*sin(t)*cos(pi*x)*cos(pi*y) + sin(pi*x)*cos(t)*cos(pi*y)
        val[..., 1] = -2*nu*pi**2*sin(t)*sin(pi*y)*cos(pi*x) + pi*sin(t)**2*sin(pi*x)**2*sin(pi*y)*cos(pi*y) + pi*sin(t)**2*sin(pi*y)*cos(pi*x)**2*cos(pi*y) + pi*sin(pi*x)*cos(t)*cos(pi*y) + (-2*epsilon*pi**3*sin(t)*sin(pi*y)*cos(pi*x) - epsilon*pi*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*sin(pi*y)*cos(pi*x)/eta**2 - 2*epsilon*pi*sin(t)**3*sin(pi*y)*cos(pi*x)**3*cos(pi*y)**2/eta**2)*sin(t)*cos(pi*x)*cos(pi*y) - sin(pi*y)*cos(t)*cos(pi*x)
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

    @cartesian
    def scalar_zero_fun_NS(self, p):
        x = p[..., 0]
        val = 0 * x
        return val



