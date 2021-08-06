#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: CahnHilliard2DData.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jul 30, 2021
# ---

import numpy as np
from fealpy.decorator import cartesian
from fealpy.mesh.TriangleMesh import TriangleMesh
from numpy import pi, sin, cos, exp


class CahnHilliardData0:
    def __init__(self, t0, T):
        self.t0 = t0
        self.T = T

    def setPDEParameters(self, parameters):
        for k, v in parameters.items():
            self.__dict__[k] = v
        return None

    def space_mesh(self, n=4):
        point = np.array([
            (0, 0),
            (2, 0),
            (4, 0),
            (0, 2),
            (2, 2),
            (4, 2)], dtype=np.float)
        cell = np.array([
            (3, 0, 4),
            (1, 4, 0),
            (2, 5, 1),
            (4, 1, 5)], dtype=np.int)

        mesh = TriangleMesh(point, cell)
        mesh.uniform_refine(n)
        return mesh

    def time_mesh(self, dt):
        n = int(np.ceil((self.T - self.t0) / dt))
        dt = (self.T - self.t0) / n
        return np.linspace(self.t0, self.T, num=n + 1), dt

    @cartesian
    def solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        u = sin(t)*cos(pi*x)*cos(pi*y)
        return u

    @cartesian
    def initdata(self, p):
        return self.solution(p, 0)

    @cartesian
    def gradient(self, p, t):
        """ The gradient of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*sin(t)*sin(pi*x)*cos(pi*y)
        val[..., 1] = -pi*sin(t)*sin(pi*y)*cos(pi*x)
        return val  # val.shape == p.shape

    @cartesian
    def laplace(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = -2*pi**2*sin(t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def neumann(self, p, t, n):
        """
        Neumann boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        t: the time
        n: (NE, 2)

        grad*n : (NQ, NE)
        """

        grad = self.gradient(p, t)  # (NQ, NE, 2)
        val = np.sum(grad * n, axis=-1)  # (NQ, NE)
        return val

    @cartesian
    def laplace_neumann(self, p, t, n):
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
    def source(self, p, t, m, epsilon, eta):
        x = p[..., 0]
        y = p[..., 1]

        val = -m * (-4 * epsilon * pi ** 4 * sin(t) * cos(pi * x) * cos(pi * y) - 2 * epsilon * pi ** 2 * (
                    sin(t) ** 2 * cos(pi * x) ** 2 * cos(pi * y) ** 2 - 1) * sin(t) * cos(pi * x) * cos(
            pi * y) / eta ** 2 + 6 * epsilon * pi ** 2 * sin(t) ** 3 * sin(pi * x) ** 2 * cos(pi * x) * cos(
            pi * y) ** 3 / eta ** 2 + 6 * epsilon * pi ** 2 * sin(t) ** 3 * sin(pi * y) ** 2 * cos(pi * x) ** 3 * cos(
            pi * y) / eta ** 2 - 4 * epsilon * pi ** 2 * sin(t) ** 3 * cos(pi * x) ** 3 * cos(pi * y) ** 3 / eta ** 2) + cos(
            t) * cos(pi * x) * cos(pi * y)
        return val








