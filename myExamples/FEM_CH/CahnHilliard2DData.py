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


class CahnHilliardData0:
    def __init__(self, t0, t1):
        self.t0 = t0
        self.t1 = t1
        self.epsilon = 0.01

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

    def time_mesh(self, tau):
        n = int(np.ceil((self.t1 - self.t0) / tau))
        tau = (self.t1 - self.t0) / n
        return np.linspace(self.t0, self.t1, num=n + 1), tau

    def initdata(self, p):
        x = p[..., 0]
        y = p[..., 1]
        u0 = np.sin(np.pi * x) ** 2 * np.sin(np.pi * y) ** 2
        return u0

    def solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        u = np.exp(-2 * t) * np.sin(np.pi * x) ** 2 * np.sin(np.pi * y) ** 2
        return u

    def gradient(self, p, t):
        """ The gradient of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi * np.sin(pi * x) * np.cos(pi * y)
        val[..., 1] = -pi * np.cos(pi * x) * np.sin(pi * y)
        return val  # val.shape == p.shape

    def laplace(self, p, t):
        pass

    def neumann(self, p):
        """
        Neumann boundary condition
        """
        return 0

    def source(self, p, t):
        epsilon = self.epsilon
        x = p[..., 0]
        y = p[..., 1]
        rhs = (-2) * np.exp(-2 * t) * np.sin(np.pi * x) ** 2 * np.sin(np.pi * y) ** 2 \
              + epsilon * 2 * np.pi ** 2 * np.exp(-2 * t) * (
                      4 * np.pi ** 2 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) - 4 * np.pi ** 2 * np.cos(
                  2 * np.pi * y) * np.sin(
                  np.pi * x) ** 2 - 4 * np.pi ** 2 * np.cos(2 * np.pi * x) * np.sin(np.pi * y) ** 2)
        return rhs
