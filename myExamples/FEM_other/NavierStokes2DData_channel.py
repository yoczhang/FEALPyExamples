#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: NavierStokes2DData.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Jun 02, 2020
# ---


import numpy as np
from fealpy.decorator import cartesian
from fealpy.mesh import PolygonMesh
from fealpy.mesh import StructureQuadMesh, QuadrangleMesh
from fealpy.mesh import TriangleMesh, TriangleMeshWithInfinityNode


class NavierStokes2DData_channel:
    """

    """

    def __init__(self, nu):
        self.nu = nu
        self.box = [0, 1, 0, 1]

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype == 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            nx = 2
            ny = 2
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh

    @cartesian
    def velocity(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        return val

    @cartesian
    def grad_velocity0(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        return val

    @cartesian
    def grad_velocity1(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        return val

    @cartesian
    def pressure(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        # pi = np.pi
        val = 0 * x
        return val

    @cartesian
    def NS_nolinearTerm_0(self, p, t):
        # x = p[..., 0]
        # y = p[..., 1]
        u0 = self.velocity(p, t)[..., 0]
        u1 = self.velocity(p, t)[..., 1]
        u0x = self.grad_velocity0(p, t)[..., 0]
        u0y = self.grad_velocity0(p, t)[..., 1]
        u1x = self.grad_velocity1(p, t)[..., 0]
        u1y = self.grad_velocity1(p, t)[..., 1]
        return u0*u0x + u1*u0y

    @cartesian
    def NS_nolinearTerm_1(self, p, t):
        # x = p[..., 0]
        # y = p[..., 1]
        u0 = self.velocity(p, t)[..., 0]
        u1 = self.velocity(p, t)[..., 1]
        u0x = self.grad_velocity0(p, t)[..., 0]
        u0y = self.grad_velocity0(p, t)[..., 1]
        u1x = self.grad_velocity1(p, t)[..., 0]
        u1y = self.grad_velocity1(p, t)[..., 1]
        return u0*u1x + u1*u1y

    @cartesian
    def source(self, p, t):
        nu = self.nu
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        return val

    @cartesian
    def dirichlet(self, p, t):
        return self.velocity(p, t)

    @cartesian
    def velocityInitialValue(self, p):
        return self.velocity(p, 0)

    @cartesian
    def pressureInitialValue(self, p):
        return self.pressure(p, 0)

    @cartesian
    def scalar_zero_fun(self, p):
        x = p[..., 0]
        val = 0*x
        return val


