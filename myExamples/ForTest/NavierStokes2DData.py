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


class NavierStokes2DData_0:
    """
    [0, 1]^2
    u(x, y) = (sin(pi*x)*cos(pi*y), -cos(pi*x)*sin(piy))
    p = 1/(y**2 + 1) - pi/4
    """

    def __init__(self, nu):
        self.nu = nu
        self.box = [0, 1, 0, 1]

    def domain(self):
        return self.box

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
        elif meshtype == 'polygon':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh

    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = sin(pi * x) * cos(pi * y)
        val[..., 1] = -cos(pi * x) * sin(pi * y)
        return val

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 1 / (y ** 2 + 1) - pi / 4
        return val

    def source(self, p):
        nu = self.nu
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float)
        convection0 = sin(pi * x) * cos(pi * y) * (pi * cos(pi * x) * cos(pi * y)) + (-cos(pi * x) * sin(pi * y)) * (
                    -pi * sin(pi * x) * sin(pi * y))
        convection1 = sin(pi * x) * cos(pi * y) * (pi * sin(pi * x) * sin(pi * y)) + (-cos(pi * x) * sin(pi * y)) * (
                    -pi * cos(pi * x) * cos(pi * y))
        val[..., 0] = nu * (2 * (pi ** 2) * sin(pi * x) * cos(pi * y)) + convection0
        val[..., 1] = nu * (- 2 * (pi ** 2) * sin(pi * y) * cos(pi * x)) - 2 * y / (y ** 2 + 1) ** 2 + convection1
        return val

    def dirichlet(self, p):
        return self.velocity(p)


class NavierStokes2DData_time:
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
        elif meshtype == 'polygon':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh

    @cartesian
    def velocity(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        # pi = np.pi
        # cos = np.cos
        # sin = np.sin
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = x*x*y*np.exp(-t)
        val[..., 1] = -x*y*y*np.exp(-t)
        return val

    @cartesian
    def grad_velocity0(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 2 * x * y * np.exp(-t)
        val[..., 1] = x**2*np.exp(-t)
        return val

    @cartesian
    def grad_velocity1(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = -y**2*np.exp(-t)
        val[..., 1] = -2*x*y*np.exp(-t)
        return val

    @cartesian
    def neumann_0(self, p, t, n):
        """
        Neumann boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        t: the time
        n: (NE, 2)

        grad*n : (NQ, NE)
        """
        grad = self.grad_velocity0(p, t)  # (NQ, NE, 2)
        val = np.sum(grad * n, axis=-1)  # (NQ, NE)
        return val

    @cartesian
    def neumann_1(self, p, t, n):
        """
        Neumann boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        t: the time
        n: (NE, 2)

        grad*n : (NQ, NE)
        """
        grad = self.grad_velocity1(p, t)  # (NQ, NE, 2)
        val = np.sum(grad * n, axis=-1)  # (NQ, NE)
        return val

    @cartesian
    def pressure(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        # pi = np.pi
        val = (x*y-1/4)*np.exp(-t)
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
        # pi = np.pi
        # sin = np.sin
        # cos = np.cos
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = y*np.exp(-t) - 2*nu*y*np.exp(-t) - x**2*y*np.exp(-t) + self.NS_nolinearTerm_0(p, t)
        val[..., 1] = x*np.exp(-t) + 2*nu*x*np.exp(-t) + x*y**2*np.exp(-t) + self.NS_nolinearTerm_1(p, t)
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


