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
        elif meshtype == 'poly':
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
        convection1 = sin(pi * x) * cos(pi * y) * (pi * cos(pi * x) * cos(pi * y)) + (-cos(pi * x) * sin(pi * y)) * (
                    -pi * sin(pi * x) * sin(pi * y))
        convection2 = sin(pi * x) * cos(pi * y) * (pi * sin(pi * x) * sin(pi * y)) + (-cos(pi * x) * sin(pi * y)) * (
                    -pi * cos(pi * x) * cos(pi * y))
        val[..., 0] = nu * (2 * (pi ** 2) * sin(pi * x) * cos(pi * y)) + convection1
        val[..., 1] = nu * (- 2 * (pi ** 2) * sin(pi * y) * cos(pi * x)) - 2 * y / (y ** 2 + 1) ** 2 + convection2
        return val

    def dirichlet(self, p):
        return self.velocity(p)




