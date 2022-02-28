#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: poisson_periodic_2d.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Feb 28, 2022
# ---

import numpy as np

from fealpy.decorator import cartesian, barycentric
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.Quadtree import Quadtree
from fealpy.mesh.QuadrangleMesh import QuadrangleMesh
from fealpy.mesh.Tritree import Tritree
from fealpy.mesh.StructureQuadMesh import StructureQuadMesh
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.TriangleMesh import TriangleMeshWithInfinityNode
from fealpy.mesh.PolygonMesh import PolygonMesh
from fealpy.mesh.HalfEdgeMesh2d import HalfEdgeMesh2d


class CosCosData:
    """
        -\\Delta u = f
        u = cos(2*pi*x)*cos(pi*y), 在 x=0 与 x=1 这两边界上, 关于 x 是周期的, 即 u(0,y)==u(1,y).
    """

    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    def init_mesh(self, n=4, meshtype='tri', h=0.1):
        """ generate the initial mesh
        """
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float64)

        if meshtype == 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int_)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        if meshtype == 'quad':
            node = np.array([
                (0, 0),
                (1, 0),
                (1, 1),
                (0, 1),
                (0.5, 0),
                (1, 0.4),
                (0.3, 1),
                (0, 0.6),
                (0.5, 0.45)], dtype=np.float64)
            cell = np.array([
                (0, 4, 8, 7), (4, 1, 5, 8),
                (7, 8, 6, 3), (8, 5, 2, 6)], dtype=np.int_)
            mesh = QuadrangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'halfedge':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh = HalfEdgeMesh2d.from_mesh(mesh)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'squad':
            mesh = StructureQuadMesh([0, 1, 0, 1], h)
            return mesh
        else:
            raise ValueError("".format)

    @cartesian
    def solution(self, p):
        """ The exact solution
        Parameters
        ---------
        p :


        Examples
        -------
        p = np.array([0, 1], dtype=np.float64)
        p = np.array([[0, 1], [0.5, 0.5]], dtype=np.float64)
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(2*pi * x) * np.cos(pi * y)
        return val  # val.shape == x.shape

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 5 * pi * pi * np.cos(2*pi * x) * np.cos(pi * y)
        return val

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -2*pi * np.sin(2*pi * x) * np.cos(pi * y)
        val[..., 1] = -pi * np.cos(2*pi * x) * np.sin(pi * y)
        return val  # val.shape == p.shape

    @cartesian
    def flux(self, p):
        return -self.gradient(p)

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        y = p[..., 1]
        return (y == 1.0) | (y == 0.0)

    @cartesian
    def neumann(self, p, n):
        """
        Neuman  boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        n: (NE, 2)

        grad*n : (NQ, NE, 2)
        """
        grad = self.gradient(p)  # (NQ, NE, 2)
        val = np.sum(grad * n, axis=-1)  # (NQ, NE)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        return x == 1.0

    @cartesian
    def robin(self, p, n):
        grad = self.gradient(p)  # (NQ, NE, 2)
        val = np.sum(grad * n, axis=-1)
        shape = len(val.shape) * (1,)
        kappa = np.array([1.0], dtype=np.float64).reshape(shape)
        val += self.solution(p)
        return val, kappa

    @cartesian
    def is_robin_boundary(self, p):
        x = p[..., 0]
        return x == 0.0
