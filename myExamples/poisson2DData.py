#!/anaconda3/envs/FEALPy/bin python3.7
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: poisson2DData.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Jan 27, 2020
# ---

import numpy as np

from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.Quadtree import Quadtree
from fealpy.mesh.QuadrangleMesh import QuadrangleMesh
from fealpy.mesh.Tritree import Tritree
from fealpy.mesh.StructureQuadMesh import StructureQuadMesh
from fealpy.mesh.TriangleMesh import TriangleMesh, TriangleMeshWithInfinityNode
from fealpy.mesh.PolygonMesh import PolygonMesh


class CosCosData:
    """
    -\Delta u = f
    u = cos(pi*x)*cos(pi*y)
    """

    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    def init_mesh(self, n=4, meshtype='quad', h=0.1):
        """ generate the initial mesh
        """
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype is 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        if meshtype is 'quad':
            node = np.array([
                (0, 0),
                (1, 0),
                (1, 1),
                (0, 1),
                (0.5, 0),
                (1, 0.4),
                (0.3, 1),
                (0, 0.6),
                (0.5, 0.45)], dtype=np.float)
            cell = np.array([
                (0, 4, 8, 7), (4, 1, 5, 8),
                (7, 8, 6, 3), (8, 5, 2, 6)], dtype=np.int)
            mesh = QuadrangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'stri':
            mesh = StructureQuadMesh([0, 1, 0, 1], h)
            return mesh
        else:
            raise ValueError("".format)

    def solution(self, p):
        """ The exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi * x) * np.cos(pi * y)
        return val

    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2 * pi * pi * np.cos(pi * x) * np.cos(pi * y)
        return val

    def gradient(self, p):
        """ The gradient of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = -pi * np.sin(pi * x) * np.cos(pi * y)
        val[..., 1] = -pi * np.cos(pi * x) * np.sin(pi * y)
        return val

    def dirichlet(self, p):
        return self.solution(p)

    def neuman(self, p, n):
        """ Neuman  boundary condition
        p: (NQ, NE, 2)
        n: (NE, 2)
        """
        grad = self.gradient(p)  # (NQ, NE, 2)
        val = np.sum(grad * n, axis=-1)
        return val

    def robin(self, p):
        pass


class ffData:
    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype is 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'tritree':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
            mesh = Tritree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("I don't know the meshtype %s".format(meshtype))

    def solution(self, p):
        return np.zeros(p.shape[0:-1])

    def gradient(self, p):
        """ The gradient of the exact solution
        """
        val = np.zeros(p.shape, dtype=p.dtype)
        return val

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = np.ones(x.shape, dtype=np.float)
        I = np.floor(4 * x) + np.floor(4 * y)
        isMinus = (I % 2 == 0)
        val[isMinus] = - 1
        return val

    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        val = np.zeros(p.shape[0:-1])
        return val


class KelloggData:
    def __init__(self):
        self.a = 161.4476387975881
        self.b = 1

    def init_mesh(self, n=4, meshtype='tri'):
        node = np.array([
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1)], dtype=np.float)
        if meshtype is 'tri':
            cell = np.array([
                (1, 4, 0),
                (3, 0, 4),
                (4, 1, 5),
                (2, 5, 1),
                (4, 7, 3),
                (6, 3, 7),
                (7, 4, 8),
                (5, 8, 4)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
        elif meshtype is 'quadtree':
            cell = np.array([
                (0, 1, 4, 3),
                (1, 2, 5, 4),
                (3, 4, 7, 6),
                (4, 5, 8, 7)], dtype=np.int)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
        else:
            raise ValueError("".format)
        return mesh

    def diffusion_coefficient(self, p):
        idx = (p[..., 0] * p[..., 1] > 0)
        k = np.ones(p.shape[:-1], dtype=np.float)
        k[idx] = self.a
        return k

    def subdomain(self, p):
        """
        get the subdomain flag of the subdomain including point p.
        """
        is_subdomain = [p[..., 0] * p[..., 1] > 0, p[..., 0] * p[..., 1] < 0]
        return is_subdomain

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        gamma = 0.1
        sigma = -14.9225565104455152
        rho = pi / 4
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        theta = (theta >= 0) * theta + (theta < 0) * (theta + 2 * pi)

        mu = ((theta >= 0) & (theta < pi / 2)) * cos((pi / 2 - sigma) * gamma) * cos((theta - pi / 2 + rho) * gamma) \
             + ((theta >= pi / 2) & (theta < pi)) * cos(rho * gamma) * cos((theta - pi + sigma) * gamma) \
             + ((theta >= pi) & (theta < 1.5 * pi)) * cos(sigma * gamma) * cos((theta - pi - rho) * gamma) \
             + ((theta >= 1.5 * pi) & (theta < 2 * pi)) * cos((pi / 2 - rho) * gamma) * cos(
            (theta - 1.5 * pi - sigma) * gamma)

        u = r ** gamma * mu
        return u

    def gradient(self, p):
        """The gradient of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape, dtype=p.dtype)
        pi = np.pi
        cos = np.cos
        sin = np.sin
        gamma = 0.1
        sigma = -14.9225565104455152
        rho = pi / 4
        theta = np.arctan2(y, x)
        theta = (theta >= 0) * theta + (theta < 0) * (theta + 2 * pi)
        t = 1 + (y / x) ** 2
        r = np.sqrt(x ** 2 + y ** 2)
        rg = r ** gamma

        ux1 = ((x >= 0.0) & (y >= 0.0)) * (
                gamma * rg * cos((pi / 2 - sigma) * gamma) * (x * cos((theta - pi / 2 + rho) * gamma) / (r * r)
                                                              + y * sin((theta - pi / 2 + rho) * gamma) / (x * x * t))
        )

        uy1 = ((x >= 0.0) & (y >= 0.0)) * (gamma * rg * cos((pi / 2 - sigma) * gamma) * (
                y * cos((theta - pi / 2 + rho) * gamma) / (r * r) - sin((theta - pi / 2 + rho) * gamma) / (x * t)))

        ux2 = ((x <= 0.0) & (y >= 0.0)) * (gamma * rg * cos(rho * gamma) * (
                x * cos((theta - pi + sigma) * gamma) / (r * r) + y * sin((theta - pi + sigma) * gamma) / (
                x * x * t)))

        uy2 = ((x <= 0.0) & (y >= 0.0)) * (gamma * rg * cos(rho * gamma) * (
                y * cos((theta - pi + sigma) * gamma) / (r * r) - sin((theta - pi + sigma) * gamma) / (x * t)))

        ux3 = ((x <= 0.0) & (y <= 0.0)) * (gamma * rg * cos(sigma * gamma) * (
                x * cos((theta - pi - rho) * gamma) / (r * r) + y * sin((theta - pi - rho) * gamma) / (x * x * t)))

        uy3 = ((x <= 0.0) & (y <= 0.0)) * (gamma * rg * cos(sigma * gamma) * (
                y * cos((theta - pi - rho) * gamma) / (r * r) - sin((theta - pi - rho) * gamma) / (x * t)))

        ux4 = ((x >= 0.0) & (y <= 0.0)) * (gamma * rg * cos((pi / 2 - rho) * gamma) * (
                x * cos((theta - 3 * pi / 2 - sigma) * gamma) / (r * r) + y * sin(
            (theta - 3 * pi / 2 - sigma) * gamma) / (x * x * t)))

        uy4 = ((x >= 0.0) & (y <= 0.0)) * (gamma * rg * cos((pi / 2 - rho) * gamma) * (
                y * cos((theta - 3 * pi / 2 - sigma) * gamma) / (r * r) - sin(
            (theta - 3 * pi / 2 - sigma) * gamma) / (x * t)))

        val[..., 0] = ux1 + ux2 + ux3 + ux4
        val[..., 1] = uy1 + uy2 + uy3 + uy4
        return val

    def source(self, p):
        """the right hand side of Possion equation
        INPUT:
            p:array object, N*2
        """
        rhs = np.zeros(p.shape[0:-1])
        return rhs

    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)
