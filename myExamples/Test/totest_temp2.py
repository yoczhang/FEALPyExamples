#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: totest_temp2.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Jun 24, 2020
# ---

from fealpy.functionspace.ScaledMonomialSpace2d import ScaledMonomialSpace2d
import numpy as np

from fealpy.mesh import PolygonMesh
from fealpy.mesh import StructureQuadMesh, QuadrangleMesh
from fealpy.mesh import TriangleMesh, TriangleMeshWithInfinityNode


# --- mesh
n = 0
node = np.array([
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1)], dtype=np.float)
cell = np.array([
    (1, 2, 0),
    (3, 0, 2)], dtype=np.int)
mesh = TriangleMesh(node, cell)
mesh.uniform_refine(n)
nmesh = TriangleMeshWithInfinityNode(mesh)
pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
pmesh = PolygonMesh(pnode, pcell, pcellLocation)
mesh = pmesh

p = 1
smspace = ScaledMonomialSpace2d(mesh, p)


def f(x, index=None):
    gphi = smspace.grad_basis(x, index=index)
    gpphi = smspace.grad_basis(x, index=index, p=p+1)
    return np.einsum('...mn, ...kn->...km', gphi, gpphi)


S = smspace.integralalg.integral(f, celltype=True, barycenter=False)

# ------------------------------------------------- #
print("End of this test file")




