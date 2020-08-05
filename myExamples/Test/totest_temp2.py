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
import matplotlib.pyplot as plt
from fealpy.mesh.mesh_tools import find_entity

from fealpy.mesh import PolygonMesh
from fealpy.mesh import StructureQuadMesh, QuadrangleMesh
from fealpy.mesh import TriangleMesh, TriangleMeshWithInfinityNode
from fealpy.decorator import cartesian, barycentric


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

# --- to pmesh
# nmesh = TriangleMeshWithInfinityNode(mesh)
# pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
# pmesh = PolygonMesh(pnode, pcell, pcellLocation)
# mesh = pmesh

# ---- plot mesh ----
# fig1 = plt.figure()
# axes = fig1.gca()
# mesh.add_plot(axes, cellcolor='w')
# find_entity(axes, mesh, entity='cell', showindex=True, color='b', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='edge', showindex=True, color='r', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='node', showindex=True, color='y', markersize=10, fontsize=8)
# plt.show()
# plt.close()
# -------------------

p = 1
smspace = ScaledMonomialSpace2d(mesh, p)


# def f(x, index=None):
#     gphi = smspace.grad_basis(x, index=index)
#     gpphi = smspace.grad_basis(x, index=index, p=p+1)
#     return np.einsum('...mn, ...kn->...km', gphi, gpphi)
#
#
# S = smspace.integralalg.integral(f, celltype=True, barycenter=False)

# --- another test
uh = smspace.function()

@cartesian
def f1(x, index=np.s_[:]):
    return uh.value(x, index)


# S1 = smspace.integralalg.integral(f1, celltype=True)
S2 = smspace.integralalg.integral(f1, celltype=True, barycenter=False)
# S3 = smspace.integralalg.cell_integral(f1)



# ------------------------------------------------- #
print("End of this test file")




