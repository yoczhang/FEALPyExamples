#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_HHOStokesSpace.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Sep 15, 2020
# ---

import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.mesh_tools import find_node, find_entity
from fealpy.functionspace.ScaledMonomialSpace2d import ScaledMonomialSpace2d
from fealpy.mesh import MeshFactory
from fealpy.mesh import HalfEdgeMesh2d
from HHOStokesSpace2d import HHOStokesSpace2d
from Stokes2DData import Stokes2DData_0


# --- begin --- #
n = 2
p = 1
nu = 1.0
pde = Stokes2DData_0(nu)  # create pde model

# --- mesh setting --- #
box = [0, 1, 0, 1]  # [0, 1]^2 domain
mf = MeshFactory()
meshtype = 'quad'
mesh = mf.boxmesh2d(box, nx=n, ny=n, meshtype=meshtype)
mesh = HalfEdgeMesh2d.from_mesh(mesh)
mesh.init_level_info()
mesh.uniform_refine(n-2)  # refine the mesh at beginning

# --- plot the mesh --- #
# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes, cellcolor='w')
# find_entity(axes, mesh, entity='cell', showindex=True, color='b', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='edge', showindex=True, color='r', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='node', showindex=True, color='y', markersize=10, fontsize=8)
# plt.show()

cell2edge, cellLocation = mesh.ds.cell_to_edge()
# #
# #
# --- HHO space setting --- #
smspace = ScaledMonomialSpace2d(mesh, p)
integralalg = smspace.integralalg
sspace = HHOStokesSpace2d(mesh, p)
vSpace = sspace.vSpace
vdof = sspace.dof


# --- test begin --- #
lastuh = vSpace.function()
lastuh[:] = np.random.rand(len(lastuh))
# lastuh = np.concatenate([lastuh, 2.0 + lastuh])

# --- test fh --- #
fh = vSpace.project(pde.source, dim=2)
def osc_f(x, index=np.s_[:]):
    fhval = vSpace.value(fh, x, index=index)  # the evalue has the same shape of x.
    fval = pde.source(x)
    return (fval - fhval)**2
err = vSpace.integralalg.integral(osc_f, celltype=True)

# --- to test the residual_estimate0 --- #
nu = 1.0
uh = vSpace.function(dim=3)
uh = np.random.standard_normal(uh.shape)

eta = sspace.residual_estimate0(nu, uh, pde.source)

# --- --- #
cb = mesh.cell_barycenter()
guhval = vSpace.grad_value(uh, cb)


def grad_uh(point, index=np.s_[:]):
    return vSpace.grad_value(uh, point=point, index=index)
guh = vSpace.integralalg.integral(grad_uh, celltype=True)

# #
# #
# ------------------------------------------------- #
print("End of this test file")





