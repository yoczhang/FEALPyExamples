#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: PosterioriStokesHHORate2d.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Sep 14, 2020
# ---

__doc__ = """
The fealpy program for posteriori Stokes problem. 
"""

from Stokes2DData import Stokes2DData_0, Stokes2DData_1, Stokes2DData_2, Stokes2DData_3, StokesLshapeData
from Stokes2DData import StokesAroundCylinderData, StokesCubeFlowData
import numpy as np
from ShowCls import ShowCls
from StokesHHOModel2d import StokesHHOModel2d
from fealpy.mesh import MeshFactory
from fealpy.mesh import HalfEdgeMesh2d
from mesh_IO import mesh_IO
import matplotlib.pyplot as plt
import datetime

# from fealpy.tools.show import showmultirate, show_error_table
# from fealpy.mesh.mesh_tools import find_entity


# --- begin setting --- #
d = 2  # the dimension
p = 2  # the polynomial order
n = 4  # the number of refine mesh
maxit = 15  # the max iteration of the mesh

nu = 1.0e-0
# pde = StokesAroundCylinderData(nu)  # create pde model
pde = StokesCubeFlowData(nu)  # create pde model

# --- error settings --- #
errorType = ['ETA']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
Ndof = np.zeros(maxit, dtype=np.int)  # the array to store the number of dofs

# --- mesh setting --- #
mIO = mesh_IO()

# --- mesh1 --- #
# box = [0, 1, 0, 1]  # [0, 1]^2 domain
# mf = MeshFactory()
# meshtype = 'quad'
# mesh = mf.boxmesh2d(box, nx=n, ny=n, meshtype=meshtype)

# --- mesh2: around cylinder --- #
# matfile = '../Meshfiles/aroundcylinder.mat'
# matfile = '../Meshfiles/aroundcylinder_poly_final.mat'

# --- mesh3: cube flow --- #
matfile = '../Meshfiles/cube_flow_mesh_quad.mat'
# matfile = '../Meshfiles/cube_flow_mesh_poly.mat'

# --- get mesh info --- #
mesh = mIO.loadMatlabMesh(filename=matfile)

# --- to halfedgemesh --- #
mesh = HalfEdgeMesh2d.from_mesh(mesh)
mesh.init_level_info()
# mesh.uniform_refine(n - 1)  # refine the mesh at beginning
# mesh.uniform_refine(2)

now_time = datetime.datetime.now()
outPath = '../Outputs/PostStokes' + now_time.strftime('%y-%m-%d(%H\'%M\'%S)')

# --- plot the mesh --- #
# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes, cellcolor='w')
# find_entity(axes, mesh, entity='cell', showindex=True, color='b', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='edge', showindex=True, color='r', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='node', showindex=True, color='y', markersize=10, fontsize=8)
# plt.show()

# --- another way plot the mesh --- #
sc = ShowCls(p, mesh, errorType=errorType, Ndof=Ndof, errorMatrix=errorMatrix, out=None)
# sc.showMeshInfo()
# sc.showMesh(markCell=False, markEdge=False, markNode=False)
# sc.showMesh()

# mesh.uniform_refine(1)
# print("------------------")
# sc.showMeshInfo()

# --- start for-loop --- #
stokes = None
sol = None
tol = 1.5e-1
print('nu = %e' % nu)
i = 0
ETA = 1.0
uh = None
for i in range(maxit):  # range(maxit), [maxit-1]
# while ETA > tol:
    print('\n# --------------------- i = %d ------------------------- #' % i)
    stokes = StokesHHOModel2d(pde, mesh, p)
    sol = stokes.solve()
    uh = sol['uh']

    # --- expand Ndof and errorMatrix --- #
    Ndof[i] = stokes.space.number_of_velocity_dofs()  # get the number of dofs

    # --- adaptive settings --- #
    eta = stokes.space.residual_estimate0(nu, uh, pde.source, pde.dirichlet, stokes.setDirichletEdges())  # (NC,)
    ETA = np.sqrt(np.sum(eta ** 2))
    errorMatrix[0, i] = ETA
    print('Posteriori Info:')
    print('  |___ ETA = %e: ' % ETA)
    print('  |___ before refine: number of cells: %d, pressure dofs: %d ' % (
        mesh.number_of_cells(), stokes.space.number_of_pressure_dofs()))

    # --- to debug --- #
    eta_l = eta[:32]
    lidx_max2min = np.argsort(eta_l)[::-1]  # 降序排序后返回索引值
    eta_r = eta[32:]
    ridx_max2min = np.argsort(eta_r)[::-1]  # 降序排序后返回索引值
    # ---------------- #

    # --- adaptive refine the mesh --- #
    if (i < maxit - 1) & (ETA > tol):
        # --- one way to refine
        # aopts = mesh.adaptive_options(method='max', theta=0.3, maxcoarsen=0.1, HB=True)
        # mesh.adaptive(eta, aopts)

        # --- another way to refine
        isMarkedCell = stokes.space.post_estimator_markcell(eta, theta=0.5)
        markedIdx = np.nonzero(isMarkedCell)
        mesh.refine_poly(isMarkedCell)
        print('  |___ after refine: number of cells: ', mesh.number_of_cells())

        # --- uniform refine the mesh --- #
        # mesh.uniform_refine()

        i += 1

        # --- save the specfied mesh --- #
        if i in {1, 4, 7, 10, 13, 15}:
            saveMeshName = outPath + '_p=' + str(p) + '_mesh_' + str(i) + '_.mat'
            mIO.save2MatlabMesh(mesh, filename=saveMeshName)
    else:
        pass

    # sc.showMesh(markCell=False, markEdge=False, markNode=False)
    fig1 = plt.figure()
    axes = fig1.gca()
    mesh.add_plot(axes)
    outPath_1 = outPath + str(i) + '-mesh.png'
    plt.savefig(outPath_1)
    plt.close()
    print('  |___ end the circle')

# --- plot solution --- #
# stokes.showSolution(sc)

# --- save mesh --- #
saveMeshName = outPath + '_p=' + str(p) + '_mesh_final.mat'
mIO.save2MatlabMesh(mesh, filename=saveMeshName)

# --- save uh --- #
ndofs = stokes.space.number_of_pressure_dofs()
saveUhName = outPath + '_p=' + str(p) + '_uh_final.mat'
mIO.save2MatlabUh(uh[:ndofs, :], filename=saveUhName)

# --- get the convergence rate --- #
print('\n')
print('# --------------------- table ------------------------- #')
# sc.show_error_table(out=outPath, Cidx=range(i+1), DofName='Velocity-Dof', tableType='dof-type')
# sc.showmultirate(i-7, Ridx=[1, 4], Cidx=range(i+1))
# plt.show()

# ---
print('end of the program')
