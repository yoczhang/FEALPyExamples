#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FEM_CH_NS_Var_addXi_CapillaryWave.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Jan 28, 2022
# ---

__doc__ = """
The fealpy-FEM program for Variable-Coefficient coupled Cahn-Hilliard-Navier-Stokes equation, 
add the solver for \\xi.
  |___ Capillary wave problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh.HalfEdgeMesh2d import HalfEdgeMesh2d
from CapillaryWaveData import CapillaryWaveSolution


# |--- logging
# make_print_to_file(filename='FEM_CH_NS_Var_addXi_CapillaryWave', setpath="/Users/yczhang/Documents/FEALPy/FEALPyExamples/FEALPyExamples/myExamples/Logs/")

# |--- begin setting
d = 2  # the dimension
p = 1  # the polynomial order
n = 2  # the number of refine mesh

# |--- time and mesh setting
t0 = 0.
T = 3
dt_space = [1.e-5, ]
dt_min = min(dt_space)
time_scheme = 1  # 1 stands for 1st-order time-scheme; 2 is the 2nd-order time-scheme
box = [0, 1, -1, 1]
mesh = MF.boxmesh2d(box, nx=10, ny=20, meshtype='tri')
mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)  # 三角形网格的单边数据结构
mesh.add_plot(plt)
plt.show()

# |--- pde setting
pdePars = {'m': 1e-3, 's': 1, 'alpha': 1, 'epsilon': 1e-3, 'eta': 1e-1, 'dt_min': dt_min, 'timeScheme': '1stOrder'
           }  # value of parameters
varCoeff = {'rho0': 1e-0, 'rho1': 3e-0, 'nu0': 1e-2, 'nu1': 2e-2}
pde = CapillaryWaveSolution(t0, T)  # create pde model
pde.setPDEParameters(pdePars)
pde.setPDEParameters(varCoeff)

# |--- print some basic info
print('\n# ------------ in FEM_CH_NS_Var_addXi_check_t code ------------ #')
print('# ------------ the initial parameters ------------ #')
print('p = ', p)
print('t0 = %.4e,  T = %.4e' % (t0, T))
print('dt_space = ', dt_space)
print('domain box = ', box)
print('rho0 = %.4e,  rho1 = %.4e' % (varCoeff['rho0'], varCoeff['rho1']))
print('nu0 = %.4e,  nu1 = %.4e, ' % (varCoeff['nu0'], varCoeff['nu1']))
print('# #')

