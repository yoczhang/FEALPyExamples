#!/anaconda3/envs/FEALPy/bin python3.9
# -*- coding: utf-8 -*-
# ---
# @File: totest_newfealpy_cell_basis_at_edges.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Time: 2025/2/7
# ---

from matplotlib import pyplot as plt
from fealpy.utils import timer
from fealpy import logger
logger.setLevel('WARNING')
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem.integrator import LinearInt, OpInt, CellInt, CoefLike, enable_cache
from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.typing import TensorLike, Index, _S
from typing import Optional
from fealpy.mesh import HomogeneousMesh
from fealpy.utils import process_coef_func
from fealpy.functionspace import TensorFunctionSpace
from fealpy.functional import bilinear_integral


bm.set_backend('numpy')


# |--- the following is only the test, copy from "press_work_integrator.py"
class theTestIntegrator(LinearInt, OpInt, CellInt):
    def __init__(self, coef: Optional[CoefLike] = None, q: Optional[int] = None, *,
                 index: Index = _S,
                 batched: bool = False) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The PressWorkIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p + 3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        # |___ bcs:(NQ,GD+1),  ws:(NQ,)

        phi = space.basis(bcs, index=index)  # (1,NQ,ldof)
        gphi = space.grad_basis(bcs, index=index)  # (NC,NQ,ldof,GD)

        # |--- 下面是测试 基函数 在 '边积分点' 处的值
        NE = mesh.number_of_edges()
        qf_e = mesh.quadrature_formula(q, 'edge')
        bcs_e, ws_e = qf_e.get_quadrature_points_and_weights()
        phi_e = space.cell_basis_on_edge(bcs_e, range(NE))  # (NE,NQ_E,ldof)
        gphi_e = space.cell_grad_basis_on_edge(bcs_e, range(NE))  # (NE,NQ_E,ldof,GD)
        return phi, gphi, cm, bcs, ws, index

    def assembly(self, space: _FS) -> TensorLike:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        phi, gphi, cm, bcs, ws, index = self.fetch(space)
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        if isinstance(space, TensorFunctionSpace):
            gphi = gphi
        else:
            gphi = bm.einsum('...ii->...', gphi)
        result = bilinear_integral(gphi, phi, ws, cm, val, batched=self.batched)
        return result


NN = 4
mesh = TriangleMesh.from_box([0, 1, 0, 1], NN, NN)
# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes)
# mesh.find_node(axes, showindex=True)
# mesh.find_cell(axes, showindex=True)
# mesh.find_edge(axes, showindex=True)
#
# # 设置标题和轴标签
# axes.set_title('Triangle Mesh')
# axes.set_xlabel('x')
# axes.set_ylabel('y')
# axes.set_aspect('equal')
# plt.show()

print('---')

p = 1
space = LagrangeFESpace(mesh, p=p)
ti = theTestIntegrator(coef=1, q=p+3)
ti.assembly(space)
print('---')



