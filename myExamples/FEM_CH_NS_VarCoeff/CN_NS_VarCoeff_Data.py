#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: CN_NS_VarCoeff_Data.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Dec 19, 2021
# ---

import numpy as np
from fealpy.decorator import cartesian
from fealpy.mesh.TriangleMesh import TriangleMesh
from numpy import pi, sin, cos, exp
from CH_NS_Data import CH_NS_Data_truesolution


class CH_NS_VarCoeff_truesolution(CH_NS_Data_truesolution):
    def __init__(self, t0, T, nu0, nu1, rho0, rho1):
        super(CH_NS_VarCoeff_truesolution, self).__init__(t0, T)


