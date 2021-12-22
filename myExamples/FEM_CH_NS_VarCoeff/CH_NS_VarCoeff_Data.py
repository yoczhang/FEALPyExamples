#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: CH_NS_VarCoeff_Data.py
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
    def __init__(self, t0, T):
        super(CH_NS_VarCoeff_truesolution, self).__init__(t0, T)
        self.box = self.box_settings()

    def box_settings(self):
        box = [0, 2, -1, 1]
        return box

    # # --- the Cahn-Hilliard data
    @cartesian
    def source_CH(self, p, t, m, epsilon, eta):
        x = p[..., 0]
        y = p[..., 1]
        val = -m*(-4*epsilon*pi**4*sin(t)*cos(pi*x)*cos(pi*y) - 2*epsilon*pi**2*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*cos(pi*x)*cos(pi*y)/eta**2 + 6*epsilon*pi**2*sin(t)**3*sin(pi*x)**2*cos(pi*x)*cos(pi*y)**3/eta**2 + 6*epsilon*pi**2*sin(t)**3*sin(pi*y)**2*cos(pi*x)**3*cos(pi*y)/eta**2 - 4*epsilon*pi**2*sin(t)**3*cos(pi*x)**3*cos(pi*y)**3/eta**2) - pi*sin(t)**2*sin(pi*x)**2*cos(pi*y)**2 + pi*sin(t)**2*sin(pi*y)**2*cos(pi*x)**2 + cos(t)*cos(pi*x)*cos(pi*y)
        return val

    # # --- the Navier-Stokes data
    @cartesian
    def velocity_NS(self, p, t):
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = sin(pi*x)*cos(pi*y)*sin(t)
        val[..., 1] = -cos(pi*x)*sin(pi*y)*sin(t)
        return val

    @cartesian
    def grad_velocity0_NS(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = pi*sin(t)*cos(pi*x)*cos(pi*y)
        val[..., 1] = -pi*sin(t)*sin(pi*x)*sin(pi*y)
        return val

    @cartesian
    def grad_velocity1_NS(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = pi*sin(t)*sin(pi*x)*sin(pi*y)
        val[..., 1] = -pi*sin(t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def pressure_NS(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        # pi = np.pi
        val = sin(pi*x)*sin(pi*y)*cos(t)
        return val

    @cartesian
    def source_NS(self, p, t, epsilon, eta, m, rho0, rho1, nu0, nu1):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = m*pi*(-rho0/2 + rho1/2)*(-2*epsilon*pi**3*sin(t)*sin(pi*x)*cos(pi*y) - epsilon*pi*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*sin(pi*x)*cos(pi*y)/eta**2 - 2*epsilon*pi*sin(t)**3*sin(pi*x)*cos(pi*x)**2*cos(pi*y)**3/eta**2)*sin(t)*cos(pi*x)*cos(pi*y) - m*pi*(-rho0/2 + rho1/2)*(-2*epsilon*pi**3*sin(t)*sin(pi*y)*cos(pi*x) - epsilon*pi*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*sin(pi*y)*cos(pi*x)/eta**2 - 2*epsilon*pi*sin(t)**3*sin(pi*y)*cos(pi*x)**3*cos(pi*y)**2/eta**2)*sin(t)*sin(pi*x)*sin(pi*y) + pi**2*(nu0/2 - nu1/2)*sin(t)**2*sin(pi*x)*cos(pi*x)*cos(pi*y)**2 + pi**2*(nu0/2 + nu1/2 + (nu0/2 - nu1/2)*sin(t)*cos(pi*x)*cos(pi*y))*sin(t)*sin(pi*x)*cos(pi*y) + pi*sin(pi*y)*cos(t)*cos(pi*x) + (pi*sin(t)**2*sin(pi*x)*sin(pi*y)**2*cos(pi*x) + pi*sin(t)**2*sin(pi*x)*cos(pi*x)*cos(pi*y)**2)*(rho0/2 + rho1/2 + (rho0/2 - rho1/2)*sin(t)*cos(pi*x)*cos(pi*y)) + (rho0/2 + rho1/2 + (rho0/2 - rho1/2)*sin(t)*cos(pi*x)*cos(pi*y))*sin(pi*x)*cos(t)*cos(pi*y) + (-2*epsilon*pi**3*sin(t)*sin(pi*x)*cos(pi*y) - epsilon*pi*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*sin(pi*x)*cos(pi*y)/eta**2 - 2*epsilon*pi*sin(t)**3*sin(pi*x)*cos(pi*x)**2*cos(pi*y)**3/eta**2)*sin(t)*cos(pi*x)*cos(pi*y)
        val[..., 1] = m*pi*(-rho0/2 + rho1/2)*(-2*epsilon*pi**3*sin(t)*sin(pi*x)*cos(pi*y) - epsilon*pi*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*sin(pi*x)*cos(pi*y)/eta**2 - 2*epsilon*pi*sin(t)**3*sin(pi*x)*cos(pi*x)**2*cos(pi*y)**3/eta**2)*sin(t)*sin(pi*x)*sin(pi*y) - m*pi*(-rho0/2 + rho1/2)*(-2*epsilon*pi**3*sin(t)*sin(pi*y)*cos(pi*x) - epsilon*pi*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*sin(pi*y)*cos(pi*x)/eta**2 - 2*epsilon*pi*sin(t)**3*sin(pi*y)*cos(pi*x)**3*cos(pi*y)**2/eta**2)*sin(t)*cos(pi*x)*cos(pi*y) - pi**2*(nu0/2 - nu1/2)*sin(t)**2*sin(pi*y)*cos(pi*x)**2*cos(pi*y) - pi**2*(nu0/2 + nu1/2 + (nu0/2 - nu1/2)*sin(t)*cos(pi*x)*cos(pi*y))*sin(t)*sin(pi*y)*cos(pi*x) + pi*sin(pi*x)*cos(t)*cos(pi*y) + (pi*sin(t)**2*sin(pi*x)**2*sin(pi*y)*cos(pi*y) + pi*sin(t)**2*sin(pi*y)*cos(pi*x)**2*cos(pi*y))*(rho0/2 + rho1/2 + (rho0/2 - rho1/2)*sin(t)*cos(pi*x)*cos(pi*y)) - (rho0/2 + rho1/2 + (rho0/2 - rho1/2)*sin(t)*cos(pi*x)*cos(pi*y))*sin(pi*y)*cos(t)*cos(pi*x) + (-2*epsilon*pi**3*sin(t)*sin(pi*y)*cos(pi*x) - epsilon*pi*(sin(t)**2*cos(pi*x)**2*cos(pi*y)**2 - 1)*sin(t)*sin(pi*y)*cos(pi*x)/eta**2 - 2*epsilon*pi*sin(t)**3*sin(pi*y)*cos(pi*x)**3*cos(pi*y)**2/eta**2)*sin(t)*cos(pi*x)*cos(pi*y)
        return val
