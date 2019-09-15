#!/anaconda3/envs/FEALPy/bin python3.7
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: StokesMACData.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Sep 11, 2019
# ---


import numpy as np


class StokesMACData:
    def __init__(self, n=4, xmin=0, xmax=1, ymin=0, ymax=1):
        # TODO: if (xmax - xmin) != (ymax - ymin), but there only has one n
        self.n = n
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        hx = np.abs(xmax - xmin) / n
        hy = np.abs(ymax - ymin) / n
        self.h = min(hx, hy)

    def init_coord(self):
        """ generate the initial coordinate of uh, vh, ph
        """
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        h = self.h
        ux = np.arange(xmin, xmax + h / 2, h, dtype=np.float)
        uy = np.arange(ymax - h / 2, ymin, -h, dtype=np.float)
        ux, uy = np.meshgrid(ux, uy)

        vx = np.arange(xmin + h / 2, xmax, h, dtype=np.float)
        vy = np.arange(ymax, ymin - h / 2, -h, dtype=np.float)
        vx, vy = np.meshgrid(vx, vy)

        px = np.arange(xmin + h / 2, xmax, h, dtype=np.float)
        py = np.arange(ymax - h / 2, ymin, -h, dtype=np.float)
        px, py = np.meshgrid(px, py)

        return ux, uy, vx, vy, px, py

    def get_u_shape(self):
        return self.init_coord()[0].shape

    def get_v_shape(self):
        return self.init_coord()[2].shape

    def get_p_shape(self):
        return self.init_coord()[4].shape

    def solution(self, p):
        """ The exact solution
        """
        x = p[..., 0]
        y = p[..., 1]

        uval = x ** 2 * (x - 1) ** 2 * y * (y - 1) * (2 * y - 1)
        vval = -x * (x - 1) * (2. * x - 1) * y ** 2. * (y - 1) ** 2
        pval = (2 * x - 1) * (2 * y - 1)

        return uval, vval, pval

    def source(self, p):
        nu = 1
        x = p[..., 0]
        y = p[..., 1]

        f1val = 2 * (2 * y - 1) * (
                - 3 * nu * x ** 4 + 6 * nu * x ** 3 - 6 * nu * x ** 2 * y ** 2 + 6 * nu * x ** 2 * y - 3 * nu * x ** 2
                + 6 * nu * x * y ** 2 - 6 * nu * x * y - nu * y ** 2 + nu * y + 1)
        f2val = 2 * (2 * x - 1) * (
                6 * nu * x ** 2 * y ** 2 - 6 * nu * x ** 2 * y + nu * x ** 2 - 6 * nu * x * y ** 2 + 6 * nu * x * y
                - nu * x + 3 * nu * y ** 4 - 6 * nu * y ** 3 + 3 * nu * y ** 2 + 1)

        return f1val, f2val

    def interpolation_vals(self, p):
        return self.solution(p)

    def get_u_dirichlet(self):
        uNrow, uNcol = self.get_u_shape()
        ux = self.init_coord()[0]

        x = ux[0, :]
        y = self.ymax * np.ones((1, uNcol))
        p = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
        uTop = self.solution(p)[0]

        y = self.ymin * np.ones((1, uNcol))
        p = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
        uBot = self.solution(p)[0]

        return np.reshape(uTop, (1, -1)), np.reshape(uBot, (1, -1))

    def get_v_dirichlet(self):
        vNrow, vNcol = self.get_v_shape()
        vy = self.init_coord()[3]

        x = self.xmin * np.ones((1, vNrow))
        y = vy[:, 0]
        p = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
        vLef = self.solution(p)[1]

        x = self.xmax * np.ones((1, vNrow))
        p = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
        vRig = self.solution(p)[1]

        return np.reshape(vLef, (-1, 1)), np.reshape(vRig, (-1, 1))

    def get_init_vals(self):
        # TODO: optimize this code
        uNrow, uNcol = self.get_u_shape()
        vNrow, vNcol = self.get_v_shape()
        pNrow, pNcol = self.get_p_shape()

        ux, uy, vx, vy, px, py = self.init_coord()

        uzeros = np.zeros((uNrow, uNcol), dtype=float)
        uzeros[:, 0] = 1.
        uzeros[:, -1] = 1.
        vzeros = np.zeros((vNrow, vNcol), dtype=float)
        vzeros[0, :] = 1.
        vzeros[-1, :] = 1.

        u_p = np.concatenate((ux.reshape((-1, 1)), uy.reshape((-1, 1))), axis=1)
        u_i = self.solution(u_p)[0]
        u_0 = np.reshape(u_i, (uNrow, uNcol)) * uzeros

        v_p = np.concatenate((vx.reshape((-1, 1)), vy.reshape((-1, 1))), axis=1)
        v_i = self.solution(v_p)[1]
        v_0 = np.reshape(v_i, (vNrow, vNcol)) * vzeros

        p_0 = np.zeros((pNrow, pNcol), dtype=float)

        return u_0, v_0, p_0



























