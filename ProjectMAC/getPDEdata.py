#!/anaconda3/envs/FEALPy/bin python3.7
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: getPDEdata.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Sep 11, 2019
# ---


import numpy as np


class getPDEBasicData:
    def __init__(self, solutionData, n=4, xmin=0, xmax=1, ymin=0, ymax=1):
        self.solutionData = solutionData

        # TODO: if (xmax - xmin) != (ymax - ymin), but there only has one n
        self.n = n
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        hx = np.abs(xmax - xmin) / n
        hy = np.abs(ymax - ymin) / n
        self.h = min(hx, hy)

    def init_coord(self, entity='all'):
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

        if entity is 'all':
            return ux, uy, vx, vy, px, py
        elif entity is 'u':
            return ux, uy
        elif entity is 'v':
            return vx, vy
        elif entity is 'uv':
            return ux, uy, vx, vy
        elif entity is 'p':
            return px, py
        else:
            raise ValueError("There is no '{}' type!".format(entity))

    def get_u_shape(self):
        return self.init_coord('u')[0].shape

    def get_v_shape(self):
        return self.init_coord('v')[0].shape

    def get_p_shape(self):
        return self.init_coord('p')[0].shape

    def interp_u(self):
        # interpolation of the given solution on mesh points
        x, y = self.init_coord('u')

        # p = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
        p = np.append(x[..., np.newaxis], y[..., np.newaxis], axis=2)
        return self.solutionData.solution(p, 'u')

    def interp_v(self):
        # interpolation of the given solution on mesh points
        x, y = self.init_coord('v')

        # p = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
        p = np.append(x[..., np.newaxis], y[..., np.newaxis], axis=2)
        return self.solutionData.solution(p, 'v')

    def interp_p(self):
        # interpolation of the given solution on mesh points
        x, y = self.init_coord('p')

        # p = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
        p = np.append(x[..., np.newaxis], y[..., np.newaxis], axis=2)
        return self.solutionData.solution(p, 'p')

    def interp_f1(self):
        x, y = self.init_coord('u')
        p = np.append(x[..., np.newaxis], y[..., np.newaxis], axis=2)
        return self.solutionData.source(p, 'f1')

    def interp_f2(self):
        x, y = self.init_coord('v')
        p = np.append(x[..., np.newaxis], y[..., np.newaxis], axis=2)
        return self.solutionData.source(p, 'f2')

    def interp_g(self):
        x, y = self.init_coord('p')
        p = np.append(x[..., np.newaxis], y[..., np.newaxis], axis=2)
        return self.solutionData.source(p, 'g')

    def get_u_dirichlet(self):
        uNrow, uNcol = self.get_u_shape()
        ux = self.init_coord()[0]

        x = ux[0, :]
        y = self.ymax * np.ones((1, uNcol))
        p = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
        uTop = self.solutionData.solution(p, 'u')

        y = self.ymin * np.ones((1, uNcol))
        p = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
        uBot = self.solutionData.solution(p, 'u')

        return np.reshape(uTop, (1, -1)), np.reshape(uBot, (1, -1))

    def get_v_dirichlet(self):
        vNrow, vNcol = self.get_v_shape()
        vy = self.init_coord()[3]

        x = self.xmin * np.ones((1, vNrow))
        y = vy[:, 0]
        p = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
        vLef = self.solutionData.solution(p, 'v')

        x = self.xmax * np.ones((1, vNrow))
        p = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
        vRig = self.solutionData.solution(p, 'v')

        return np.reshape(vLef, (-1, 1)), np.reshape(vRig, (-1, 1))

    def get_init_vals(self):
        # get the initial values
        # TODO: optimize this code
        uNrow, uNcol = self.get_u_shape()
        vNrow, vNcol = self.get_v_shape()
        pNrow, pNcol = self.get_p_shape()

        ux, uy, vx, vy = self.init_coord('uv')

        uzeros = np.zeros((uNrow, uNcol), dtype=float)
        uzeros[:, 0] = 1.
        uzeros[:, -1] = 1.
        vzeros = np.zeros((vNrow, vNcol), dtype=float)
        vzeros[0, :] = 1.
        vzeros[-1, :] = 1.

        u_coord = np.concatenate((ux.reshape((-1, 1)), uy.reshape((-1, 1))), axis=1)
        u_I = self.solutionData.solution(u_coord, 'u')
        u_0 = np.reshape(u_I, (uNrow, uNcol)) * uzeros

        v_coord = np.concatenate((vx.reshape((-1, 1)), vy.reshape((-1, 1))), axis=1)
        v_I = self.solutionData.solution(v_coord, 'v')
        v_0 = np.reshape(v_I, (vNrow, vNcol)) * vzeros

        p_0 = np.zeros((pNrow, pNcol), dtype=float)

        return u_0, v_0, p_0



























