#!/anaconda3/envs/FEALPy/bin python3.7
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site:
# @File: Smoothers_2D.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Sep 10, 2019
# ---


import numpy as np
# import copy


class DGS_smoother:
    def __init__(self, pde):
        self.pde = pde
        self.uh, self.vh, self.ph = pde.get_init_vals()
        self.uhTop, self.uhBot = pde.get_u_dirichlet()
        self.vhLef, self.vhRig = pde.get_v_dirichlet()
        self.f1h = pde.interp_f1()
        self.f2h = pde.interp_f2()
        self.gh = pde.interp_g()
        self.h = pde.h

    def smoother(self, nstep=1):
        uh, vh, ph = self.uh, self.vh, self.ph
        uhTop, uhBot = self.uhTop, self.uhBot
        vhLef, vhRig = self.vhLef, self.vhRig
        f1h = self.f1h
        f2h = self.f2h
        gh = self.gh
        h = self.h

        (uNrow, uNcol) = uh.shape
        (vNrow, vNcol) = vh.shape
        pNrow = uNrow
        pNcol = vNcol

        # --- ---
        # Step 1: Gauss-Seidel relaxation of velocity
        for ite in range(0, nstep):
            # --- uh ---
            # uh, treat boundary, only treat the top and bottom boundary
            u_col = np.arange(1, uNcol - 1)
            uh[0, u_col] = (2 * uhTop[0, u_col] + uh[1, u_col] + uh[0, u_col - 1] + uh[0, u_col + 1] - h * (ph[0, u_col] - ph[0, u_col - 1]) + h ** 2 * f1h[0, u_col]) / 5
            uh[-1, u_col] = (2 * uhBot[0, u_col] + uh[-2, u_col] + uh[-1, u_col - 1] + uh[-1, u_col + 1] - h * (ph[-1, u_col] - ph[-1, u_col - 1]) + h ** 2 * f1h[-1, u_col]) / 5

            # uh, red-black iteration
            # case1, (red points): mod(u_row + u_col, 2) == 0
            u_row = np.arange(1, uNrow - 1, 2)
            u_row = u_row[:, None]
            u_col = np.arange(1, uNcol - 1, 2)
            uh[u_row, u_col] = (uh[u_row - 1, u_col] + uh[u_row + 1, u_col] + uh[u_row, u_col - 1] + uh[u_row, u_col + 1] - h * (ph[u_row, u_col] - ph[u_row, u_col - 1]) + h ** 2 * f1h[u_row, u_col]) / 4
            u_row = np.arange(2, uNrow - 1, 2)
            u_row = u_row[:, None]
            u_col = np.arange(2, uNcol - 1, 2)
            uh[u_row, u_col] = (uh[u_row - 1, u_col] + uh[u_row + 1, u_col] + uh[u_row, u_col - 1] + uh[u_row, u_col + 1] - h * (ph[u_row, u_col] - ph[u_row, u_col - 1]) + h ** 2 * f1h[u_row, u_col]) / 4
            # case2, (black points): mod(u_row + u_col, 2) == 1
            u_row = np.arange(1, uNrow - 1, 2)
            u_row = u_row[:, None]
            u_col = np.arange(2, uNcol - 1, 2)
            uh[u_row, u_col] = (uh[u_row - 1, u_col] + uh[u_row + 1, u_col] + uh[u_row, u_col - 1] + uh[u_row, u_col + 1] - h * (ph[u_row, u_col] - ph[u_row, u_col - 1]) + h ** 2 * f1h[u_row, u_col]) / 4
            u_row = np.arange(2, uNrow - 1, 2)
            u_row = u_row[:, None]
            u_col = np.arange(1, uNcol - 1, 2)
            uh[u_row, u_col] = (uh[u_row - 1, u_col] + uh[u_row + 1, u_col] + uh[u_row, u_col - 1] + uh[u_row, u_col + 1] - h * (ph[u_row, u_col] - ph[u_row, u_col - 1]) + h ** 2 * f1h[u_row, u_col]) / 4

            # --- vh ---
            # vh, treat boundary, only treat left and right boundary
            v_row = np.arange(1, vNrow - 1)
            v_row = v_row[:, None]
            vh[v_row, 0] = (2 * vhLef[v_row, 0] + vh[v_row - 1, 0] + vh[v_row + 1, 0] + vh[v_row, 1] - h * (ph[v_row - 1, 0] - ph[v_row, 0]) + h ** 2 * f2h[v_row, 0]) / 5
            vh[v_row, -1] = (2 * vhRig[v_row, 0] + vh[v_row - 1, -1] + vh[v_row + 1, -1] + vh[v_row, -2] - h * (ph[v_row - 1, -1] - ph[v_row, -1]) + h ** 2 * f2h[v_row, -1]) / 5

            # vh, red-black iteration
            # case1, (red points): mod(v_row + v_col, 2) == 0
            v_row = np.arange(1, vNrow - 1, 2)
            v_row = v_row[:, None]
            v_col = np.arange(1, vNcol - 1, 2)
            vh[v_row, v_col] = (vh[v_row - 1, v_col] + vh[v_row + 1, v_col] + vh[v_row, v_col - 1] + vh[v_row, v_col + 1] - h * (ph[v_row - 1, v_col] - ph[v_row, v_col]) + h ** 2 * f2h[v_row, v_col]) / 4
            v_row = np.arange(2, vNrow - 1, 2)
            v_row = v_row[:, None]
            v_col = np.arange(2, vNcol - 1, 2)
            vh[v_row, v_col] = (vh[v_row - 1, v_col] + vh[v_row + 1, v_col] + vh[v_row, v_col - 1] + vh[v_row, v_col + 1] - h * (ph[v_row - 1, v_col] - ph[v_row, v_col]) + h ** 2 * f2h[v_row, v_col]) / 4
            # case2, (black points): mod(v_row + v_col, 2) == 1
            v_row = np.arange(1, vNrow - 1, 2)
            v_row = v_row[:, None]
            v_col = np.arange(2, vNcol - 1, 2)
            vh[v_row, v_col] = (vh[v_row - 1, v_col] + vh[v_row + 1, v_col] + vh[v_row, v_col - 1] + vh[v_row, v_col + 1] - h * (ph[v_row - 1, v_col] - ph[v_row, v_col]) + h ** 2 * f2h[v_row, v_col]) / 4
            v_row = np.arange(2, vNrow - 1, 2)
            v_row = v_row[:, None]
            v_col = np.arange(1, vNcol - 1, 2)
            vh[v_row, v_col] = (vh[v_row - 1, v_col] + vh[v_row + 1, v_col] + vh[v_row, v_col - 1] + vh[v_row, v_col + 1] - h * (ph[v_row - 1, v_col] - ph[v_row, v_col]) + h ** 2 * f2h[v_row, v_col]) / 4

        # --- ---
        # Step2: Distributive relaxation of velocity and pressure
        # define the rp = gh + div([u,v]^T)
        p_row = np.arange(0, pNrow)
        p_row = p_row[:, None]
        p_col = np.arange(0, pNcol)
        rp = np.zeros((pNrow, pNcol))
        rp[p_row, p_col] = gh[p_row, p_col] + (uh[p_row, p_col + 1] - uh[p_row, p_col]) / h + (vh[p_row, p_col] - vh[p_row + 1, p_col]) / h

        # Gauss-Seidel smoother, solve Ap * dp = rp
        dp = np.zeros((pNrow, pNcol))
        maxIt = 1000
        tol = 1e-3
        for ite in range(0, maxIt):
            # dp_temp = copy.copy(dp)
            dp_temp = dp.copy()

            # --- ---
            # interior points
            p_row = np.arange(1, pNrow - 1, 2)
            p_row = p_row[:, None]
            p_col = np.arange(1, pNcol - 1, 2)
            dp[p_row, p_col] = (h ** 2 * rp[p_row, p_col] + dp[p_row - 1, p_col] + dp[p_row + 1, p_col] + dp[p_row, p_col - 1] + dp[p_row, p_col + 1]) / 4
            p_row = np.arange(2, pNrow - 1, 2)
            p_row = p_row[:, None]
            p_col = np.arange(2, pNcol - 1, 2)
            dp[p_row, p_col] = (h ** 2 * rp[p_row, p_col] + dp[p_row - 1, p_col] + dp[p_row + 1, p_col] + dp[p_row, p_col - 1] + dp[p_row, p_col + 1]) / 4
            p_row = np.arange(1, pNrow - 1, 2)
            p_row = p_row[:, None]
            p_col = np.arange(2, pNcol - 1, 2)
            dp[p_row, p_col] = (h ** 2 * rp[p_row, p_col] + dp[p_row - 1, p_col] + dp[p_row + 1, p_col] + dp[p_row, p_col - 1] + dp[p_row, p_col + 1]) / 4
            p_row = np.arange(2, pNrow - 1, 2)
            p_row = p_row[:, None]
            p_col = np.arange(1, pNcol - 1, 2)
            dp[p_row, p_col] = (h ** 2 * rp[p_row, p_col] + dp[p_row - 1, p_col] + dp[p_row + 1, p_col] + dp[p_row, p_col - 1] + dp[p_row, p_col + 1]) / 4

            # --- ---
            # boundary points but no corner points
            # left-, right- boundary
            p_row = np.arange(1, pNrow - 1)
            p_row = p_row[:, None]
            dp[p_row, 0] = (h ** 2 * rp[p_row, 0] + dp[p_row + 1, 0] + dp[p_row - 1, 0] + dp[p_row, 1]) / 3
            dp[p_row, -1] = (h ** 2 * rp[p_row, -1] + dp[p_row + 1, -1] + dp[p_row - 1, -1] + dp[p_row, -2]) / 3
            # top-, bottom- boundary
            p_col = np.arange(1, pNcol - 1)
            dp[0, p_col] = (h ** 2 * rp[0, p_col] + dp[0, p_col + 1] + dp[0, p_col - 1] + dp[1, p_col]) / 3
            dp[-1, p_col] = (h ** 2 * rp[-1, p_col] + dp[-1, p_col + 1] + dp[-1, p_col - 1] + dp[-2, p_col]) / 3

            # --- ---
            # corner points
            dp[0, 0] = (h ** 2 * rp[0, 0] + dp[1, 0] + dp[0, 1]) / 2  # left top
            dp[-1, 0] = (h ** 2 * rp[-1, 0] + dp[-2, 0] + dp[-1, 1]) / 2  # left bottom
            dp[0, -1] = (h ** 2 * rp[0, -1] + dp[1, -1] + dp[0, -2]) / 2  # right top
            dp[-1, -1] = (h ** 2 * rp[-1, -1] + dp[-2, -1] + dp[-1, -2]) / 2  # right bottom

            # --- ---
            # compute error
            e_max = np.max(np.abs(dp - dp_temp))
            if e_max < tol:
                break

        # --- ---
        # update uh, vh
        # uh^{k+1} = uh^{k} + \partial_x dp
        # vh^{k+1} = vh^{k} + \partial_y dp
        u_row = np.arange(0, uNrow)
        u_row = u_row[:, None]
        u_col = np.arange(1, uNcol-1)
        uh[u_row, u_col] = uh[u_row, u_col] + (dp[u_row, u_col] - dp[u_row, u_col - 1]) / h
        v_row = np.arange(1, vNrow-1)
        v_row = v_row[:, None]
        v_col = np.arange(0, vNcol)
        vh[v_row, v_col] = vh[v_row, v_col] + (dp[v_row - 1, v_col] - dp[v_row, v_col]) / h

        # --- ---
        # update ph: ph^{k+1} = ph^{k} -  Ap * dp
        # interior nodes
        p_row = np.arange(1, pNrow - 1)
        p_row = p_row[:, None]
        p_col = np.arange(1, pNcol - 1)
        ph[p_row, p_col] = ph[p_row, p_col] - (4 * dp[p_row, p_col] - dp[p_row - 1, p_col] - dp[p_row + 1, p_col] - dp[p_row, p_col - 1] - dp[p_row, p_col + 1]) / (h ** 2)

        # boundary nodes but not corner nodes
        ph[p_row, 0] = ph[p_row, 0] - (3 * dp[p_row, 0] - dp[p_row + 1, 0] - dp[p_row - 1, 0] - dp[p_row, 1]) / (h ** 2)
        ph[p_row, -1] = ph[p_row, -1] - (3 * dp[p_row, -1] - dp[p_row + 1, -1] - dp[p_row - 1, -1] - dp[p_row, -2]) / (h ** 2)
        ph[0, p_col] = ph[0, p_col] - (3 * dp[0, p_col] - dp[0, p_col + 1] - dp[0, p_col - 1] - dp[1, p_col]) / (h ** 2)
        ph[-1, p_col] = ph[-1, p_col] - (3 * dp[-1, p_col] - dp[-1, p_col + 1] - dp[-1, p_col - 1] - dp[-2, p_col]) / (h ** 2)

        # corner nodes
        ph[0, 0] = ph[0, 0] - (2 * dp[0, 0] - dp[1, 0] - dp[0, 1]) / (h ** 2)  # left top
        ph[-1, 0] = ph[-1, 0] - (2 * dp[-1, 0] - dp[-2, 0] - dp[-1, 1]) / (h ** 2)  # left top
        ph[0, -1] = ph[0, -1] - (2 * dp[0, -1] - dp[1, -1] - dp[0, -2]) / (h ** 2)  # right top
        ph[-1, -1] = ph[-1, -1] - (2 * dp[-1, -1] - dp[-2, -1] - dp[-1, -2]) / (h ** 2)  # right bottom

        # --- ---
        # return

        return uh, vh, ph

    def u_l2_err(self):
        e = self.uh - self.pde.interp_u()
        return np.sqrt(np.mean(e ** 2))

    def v_l2_err(self):
        e = self.vh - self.pde.interp_v()
        return np.sqrt(np.mean(e ** 2))

    def p_l2_err(self):
        e = self.ph - self.pde.interp_p()
        return np.sqrt(np.mean(e ** 2))

    def get_residual(self, entity='all'):
        uh, vh, ph = self.uh, self.vh, self.ph
        uhTop, uhBot = self.uhTop, self.uhBot
        vhLef, vhRig = self.vhLef, self.vhRig
        f1h = self.f1h
        f2h = self.f2h
        gh = self.gh
        h = self.h

        uNrow, uNcol = uh.shape
        vNrow, vNcol = vh.shape

        r_u = np.zeros((uNrow, uNcol), dtype=float)
        r_v = np.zeros((vNrow, vNcol), dtype=float)
        r_div = np.zeros((uNrow, vNcol), dtype=float)

        # uh residual
        row = np.arange(1, uNrow - 1)
        row = row[:, None]
        col = np.arange(1, uNcol - 1)
        r_u[row, col] = f1h[row, col] - (ph[row, col] - ph[row, col - 1]) / h + (uh[row - 1, col] + uh[row + 1, col] + uh[row, col - 1] + uh[row, col + 1] - 4 * uh[row, col]) / h ** 2
        r_u[0, col] = f1h[0, col] - (ph[0, col] - ph[0, col - 1]) / h + (2 * uhTop[0, col] + uh[1, col] + uh[0, col - 1] + uh[0, col + 1] - 5 * uh[0, col]) / h ** 2
        r_u[-1, col] = f1h[-1, col] - (ph[-1, col] - ph[-1, col - 1]) / h + (2 * uhBot[0, col] + uh[-2, col] + uh[-1, col - 1] + uh[-1, col + 1] - 5 * uh[-1, col]) / h ** 2

        # vh residual
        row = np.arange(1, vNrow - 1)
        row = row[:, None]
        col = np.arange(1, vNcol - 1)
        r_v[row, col] = f2h[row, col] - (ph[row - 1, col] - ph[row, col]) / h + (vh[row - 1, col] + vh[row + 1, col] + vh[row, col - 1] + vh[row, col + 1] - 4 * vh[row, col]) / h ** 2
        r_v[row, 0] = f2h[row, 0] - (ph[row - 1, 0] - ph[row, 0]) / h + (2 * vhLef[row, 0] + vh[row - 1, 0] + vh[row + 1, 0] + vh[row, 1] - 5 * vh[row, 0]) / h ** 2
        r_v[row, -1] = f2h[row, -1] - (ph[row - 1, -1] - ph[row, -1]) / h + (2 * vhRig[row, 0] + vh[row - 1, -1] + vh[row + 1, -1] + vh[row, -2] - 5 * vh[row, -1]) / h ** 2

        # div residual
        row = np.arange(0, uNrow)
        row = row[:, None]
        col = np.arange(0, vNcol)
        r_div[row, col] = gh[row, col] + (uh[row, col + 1] - uh[row, col]) / h + (vh[row, col] - vh[row + 1, col]) / h

        if entity is 'all':
            return r_u, r_v, r_div
        elif entity is 'u':
            return r_u
        elif entity is 'v':
            return r_v
        elif entity in ('p', 'div'):
            return r_div
        else:
            raise ValueError("There is no '{}' type!".format(entity))


