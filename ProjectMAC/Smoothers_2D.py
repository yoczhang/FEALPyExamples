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


class DGS_smoother:
    def __init__(self, pde):
        self.pde = pde

    def smoother(self, nstep=1):
        uh, vh, ph = self.pde.get_init_vals()
        uhTop, uhBot = self.pde.get_u_dirichlet()
        vhLef, vhRig = self.pde.get_v_dirichlet()
        f1h = self.pde.interp_f1()
        f2h = self.pde.interp_f2()
        gh = self.pde.interp_g()
        h = self.pde.h

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
            uh[0, u_col] = (2 * uhTop[0, u_col] + uh[1, u_col] + uh[0, u_col - 1] + uh[0, u_col + 1]
                            - h * (ph[0, u_col] - ph[0, u_col - 1]) + h ** 2 * f1h[0, u_col]) / 5
            uh[-1, u_col] = (2 * uhBot[0, u_col] + uh[-2, u_col] + uh[-1, u_col - 1] + uh[-1, u_col + 1]
                             - h * (ph[-1, u_col] - ph[-1, u_col - 1]) + h ** 2 * f1h[-1, u_col]) / 5

            # uh, red-black iteration
            # case1, (red points): mod(u_row + u_col, 2) == 0
            u_row = np.arange(1, uNrow - 1, 2)
            u_col = np.arange(1, uNcol - 1, 2)
            uh[u_row, u_col] = (uh[u_row - 1, u_col] + uh[u_row + 1, u_col] + uh[u_row, u_col - 1]
                                + uh[u_row, u_col + 1] - h * (ph[u_row, u_col] - ph[u_row, u_col - 1])
                                + h ** 2 * f1h[u_row, u_col]) / 4
            u_row = np.arange(2, uNrow - 1, 2)
            u_col = np.arange(2, uNcol - 1, 2)
            uh[u_row, u_col] = (uh[u_row - 1, u_col] + uh[u_row + 1, u_col] + uh[u_row, u_col - 1]
                                + uh[u_row, u_col + 1] - h * (ph[u_row, u_col] - ph[u_row, u_col - 1])
                                + h ** 2 * f1h[u_row, u_col]) / 4
            # case2, (black points): mod(u_row + u_col, 2) == 1
            u_row = np.arange(1, uNrow - 1, 2)
            u_col = np.arange(2, uNcol - 1, 2)
            uh[u_row, u_col] = (uh[u_row - 1, u_col] + uh[u_row + 1, u_col] + uh[u_row, u_col - 1]
                                + uh[u_row, u_col + 1] - h * (ph[u_row, u_col] - ph[u_row, u_col - 1])
                                + h ** 2 * f1h[u_row, u_col]) / 4
            u_row = np.arange(2, uNrow - 1, 2)
            u_col = np.arange(1, uNcol - 1, 2)
            uh[u_row, u_col] = (uh[u_row - 1, u_col] + uh[u_row + 1, u_col] + uh[u_row, u_col - 1]
                                + uh[u_row, u_col + 1] - h * (ph[u_row, u_col] - ph[u_row, u_col - 1])
                                + h ** 2 * f1h[u_row, u_col]) / 4

            # --- vh ---
            # vh, treat boundary, only treat left and right boundary
            v_row = np.arange(1, vNrow - 1)
            vh[v_row, 0] = (2 * vhLef[v_row, 0] + vh[v_row - 1, 0] + vh[v_row + 1, 0] + vh[v_row, 1]
                            - h * (ph[v_row - 1, 0] - ph[v_row, 0]) + h ** 2 * f2h[v_row, 0]) / 5
            vh[v_row, -1] = (2 * vhRig[v_row, 0] + vh[v_row - 1, -1] + vh[v_row + 1, -1] + vh[v_row, -2]
                             - h * (ph[v_row - 1, -1] - ph[v_row, -1]) + h ** 2 * f2h[v_row, -1]) / 5

            # vh, red-black iteration
            # case1, (red points): mod(v_row + v_col, 2) == 0
            v_row = np.arange(1, vNrow - 1, 2)
            v_col = np.arange(1, vNcol - 1, 2)
            vh[v_row, v_col] = (vh[v_row - 1, v_col] + vh[v_row + 1, v_col] + vh[v_row, v_col - 1]
                                + vh[v_row, v_col + 1] - h * (ph[v_row - 1, v_col] - ph[v_row, v_col])
                                + h ** 2 * f2h[v_row, v_col]) / 4
            v_row = np.arange(2, vNrow - 1, 2)
            v_col = np.arange(2, vNcol - 1, 2)
            vh[v_row, v_col] = (vh[v_row - 1, v_col] + vh[v_row + 1, v_col] + vh[v_row, v_col - 1]
                                + vh[v_row, v_col + 1] - h * (ph[v_row - 1, v_col] - ph[v_row, v_col])
                                + h ** 2 * f2h[v_row, v_col]) / 4
            # case2, (black points): mod(v_row + v_col, 2) == 1
            v_row = np.arange(1, vNrow - 1, 2)
            v_col = np.arange(2, vNcol - 1, 2)
            vh[v_row, v_col] = (vh[v_row - 1, v_col] + vh[v_row + 1, v_col] + vh[v_row, v_col - 1]
                                + vh[v_row, v_col + 1] - h * (ph[v_row - 1, v_col] - ph[v_row, v_col])
                                + h ** 2 * f2h[v_row, v_col]) / 4
            v_row = np.arange(2, vNrow - 1, 2)
            v_col = np.arange(1, vNcol - 1, 2)
            vh[v_row, v_col] = (vh[v_row - 1, v_col] + vh[v_row + 1, v_col] + vh[v_row, v_col - 1]
                                + vh[v_row, v_col + 1] - h * (ph[v_row - 1, v_col] - ph[v_row, v_col])
                                + h ** 2 * f2h[v_row, v_col]) / 4

        # --- ---
        # Step2: Distributive relaxation of velocity and pressure
        # define the rp = gh + div([u,v]^T)
        p_row = np.arange(0, pNrow)
        p_col = np.arange(0, pNcol)
        rp = np.zeros((pNrow, pNcol))
        rp[p_row, p_col] = gh[p_row, p_col] + (uh[p_row, p_col + 1] - uh[p_row, p_col]) / h + (
                    vh[p_row, p_col] - vh[p_row + 1, p_col]) / h

        # Gauss-Seidel smoother, solve Ap * dp = rp
        dp = np.zeros((pNrow, pNcol))
        maxIt = 1000
        tol = 1e-3
        for ite in range(0, maxIt):
            dp_temp = dp

            # --- ---
            # interior points
            p_row = np.arange(1, pNrow - 1)
            p_col = np.arange(1, pNcol - 1)
            dp[p_row, p_col] = (h ** 2 * rp[p_row, p_col] + dp[p_row - 1, p_col] + dp[p_row + 1, p_col] + dp[
                p_row, p_col - 1] + dp[p_row, p_col + 1]) / 4
            p_row = np.arange(2, pNrow - 1)
            p_col = np.arange(2, pNcol - 1)
            dp[p_row, p_col] = (h ** 2 * rp[p_row, p_col] + dp[p_row - 1, p_col] + dp[p_row + 1, p_col] + dp[
                p_row, p_col - 1] + dp[p_row, p_col + 1]) / 4
            p_row = np.arange(1, pNrow - 1)
            p_col = np.arange(2, pNcol - 1)
            dp[p_row, p_col] = (h ** 2 * rp[p_row, p_col] + dp[p_row - 1, p_col] + dp[p_row + 1, p_col] + dp[
                p_row, p_col - 1] + dp[p_row, p_col + 1]) / 4
            p_row = np.arange(2, pNrow - 1)
            p_col = np.arange(1, pNcol - 1)
            dp[p_row, p_col] = (h ** 2 * rp[p_row, p_col] + dp[p_row - 1, p_col] + dp[p_row + 1, p_col] + dp[
                p_row, p_col - 1] + dp[p_row, p_col + 1]) / 4

            # --- ---
            # boundary points but no corner points
            # left-, right- boundary
            p_row = np.arange(1, pNrow - 1)
            dp[p_row, 0] = (h ** 2 * rp[p_row, 0] + dp[p_row + 1, 0] + dp[p_row - 1, 0] + dp[p_row, 1]) / 3
            dp[p_row, -1] = (h ** 2 * rp[p_row, -1] + dp[p_row + 1, -1] + dp[p_row - 1, -1] + dp[p_row, -2]) / 3
            # top-, bottom- boundary
            p_col = np.arange(1, pNcol - 1)
            dp[0, p_col] = (h ** 2 * rp[0, p_col] + dp[0, p_col + 1] + dp[0, p_col - 1] + dp[1, p_col]) / 3
            dp[-1, p_col] = (h ** 2 * rp[-1, p_col] + dp[-1, p_col + 1] + dp[-1, p_col - 1] + dp[-2, p_col]) / 3

            # --- ---
            # corner points
            dp[0, 0] = (h ** 2 * rp[0, 0] + dp[1, 0] + dp[0, 1]) / 2
            dp[-1, 0] = (h ** 2 * rp[-1, 0] + dp[-2, -1] + dp[-1, -2]) / 2
            dp[0, -1] = (h ** 2 * rp[0, -1] + dp[1, -1] + dp[0, -2]) / 2
            dp[-1, -1] = (h ** 2 * rp[-1, -1] + dp[-2, -1] + dp[-2, -1]) / 2

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
        u_col = np.arange(1, uNcol)
        uh[u_row, u_col] = uh[u_row, u_col] + (dp[u_row, u_col] - dp[u_row, u_col - 1]) / h
        v_row = np.arange(1, vNrow)
        v_col = np.arange(0, vNcol)
        vh[v_row, v_col] = vh[v_row, v_col] + (dp[v_row - 1, v_col] - dp[v_row, v_col]) / h

        # --- ---
        # update ph: ph^{k+1} = ph^{k} -  Ap * dp
        # interior nodes
        p_row = np.arange(1, pNrow - 1)
        p_col = np.arange(1, pNcol - 1)
        ph[p_row, p_col] = ph[p_row, p_col] - (
                4 * dp[p_row, p_col] - dp[p_row - 1, p_col] - dp[p_row + 1, p_col] - dp[p_row, p_col - 1]
                - dp[p_row, p_col + 1]) / (h ** 2)

        # boundary nodes but not corner nodes
        ph[p_row, 0] = ph[p_row, 0] - (3 * dp[p_row, 0] - dp[p_row + 1, 0] - dp[p_row - 1, 0] - dp[p_row, 1]) / (h ** 2)
        ph[p_row, -1] = ph[p_row, -1] - (3 * dp[p_row, -1] - dp[p_row + 1, -1] - dp[p_row - 1, -1] - dp[p_row, -2]) / (h ** 2)
        ph[0, p_col] = ph[0, p_col] - (3 * dp[0, p_col] - dp[0, p_col + 1] - dp[0, p_col - 1] - dp[1, p_col]) / (h ** 2)
        ph[-1, p_col] = ph[-1, p_col] - (3 * dp[-1, p_col] - dp[-1, p_col + 1] - dp[-1, p_col - 1] - dp[-2, p_col]) / (h ** 2)

        # corner nodes
        ph[0, 0] = ph[0, 0] - (2 * dp[0, 0] - dp[1, 0] - dp[0, 1]) / (h ** 2)
        ph[0, -1] = ph[0, -1] - (2 * dp[0, -1] - dp[1, -1] - dp[0, -2]) / (h ** 2)
        ph[-1, -1] = ph[-1, -1] - (2 * dp[-1, -1] - dp[-2, -1] - dp[-1, -2]) / (h ** 2)
        ph[-1, 0] = ph[-1, 0] - (2 * dp[-1, 0] - dp[-2, 0] - dp[0, -2]) / (h ** 2)

        # --- ---
        # return
        return uh, vh, ph





















