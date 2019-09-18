#!/anaconda3/envs/FEALPy/bin python3.7
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: solutionData.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Sep 18, 2019
# ---


class solutionStokesData:
    def __init__(self):
        pass

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
