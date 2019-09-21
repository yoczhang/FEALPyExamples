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

    def solution(self, p, entity):
        """ The exact solution
        """
        x = p[..., 0]
        y = p[..., 1]

        if entity is 'u':
            uval = x ** 2 * (x - 1) ** 2 * y * (y - 1) * (2 * y - 1)
            return uval
        elif entity is 'v':
            vval = -x * (x - 1) * (2. * x - 1) * y ** 2. * (y - 1) ** 2
            return vval
        elif entity is 'p':
            pval = (2 * x - 1) * (2 * y - 1)
            return pval
        else:
            raise ValueError("There is no '{}' type!".format(entity))

    def source(self, p, entity):
        nu = 1
        x = p[..., 0]
        y = p[..., 1]

        if entity is 'f1':
            f1val = 2 * (2 * y - 1) * (
                    - 3 * nu * x ** 4 + 6 * nu * x ** 3 - 6 * nu * x ** 2 * y ** 2 + 6 * nu * x ** 2 * y - 3 * nu * x ** 2
                    + 6 * nu * x * y ** 2 - 6 * nu * x * y - nu * y ** 2 + nu * y + 1)
            return f1val
        elif entity is 'f2':
            f2val = 2 * (2 * x - 1) * (
                    6 * nu * x ** 2 * y ** 2 - 6 * nu * x ** 2 * y + nu * x ** 2 - 6 * nu * x * y ** 2 + 6 * nu * x * y
                    - nu * x + 3 * nu * y ** 4 - 6 * nu * y ** 3 + 3 * nu * y ** 2 + 1)
            return f2val
        elif entity is 'g':
            return 0. * x
        else:
            raise ValueError("There is no '{}' type!".format(entity))
