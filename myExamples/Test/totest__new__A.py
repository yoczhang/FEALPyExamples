#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: totest__new__A.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: May 05, 2020
# ---


import numpy as np
from totest__new__B import totest__new__B


class totest__new__A(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.uh = np.zeros(5)

    def call__new__B(self):
        uh = self.uh
        BB = totest__new__B(self, uh)
        ageplus = BB.add_one()
        return ageplus



