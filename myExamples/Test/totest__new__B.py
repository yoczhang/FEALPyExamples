#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: totest__new__B.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: May 05, 2020
# ---


import numpy as np


class totest__new__B:
    def __init__(self, spaceA, uh):
        self.age = spaceA.age
        self.uh = uh

    def add_one(self):
        uh = self.uh
        age = self.age
        uh[0] = 1
        return age+1

