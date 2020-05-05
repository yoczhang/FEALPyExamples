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

    def myprintA(self):
        BB = totest__new__B(self)
        return BB

    def myprintB(self, content=None):
        if content is None:
            print('in AAA')
        else:
            print(content)



