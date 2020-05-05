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


class totest__new__B:
    def __new__(cls, new_A):
        self = new_A
        return self

    def myprintB(self, content='in BBB'):
        return self.myprintB(content)
