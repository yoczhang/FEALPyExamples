#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_Inheritance.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Dec 13, 2021
# ---


class FatherA:
    def __init__(self):
        print('init action in father class A')
        self.print_x()
        print("*******************")

    def print_x(self):
        print("testV2")


class SubClassB(FatherA):
    def __init__(self):
        print('init action in subclass B')

        super(SubClassB, self).__init__()

    def print_x(self):
        super(SubClassB, self).print_x()
        print("testV1")


if __name__ == '__main__':
    b = SubClassB()
    b.print_x()