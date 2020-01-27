#!/anaconda3/envs/FEALPy/bin python3.7
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: YC_test_new_func.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Jan 27, 2020
# ---



class Person(object):
    """Silly Person"""

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return '<Person: %s(%s)>' % (self.name, self.age)


if __name__ == '__main__':
    piglei = Person('piglei', 24)
    print(piglei)

