#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_decorator.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jun 17, 2021
# ---


from functools import wraps


def decorator_name(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        print("can_run:", can_run)
        if not can_run:
            print("args, kwargs: ", args, kwargs)
            return "Function will not run"
        return f(args[0]+0.1, **kwargs)

    return decorated


@decorator_name
def func(a):
    print(a)
    return "Function is running"


can_run = True
print(func(1))
# Output: Function is running

can_run = False
print(func(2))
# Output: Function will not run

