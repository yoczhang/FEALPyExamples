#!/anaconda3/envs/FEALPy/bin python3.9
# -*- coding: utf-8 -*-
# ---
# @File: totest_decorator.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Time: 2025/2/11
# ---
import time


def cache_result(func):
    cached_data = {}  # 用字典存储缓存结果

    def wrapper(n):
        if n not in cached_data:  # 如果结果未缓存
            print(f"计算 {n} 的结果...")
            cached_data[n] = func(n)
        else:
            print(f"从缓存中读取 {n} 的结果")
        return cached_data[n]
    return wrapper


@cache_result
def factorial(n):
    time.sleep(1)  # 模拟复杂计算
    return 1 if n == 0 else n * factorial(n-1)


print(factorial(5))  # 第一次计算, 耗时较长
print("---------------------")
print(factorial.__closure__[0].cell_contents)  # __closure__[0]: [0] 表示 wapper 外部只有 1 个变量, 当有多个变量时, 可以用 [1] [2] ... 来访问
print("---------------------")
print(factorial(5))  # 第二次直接读缓存, 瞬间返回
