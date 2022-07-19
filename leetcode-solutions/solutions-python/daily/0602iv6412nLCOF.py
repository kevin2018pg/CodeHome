# -*- coding: utf-8 -*-
# @Time : 2020/6/2 8:51
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : iv6412nLCOF
# @Description: 求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句。


"""平均法，不满足条件"""


def sumNums_s1(n):
    return (1 + n) * n // 2


"""迭代法，不满足条件"""


def sumNums_s20(n):
    res = 0
    for i in range(1, n + 1):
        res += i
    return res


def sumNums_s21(n):
    """对生成器求和"""
    return sum(range(1, n + 1))


"""递归法，需要改写（依赖and关系的短路法则）"""


def sumNums_s30(n):
    if n == 1: return 1
    n += sumNums_s30(n - 1)
    return n


def sumNums_s31(n):
    return n != 0 and n + sumNums_s31(n - 1)


def sumNums_s32(n):
    return n and n + sumNums_s32(n - 1)  # n=0时为false，后面就不执行了退出递归 print(0 and 1 + 1)=0


def sumNums_s33(n):
    res = 0
    n > 1 and sumNums_s33(n - 1)
    res += n
    return res
