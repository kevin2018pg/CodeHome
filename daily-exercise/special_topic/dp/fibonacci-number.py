# -*- coding: utf-8 -*-
# @Time    : 2022/8/5 13:12
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 斐波那契数列

class Solution:
    def fib(self, n: int) -> int:
        if n < 2:
            return n
        a, b = 0, 1
        for i in range(1, n):  # 剩余n-1次遍历
            a, b = b, a + b
        return b


# 递归实现
class Solution1:
    def fib(self, n: int) -> int:
        if n < 2:
            return n
        return self.fib(n - 1) + self.fib(n - 2)
