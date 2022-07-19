# -*- coding: utf-8 -*-
# @Time : 2020/6/8 21:04
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : fibonacciSequence
# @Description: 斐波那契数列


# 记忆性递归1，也叫备忘录递归，将重复计算的值存入字典
class Solution_s10:
    def fib(self, n: int) -> int:
        hashmap = {}
        hashmap[0] = 0
        hashmap[1] = 1
        mod = 10 ** 9 + 7

        def helper(n):
            if n < 2:
                return n
            if n in hashmap:
                return hashmap[n]
            res = helper(n - 1) + helper(n - 2)
            hashmap[n] = res
            return res

        return helper(n) % mod


# 超出时间限制
class Solution_s11:
    def fib(self, n: int) -> int:
        if n < 2:
            return n
        records = [-1 for i in range(n + 1)]
        if records[n] == -1:
            records[n] = self.fib(n - 1) + self.fib(n - 2)
        return records[n] % 1000000007


import functools


# 传统递归方法会超时，需要加入@functools.lru_cache装饰器，实现缓存机制
class Solution_s2:
    @functools.lru_cache
    def fib(self, n: int) -> int:
        if n < 2:
            return n
        return (self.fib(n - 1) + self.fib(n - 2)) % (10 ** 9 + 7)


# 传统DP方法(会超时)
class Solution_s3:
    def fib(self, n: int) -> int:
        dp = {}
        dp[0] = 0
        dp[1] = 1
        if n >= 2:
            for i in range(2, n + 1):
                dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n] % 1000000007


# 最优化DP方法
# 由于 Python 中整形数字的大小限制 取决计算机的内存 （可理解为无限大），因此可不考虑大数越界问题。
class Solution_s4:
    def fib(self, n: int) -> int:
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a % 1000000007
