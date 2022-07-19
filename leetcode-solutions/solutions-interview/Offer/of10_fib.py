'''
Author: your name
Date: 2020-11-18 13:16:40
LastEditTime: 2020-11-18 18:01:30
LastEditors: Please set LastEditors
Description: 剑指 Offer 10- I. 斐波那契数列

写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项。斐波那契数列的定义如下：

F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。


FilePath: \learningflow\Review\OfferSolution\of10_fib.py
'''


class Solution:
    # 传统递归方法会超时，需要加入@functools.lru_cache装饰器，实现缓存机制
    import functools

    @functools.lru_cache
    def fib_01(self, n: int) -> int:
        if n < 2:
            return n
        return (self.fib(n - 1) + self.fib(n - 2)) % (10 ** 9 + 7)

    # 最优化DP
    def fib_02(self, n: int) -> int:
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a % 1000000007

    # 传统DP
    def fib_03(self, n: int) -> int:
        dp = {}
        dp[0] = 0
        dp[1] = 1
        if n >= 2:
            for i in range(2, n + 1):
                dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n] % 1000000007

    # 记忆性递归，存储值，减少递归次数
    def fib_04(self, n: int) -> int:
        map = {}
        map[0] = 0
        map[1] = 1

        def helper(n):
            if n < 2:
                return n
            if n in map:
                return map[n]
            res = helper(n - 1) + helper(n - 2)
            map[n] = res
            return res

        return helper(n) % 1000000007
