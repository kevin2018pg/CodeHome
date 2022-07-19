# -*- coding: utf-8 -*-
# @Time : 2020/6/15 20:20
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : frogjumps on the steps
# @Description: 青蛙跳台阶问题


"""青蛙跳台阶问题和爬楼梯最大的区别就是0级台阶也有1种跳法，还需要取模
取模为了防止溢出，计算更多的数据
这是最骚的
"""
import functools


class Solution:
    @functools.lru_cache()  # 缓存装饰器
    def numWays_s1(self, n: int) -> int:
        if n <= 1: return 1
        if n == 2: return 2
        return (self.numWays_s1(n - 1) + self.numWays_s1(n - 2)) % 1000000007

    # 记忆数组法，传统DP
    def numWays_s2(self, n: int) -> int:
        dp = {}
        dp[0] = 1
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n] % 1000000007

    # 最优化DP，状态转移法
    def numWays_s3(self, n: int) -> int:
        if n <= 1:
            return 1
        if n == 2:
            return 2
        a, b = 1, 2
        for i in range(3, n + 1):
            a, b = b, a + b
        return b % 1000000007

    # 记忆性递归1，也叫备忘录递归，将重复计算的值存入字典
    def numWays_s4(self, n: int) -> int:
        hashmap = {}
        hashmap[0] = 0
        hashmap[1] = 1

        def helper(n):
            if n <= 1:
                return 1
            if n == 2:
                return 2
            if n in hashmap:
                return hashmap[n]
            res = helper(n - 1) + helper(n - 2)
            hashmap[n] = res
            return res

        return helper(n)
