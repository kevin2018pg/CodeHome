# -*- coding: utf-8 -*-
# @Time : 2020/6/14 22:44
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : 0613ClimbingStairs
# @Description: 爬楼梯

# 爬楼梯解法
import functools


class Solution:
    @functools.lru_cache()  # 缓存装饰器
    def climbStairs_s1(self, n: int) -> int:
        if n == 1: return 1
        if n == 2: return 2
        return self.climbStairs_s1(n - 1) + self.climbStairs_s1(n - 2)

    # 记忆数组法，传统DP
    def climbStairs_s2(self, n: int) -> int:
        dp = {}
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

    # 最优化DP，状态转移法
    def climbStairs_s3(self, n: int) -> int:
        if n == 0 or n == 1:
            return 1
        if n == 2:
            return 2
        a, b = 1, 2
        for i in range(3, n + 1):
            a, b = b, a + b
        return b

    # 记忆性递归1，也叫备忘录递归，将重复计算的值存入字典
    def climbStairs_s4(self, n: int) -> int:
        hashmap = {}
        hashmap[0] = 0
        hashmap[1] = 1

        def helper(n):
            if n <= 2:
                return n
            if n in hashmap:
                return hashmap[n]
            res = helper(n - 1) + helper(n - 2)
            hashmap[n] = res
            return res

        return helper(n)
