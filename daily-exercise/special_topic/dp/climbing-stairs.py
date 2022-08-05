# -*- coding: utf-8 -*-
# @Time    : 2022/8/5 14:15
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 爬楼梯


class Solution:
    def climbStairs(self, n):
        # 不考虑0级台阶，1级作为特殊情况，初始化1级和2级台阶，dp[n] = dp[n - 1] + dp[n - 2]，返回最终n即可
        if n == 1:
            return n
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]
