# -*- coding: utf-8 -*-
# @Time    : 2022/8/5 14:45
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 使用最小花费爬楼梯


class Solution:
    def minCostClimbingStairs(self, cost):
        dp = [0] * (len(cost))
        # 初始化dp[0]和dp[1]，一阶和二阶台阶花费
        dp[0] = cost[0]
        dp[1] = cost[1]
        for i in range(2, len(cost)):  # 从三阶台阶开始遍历
            dp[i] = min(dp[i - 1], dp[i - 2]) + cost[i]  # 当前阶花费是由前一阶或前两阶得到，取最小花费；需要加上爬上当前台阶的耗费
        return min(dp[len(cost) - 1], dp[len(cost) - 2])  # 注意最后一步可以理解为不用花费，所以取倒数第一步，第二步的最少值
