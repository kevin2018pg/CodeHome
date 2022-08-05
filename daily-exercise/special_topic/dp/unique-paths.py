# -*- coding: utf-8 -*-
# @Time    : 2022/8/5 19:47
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 不同路径


class Solution:
    def uniquePaths(self, m, n):
        # dp[i][j] ：表示从（0 ，0）出发，到(i, j)有dp[i][j]条不同的路径
        # 初始化：dp[0][j]和dp[i][0]一定都=1，因为机器人只有这一条路径
        dp = [[0] * n for _ in range(m)]
        # 为了方便可以直接初始化全部为1，不影响状态转移计算
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        # 状态转移：dp[i][j]，只能由两个方向来推导出来，即dp[i - 1][j] 和 dp[i][j - 1]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i][j - 1] + dp[i - 1][j]
        return dp[m - 1][n - 1]
