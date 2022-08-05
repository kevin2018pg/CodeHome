# -*- coding: utf-8 -*-
# @Time    : 2022/8/5 20:27
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 机器人不同路径II（有障碍物版）

# 第一行和第一列初始化为0，障碍物之前都是1；但如果(i, 0)/(0, j) 这条边有了障碍之后，障碍之后（包括障碍）都是走不到的位置了，所以障碍之后的dp[i][0]/dp[0][j]应该还是初始值0

class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid):
        row = len(obstacleGrid)
        col = len(obstacleGrid[0])
        dp = [[0] * col for _ in range(row)]
        dp[0][0] = 1 if obstacleGrid[0][0] != 1 else 0
        if dp[0][0] == 0:  # 如果第一个格子就是障碍，return 0
            return 0
        # 第一行：障碍物之前都为1
        for i in range(1, col):
            if obstacleGrid[0][i] != 1:  # dp[0][i]出现障碍物就切断，全部变成0，否则为1
                dp[0][i] = dp[0][i - 1]
            # else:
            #     break   # 初始化为0，可提前结束
        # 第一列：障碍物之前都为1
        for i in range(1, row):
            if obstacleGrid[i][0] != 1:
                dp[i][0] = dp[i - 1][0]  # dp[i][0]出现障碍物就切断，全部变成0，否则为1
            # else:
            #     break
        for i in range(1, row):
            for j in range(1, col):
                if obstacleGrid[i][j] != 1:  # 越过障碍物，障碍物为0
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[row - 1][col - 1]
