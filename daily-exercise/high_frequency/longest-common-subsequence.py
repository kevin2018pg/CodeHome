# -*- coding: utf-8 -*-
# @Time    : 2022/8/5 9:18
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 最长公共子序列

# 二维动态规划
class Solution:
    def longestCommonSubsequence(self, text1, text2):
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # 状态转移方程，分两种情况
                if text1[i - 1] == text2[j - 1]:  # 两个字符最后一位相等，最长序列增加1
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:  # 不相等长度最少也是dp[i][j] = dp[i - 1][j - 1]，不相等可得到dp[i][j-1]和dp[i-1][j]>=dp[i][j]，所以取二者较大的即可
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]  # 返回的值最长序列，及i=m，j=n
