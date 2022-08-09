# -*- coding: utf-8 -*-
# @Time    : 2022/8/9 17:16
# @Author  : kevin
# @Version : python 3.7
# @Desc    : uncrossed-lines

# 此题完全等于最长公共子序列

class Solution:
    def maxUncrossedLines(self, A, B):
        m, n = len(A), len(B)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if A[i - 1] == B[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]
