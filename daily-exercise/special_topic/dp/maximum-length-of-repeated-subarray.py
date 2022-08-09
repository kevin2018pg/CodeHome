# -*- coding: utf-8 -*-
# @Time    : 2022/8/9 12:43
# @Author  : kevin
# @Version : python 3.7
# @Desc    : maximum-length-of-repeated-subarray
class Solution:
    def findLength(self, A, B):
        dp = [[0] * (len(B) + 1) for _ in range(len(A) + 1)]  # 等同于机器人不同路径数
        result = 0
        for i in range(1, len(A) + 1):
            for j in range(1, len(B) + 1):
                if A[i - 1] == B[j - 1]:  # 判断相等
                    dp[i][j] = dp[i - 1][j - 1] + 1  # 当前数由前一状态得到
                result = max(result, dp[i][j])
        return result  # 不能返回索引最大，因为有相等条件限制，所以要保存max值
