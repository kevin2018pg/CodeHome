# -*- coding: utf-8 -*-
# @Time    : 2022/8/8 21:26
# @Author  : kevin
# @Version : python 3.7
# @Desc    : number-of-longest-increasing-subsequence
# 最大上升子序列个数
class Solution:
    def findNumberOfLIS(self, nums):
        n = len(nums)
        if n == 1: return 1

        dp_length = [1] * n
        dp_count = [1] * n
        max_length = 1
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    if dp_length[j] + 1 > dp_length[i]:  # 说明最长递增子序列的长度增加了，dp[i]=dp[j]+1，长度增加，数量不变count[i]=count[j]
                        dp_length[i] = dp_length[j] + 1
                        dp_count[i] = dp_count[j]
                    elif dp_length[j] + 1 == dp_length[i]:  # 说明最长递增子序列的长度并没有增加，但是出现了长度一样的情况，数量增加count[i]+=count[j]
                        dp_count[i] += dp_count[j]
            max_length = max(max_length, dp_length[i])

        res = 0
        for i in range(n):
            if dp_length[i] == max_length:
                res += dp_count[i]
        return res
