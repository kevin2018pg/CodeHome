# -*- coding: utf-8 -*-
# @Time    : 2022/8/4 20:07
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 最长上升子序列


class Solution:
    def lengthOfLIS(self, nums):
        if not nums:
            return 0
        dp = [1] * len(nums)  # dp初始状态：都为1，代表每个元素都可以是单独的子序列
        for i in range(len(nums)):  # 遍历数组
            for j in range(i):  # j 属于[0,i-1]
                if nums[j] < nums[i]:  # 满足递增条件：如果要求非严格递增，将此行 '<' 改为 '<=' 即可。
                    dp[i] = max(dp[i], dp[j] + 1)  # 转移方程：dp[i]的值代表nums以nums[i]结尾的最长子序列长度
        return max(dp)
