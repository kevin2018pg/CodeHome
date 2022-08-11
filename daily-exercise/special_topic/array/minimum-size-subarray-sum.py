# -*- coding: utf-8 -*-
# @Time    : 2022/8/11 19:58
# @Author  : kevin
# @Version : python 3.7
# @Desc    : minimum-size-subarray-sum
# 长度最小的子数组：滑动窗口

class Solution:
    def minSubArrayLen(self, target, nums):
        n = len(nums)
        if not nums or n == 0:
            return 0
        slow = 0
        lengthMin = len(nums) + 1
        numSum = 0
        for fast in range(n):
            numSum += nums[fast]
            while numSum >= target:
                lengthMin = min(lengthMin, fast - slow + 1)
                numSum -= nums[slow]
                slow += 1
        if lengthMin == len(nums) + 1:
            return 0
        return lengthMin


test = Solution()
l = test.minSubArrayLen(7, [2, 3, 1, 2, 4, 3])
print(l)
