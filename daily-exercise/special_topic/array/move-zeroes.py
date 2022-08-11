# -*- coding: utf-8 -*-
# @Time    : 2022/8/11 13:07
# @Author  : kevin
# @Version : python 3.7
# @Desc    : move-zeroes

# 移动0至末尾
# 移动0到末尾，等同于第5第6题（快慢指针，寻找非元素值、非重复值、非0值）
class Solution:
    def moveZeroes(self, nums):
        n = len(nums)
        slow = 0
        for fast in range(n):
            if nums[fast] != 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1
