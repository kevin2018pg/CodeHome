# -*- coding: utf-8 -*-
# @Time    : 2022/8/11 11:13
# @Author  : kevin
# @Version : python 3.7
# @Desc    : remove-duplicates-from-sorted-array

#  删除有序数组中的重复项
# 快慢指针：
class Solution:
    def removeDuplicates(self, nums):
        if not nums:
            return 0
        n = len(nums)
        slow = 1
        for fast in range(1, n):
            if nums[fast] != nums[fast - 1]:
                nums[slow] = nums[fast]
                slow += 1
        return slow
