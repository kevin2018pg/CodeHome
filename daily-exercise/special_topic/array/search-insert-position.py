# -*- coding: utf-8 -*-
# @Time    : 2022/8/11 9:33
# @Author  : kevin
# @Version : python 3.7
# @Desc    : search-insert-position

# 寻找插入位置
class Solution:
    def searchInsert(self, nums, target):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                return mid
        return right + 1  # return left
