# -*- coding: utf-8 -*-
# @Time    : 2022/8/10 21:23
# @Author  : kevin
# @Version : python 3.7
# @Desc    : find-first-and-last-position-of-element-in-sorted-array

# 返回数组中指定数字下标的起止点
# 比如[1,2,3,6,6,7]，返回[3,4]
# 二分查找到第一个+循环再查找
class Solution:
    def searchRange(self, nums, target):
        def binarySearch(nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = left + (right - left) // 2
                if nums[mid] > target:
                    right = mid - 1
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    return mid
            return -1

        index = binarySearch(nums, target)
        if index == -1: return [-1, -1]  # nums 中不存在 target，直接返回 {-1, -1}
        # nums 中存在 targe，则左右滑动指针，来找到符合题意的区间
        left, right = index, index
        # 向左滑动，找左边界
        while left - 1 >= 0 and nums[left - 1] == target: left -= 1
        # 向右滑动，找右边界
        while right + 1 < len(nums) and nums[right + 1] == target: right += 1
        return [left, right]
